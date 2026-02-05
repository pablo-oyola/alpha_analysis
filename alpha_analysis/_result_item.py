import unyt
import xarray as xr
import os
import numpy as np
from pathlib import Path
from typing import Union

import a5py
import desc
from a5py.ascot5io.dist import DistData
from a5py.physlib import parseunits
from ._logger import get_logger
from ._dist5d_epitch import transform2Epitch
from ._git import get_git_info

# Let's try to load the MPI controller
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    MPI_ENABLED = True
except ImportError:
    comm = None
    rank = 0
    MPI_ENABLED = False

logger = get_logger(__name__)         

# --- Main class to parse the results ---
class ResultItem:
    def __init__(self, filepath: Union[str, Path], descfn: Union[str, Path]):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist.")
        if not os.path.isfile(descfn):
            raise FileNotFoundError(f"DESC file {descfn} does not exist.")
        
        self.filepath = Path(filepath)
        self.descfn = Path(descfn)
        self.a5 = a5py.Ascot(self.filepath, create=False)

        try:
            qid = self.data.results.attrs['active']
        except:
            raise KeyError("Could not find 'active' attribute in results: \
                       maybe there is not results in the file?")
    
        # == Particle losses ==
        # We don't need to load them here, to avoid overhead.

        # == Distribution loading ==
        if 'distrho5d' not in self.results:
            logger.warning(f"'distrho5d' not found in results.")
            self._dist5d_flag = False
        else:
            self._dist5d_flag = True

            self.data = xr.open_datatree(self.filepath)
            self.results = self.data.results['run_' + str(qid)]
            self.distdata_on_disk = self.results['distrho5d']
            self.ntime = self.distdata_on_disk.ordinate.shape[-2]

            self.n_dims = self.distdata_on_disk.ordinate.ndim
            self.time_axis = self.n_dims - 2  # as used above

            self.abscissas = {}
            names = ['r', 'phi', 'z', 'ppar', 'pperp', 'time', 'charge']
            units = ['m', 'deg', 'm', 'kg*m/s', 'kg*m/s', 's', 'e']
            for i in range(len(self.distdata_on_disk.ordinate.shape)-1):
                self.abscissas[names[i]] = self.distdata_on_disk['abscissa_vec_%02d' % (i+1)][:].values * unyt.Unit(units[i])

    def load_losses(self, loss_convergence: bool=True, nmc: int=100, 
                    nmrk_step: int=10000) -> xr.Dataset:
        """
        Load the particle losses and returns a Dataset with all the
        relevant information.

        When the convergence is enabled, a MC analysis of the 
        losses is performed to estimate the convergence and 
        uncertainties in the losses.

        Parameters
        ----------
        loss_convergence : bool, optional
            Whether to perform the convergence analysis, by default True
        nmc : int, optional
            Number of Monte Carlo samples for the convergence analysis,
            by default 100
        nmrk_step : int, optional
            Number of markers step for the convergence analysis,
            by default 10000
        """
        if rank == 0:
            # We need to get the loss information.
            ids = self.a5.data.active.getstate("ids", endcond=["none", "wall", "rhomax"])
            mrk = self.a5.data.active.getstate_markers("gc", ids=ids)

            # Checking initial total power.
            energy_ini, weight_ini, ids_ini = self.a5.data.active.getstate("ekin", "weight", "ids", state='ini')

            # Computing total lost power with respect to original power.
            power_ini  = np.sum(energy_ini * weight_ini).to('MW')
            power_loss = np.sum(mrk['energy'] * mrk['weight']).to('MW')

            fraction_lost = power_loss / power_ini

            # We generate now the datatree.
            dset = xr.DataTree()
            dset['initial'] = xr.Dataset()
            dset.initial['energy'] = xr.DataArray(energy_ini.to('J').value, 
                                                dims=['ids'], 
                                                coords={'ids': ids_ini},
                                                attrs={'units': 'J'})
            dset.initial['weight'] = xr.DataArray(weight_ini.value, 
                                                dims=['ids'], 
                                                attrs={'units': str(weight_ini.units)})
            dset.initial.attrs['total_power'] = power_ini.to('MW').value

            dset['losses'] = xr.Dataset()
            dset.losses['energy'] = xr.DataArray(mrk['energy'].to('J').value, 
                                                dims=['ids'],
                                                coords={'ids': mrk['ids']},
                                                attrs={'units': 'J'})
            dset.losses['weight'] = xr.DataArray(mrk['weight'].value, 
                                                dims=['ids'],
                                                coords={'ids': mrk['ids']},
                                                attrs={'units': str(mrk['weight'].units)})
            dset.losses.attrs['total_power'] = power_loss.to('MW').value
            dset.losses.attrs['fraction_lost'] = fraction_lost.value     

            if loss_convergence:
                # Monte Carlo analysis of the losses convergence.
                nmrk_conv = np.arange(nmrk_step, len(ids)+1, nmrk_step)
                loss_conv = np.zeros((len(nmrk_conv), nmc)) * unyt.dimensionless
                
                for imrk in range(len(nmrk_conv)):
                    imrk_conv = nmrk_conv[imrk]
                    
                    for iboot in range(nmc):
                        if imrk_conv >= len(ids_ini):
                            selected_ids = ids_ini
                        else:
                            selected_ids = np.random.choice(ids_ini, size=imrk_conv, replace=False)

                            # We compute the initial power for these markers.
                            mask = np.isin(ids_ini, selected_ids)
                            energy_ini_sel = energy_ini[mask]
                            weight_ini_sel = weight_ini[mask]
                            power_ini_sel  = np.sum(energy_ini_sel * weight_ini_sel).to('MW')

                            # From these subselection of ids_ini, we find which of them were lost.
                            lost_mask = np.isin(mrk['ids'], selected_ids)
                            energy_lost_sel = mrk['energy'][lost_mask]
                            weight_lost_sel = mrk['weight'][lost_mask]
                            power_lost_sel  = np.sum(energy_lost_sel * weight_lost_sel).to('MW')

                            fraction_lost_sel = power_lost_sel / power_ini_sel
                            loss_conv[imrk, iboot] = fraction_lost_sel

                # We add the convergence information to the datatree.
                dset['loss_conv'] = xr.DataArray(loss_conv,
                                                dims=['nmrk_conv', 'iboot'],
                                                coords={'nmrk_conv': nmrk_conv, 'iboot': np.arange(nmc)},
                                                attrs={'units': 'dimensionless',
                                                    'description': 'Monte Carlo convergence of the losses fraction'})
            
            self.losses = dset
        
        if MPI_ENABLED:
            # We need to broadcast the datatree to all ranks.
            dset = comm.bcast(self.losses if rank == 0 else None, root=0)

        return dset
    
    def get_dist5d(self, time_index: int) -> xr.Dataset:
        """
        Load the 5D distribution at a given time index.

        Parameters
        ----------
        time_index : int
            Time index to load the distribution from.
        """
        if not self._dist5d_flag:
            raise ValueError("Distribution data is not available in this result item.")

        if time_index < 0 or time_index >= self.ntime:
            raise IndexError(f"Time index {time_index} is out of bounds for ntime={self.ntime}.")

        # We create a Dataset with the distribution data and the abscissas.
        idx = [slice(None)] * self.n_dims
        idx[0] = 0
        idx[self.time_axis] = time_index

        # We force the loading here.
        data = self.distdata_on_disk.ordinate[tuple(idx)].values

        # We copy here the abscissas.
        abscissas = {}
        for key in self.abscissas:
            if key == 'time':
                abscissas[key] = self.abscissas[key][time_index:time_index+2]
            else:
                abscissas[key] = self.abscissas[key].copy()
            
        out_dist = DistData(data, **abscissas)

        return out_dist

    @parseunits(Emin='keV', Emax='keV')
    def make_profiles(self, preserve_angles: bool=True, 
                      npitch_bins: int=51,
                      nenergy_bins: int=51,
                      Emin: unyt.unyt_quantity=10*unyt.keV,
                      Emax: unyt.unyt_quantity=3.55*unyt.MeV,
                      nmc: int=100) -> xr.Dataset:
        """
        Computes density and pressure profiles from the 5D distribution.

        This will run over MPI if enabled to accelerate the production, and
        then communicate the results among them all.

        Parameters
        ----------
        preserve_angles : bool, optional
            Whether to preserve the angular dimensions (phi and theta), 
            by default True.
        npitch_bins : int, optional
            Number of pitch bins for the energy-pitch profiles,
            by default 51
        nenergy_bins : int, optional
            Number of energy bins for the energy-pitch profiles,
            by default 51
        nmc: int, optional
            Number of Monte Carlo samples for the energy-pitch transformation,
            by default 100
        """
        # We check first that we have the distribution data.
        if not self._dist5d_flag:
            raise ValueError("Distribution data is not available in this result item.")
        
        # We read from the ASCOT file the number rho grid to compute the volumes.
        if rank == 0:
            opts = self.a5.data.options.active.read()
        
        if MPI_ENABLED:
            opts = comm.bcast(opts if rank == 0 else None, root=0)

        rhomin = float(opts['DIST_MIN_RHO'])
        rhomax = float(opts['DIST_MAX_RHO'])
        nrho   = int(opts['DIST_NBIN_RHO'])

        rho = np.linspace(rhomin, rhomax, nrho+1) * unyt.dimensionless
        
        if rank == 0:
            fam = desc.io.load(self.descfn)
            try:  # if file is an EquilibriaFamily, use final Equilibrium
                eq = fam[-1]
            except:  # file is already an Equilibrium
                eq = fam
            grid = desc.grid.LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, 
                                            NFP=eq.NFP, sym=False)
            data = eq.compute("V(r)", grid=grid)
            vol = np.diff(np.array(grid.compress(data["V(r)"]))) # Volume contained within each rho shell
            vol *= unyt.m**3
        
        if MPI_ENABLED:
            vol = comm.bcast(vol if rank == 0 else None, root=0)
        
        # Computing the ranks to work on each time slice.
        time_indices = np.arange(self.ntime)
        if MPI_ENABLED:
            if len(time_indices) < comm.size:
                logger.warning(f"Number of time indices {len(time_indices)} is less than number of MPI ranks {comm.size}. Some ranks will be idle.")
                if rank < len(time_indices):
                    time_indices = np.array((time_indices[rank],), dtype=int)
                else: 
                    time_indices = np.array([], dtype=int)
            else:
                time_indices = time_indices[rank::comm.size]
            
        
        # We create the empty arrays to store the profiles.
        if preserve_angles:
            nphi = self.abscissas['phi'].shape[0]
            ntheta = self.abscissas['theta'].shape[0]
            shape = (nrho, ntheta, nphi)
        else:
            shape = (nrho,)

        # Reading the Dt
        if rank == 0:
            timemin = float(opts['DIST_MIN_TIME'])
            timemax = float(opts['DIST_MAX_TIME'])
            ntime = int(opts['DIST_NBIN_TIME'])
            Dt = (timemax - timemin) / ntime * unyt.s
        
        density = np.zeros((len(time_indices),) + shape) * unyt.m**-3
        pressure = np.zeros((len(time_indices),) + shape) * unyt.Pa
        fEpitch = np.zeros((len(time_indices), npitch_bins, nenergy_bins)) * 1/unyt.eV * 1/unyt.dimensionless
        
        mass = 4.001506179127 * unyt.amu

        for itime, time_index in enumerate(time_indices):
            dist5d = self.get_dist5d(time_index)
            vars2intg = ['ppar', 'pperp', 'charge']
            if not preserve_angles:
                vars2intg += ['phi', 'theta']
            
            tmp = dist5d.integrate(True, vars2intg)
            density[itime, ...] = (tmp.distribution() / vol * Dt).to('1/m**3')
            del tmp

            # To compute the pressure we need to multiply the distribution
            # by the energy.
            ppara = tmp.abscissa('ppar')
            pperp = tmp.abscissa('pperp')
            energy = 0.5 * (ppara[:, None]**2/mass + pperp[None, :]**2/mass)

            dist2 = dist5d._copy()
            dist2._multiply(energy, 'ppar', 'pperp')
            tmp = dist2.integrate(True, vars2intg)
            pressure[itime, ...] = 2/3 * (tmp.distribution() / vol * Dt).to('Pa')
            del dist2 # Releasing memory.
            del tmp

            # We integrate all the spatial variables to get f(ppar, pperp)
            fmom = dist5d.integrate(True, ['r', 'phi', 'theta', 'charge'])
            fep = transform2Epitch(fmom, Ebins=nenergy_bins, 
                                   pitchbins=npitch_bins, mass=mass,
                                   Emin=Emin, Emax=Emax, Nmc=nmc)
            fEpitch[itime, ...] = (fep.distribution() * Dt).to("1/(eV * dimensionless)")

            del dist5d  # Releasing memory.
            del fmom
            del fep
        
        # We combine the results from all the ranks.
        if MPI_ENABLED:
            payload = {
                "time_indices": time_indices,
                "density": density,
                "pressure": pressure,
                "fEpitch": fEpitch,
            }
            all_payloads = comm.gather(payload, root=0)

            if rank == 0:
                # Allocate global arrays
                density_all  = np.zeros((ntime,) + shape) * unyt.m**-3
                pressure_all = np.zeros((ntime,) + shape) * unyt.Pa
                fEpitch_all  = np.zeros((ntime, npitch_bins, nenergy_bins)) \
                            * (1/unyt.eV) * (1/unyt.dimensionless)

                # Insert slices
                for payload in all_payloads:
                    ti = payload["time_indices"]
                    density_all[ti, ...]  = payload["density"]
                    pressure_all[ti, ...] = payload["pressure"]
                    fEpitch_all[ti, ...]  = payload["fEpitch"]
        else:
            density_all  = density
            pressure_all = pressure
            fEpitch_all  = fEpitch 
                
        if rank == 0:
            # Let's make a Dataset to return
            ds = xr.Dataset()
            
            dims = ['time', 'rho'] + (['theta', 'phi'] if preserve_angles else [])
            coords = {'time': self.abscissas['time'][:-1],
                        'rho': rho[:-1]}
            if preserve_angles:
                coords['theta'] = self.abscissas['theta']
                coords['phi'] = self.abscissas['phi']
            
            ds['density'] = xr.DataArray(density_all.value,
                                        dims=dims,
                                        coords=coords,
                                        attrs={'units': '1/m**3',
                                                'description': 'Particle density profile'})
            ds['pressure'] = xr.DataArray(pressure_all.value,
                                        dims=dims,
                                        coords=coords,
                                        attrs={'units': 'Pa',
                                                'description': 'Pressure profile'})
            
            ds['fEpitch'] = xr.DataArray(fEpitch_all.value,
                                        dims=['time', 'pitch', 'energy'],
                                        coords={'time': self.abscissas['time'][:-1],
                                                'pitch': np.linspace(-1, 1, npitch_bins),
                                                'energy': np.linspace(Emin.to('eV').value, 
                                                                        Emax.to('eV').value, 
                                                                        nenergy_bins)},
                                        attrs={'units': '1/(eV * dimensionless)',
                                                'description': 'Energy-pitch distribution function'})
        else:
            ds = None
        
        self.profiles = ds
        if MPI_ENABLED:
            ds = comm.bcast(self.profiles if rank == 0 else None, root=0)
        
        return ds

    def dump_results(self, outfile: Union[str, Path], mode: str='a'):
        """
        Run all the analysis and generates a HDF5 output file with the results.

        Parameters
        ----------
        outfile : Union[str, Path]
            Output file path.
        mode : str, optional
            File mode, by default 'a'
        """
        losses_ds = self.load_losses()
        if self._dist5d_flag:
            profiles_ds = self.make_profiles()
        gitinfo = get_git_info('dict')

        # We combine all the results in a single datatree.
        result_tree = xr.DataTree()
        result_tree['losses'] = losses_ds
        if self._dist5d_flag:
            result_tree['profiles'] = profiles_ds
        else:
            logger.warning("Distribution data not available, skipping profiles generation.")
        
        for ikey in gitinfo:
            result_tree.attrs[ikey] = gitinfo[ikey]

        # We dump the datatree to a HDF5 file.
        result_tree.to_netcdf(outfile, mode=mode, engine='h5netcdf')
