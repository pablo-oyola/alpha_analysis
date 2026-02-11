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
from a5py import physlib
from ._logger import get_logger
from ._dist5d_epitch import transform2Epitch
from ._git import get_ascot_info

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

        self.data = xr.open_datatree(self.filepath)
        try:
            qid = self.data.results.attrs['active']
        except:
            raise KeyError("Could not find 'active' attribute in results: " +
                           "maybe there is not results in the file?")
    
        # == Particle losses ==
        # We don't need to load them here, to avoid overhead.

        # == Distribution loading ==
        self.results = self.data.results['run_' + str(qid)]
        if 'distrho5d' not in self.results:
            logger.warning(f"'distrho5d' not found in results.")
            self._dist5d_flag = False
        else:
            self._dist5d_flag = True

            self.distdata_on_disk = self.results['distrho5d']
            self.ntime = self.distdata_on_disk.ordinate.shape[-2]

            self.n_dims = self.distdata_on_disk.ordinate.ndim
            self.time_axis = self.n_dims - 2  # as used above

            self.abscissas = {}
            names = ['r', 'phi', 'z', 'ppar', 'pperp', 'time', 'charge']
            units = ['m', 'deg', 'm', 'kg*m/s', 'kg*m/s', 's', 'e']
            for i in range(len(self.distdata_on_disk.ordinate.shape)-1):
                self.abscissas[names[i]] = self.distdata_on_disk['abscissa_vec_%02d' % (i+1)][:].values * unyt.Unit(units[i])

    def get_particle_info(self, state: str='ini', only_lost: bool=False) -> xr.Dataset:
        """
        Load the initial particle information (energy, weight, ids) and returns it as a Dataset.

        Parameters
        ----------
        state : str, optional
            State to load the particle information from, by default 'ini'
        only_lost : bool, optional
            Whether to load only the lost particles, by default False
        """
        if only_lost:
            ids = self.a5.data.active.getstate("ids", endcond=["wall", "rhomax", "none"])
        else:
            ids = self.a5.data.active.getstate("ids", state='ini')

        keys = ['r', 'phi', 'z', 'ekin', 'pitch', 'weight', 'time', 'rho', 'theta']
        data = self.a5.data.active.getstate(*keys, state=state, ids=ids)

        dset = xr.Dataset()
        for ii, ikey in enumerate(keys):
            if isinstance(data[ii], unyt.unyt_array):
                units = str(data[ii].units)
                tmp = data[ii].value
            else:
                units = ''
                tmp = data[ii]
            dset[ikey] = xr.DataArray(tmp, 
                                      dims=['ids'], coords={'ids': ids},
                                      attrs={'units': units})

        return dset

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
        Computes density, flow velocities, pressure components, heat fluxes,
        and energy-pitch distribution from the 5D distribution.

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
        
        if MPI_ENABLED:
            Dt = comm.bcast(Dt if rank == 0 else None, root=0)
        
        # Storage arrays for all computed quantities
        density = np.zeros((len(time_indices),) + shape) * unyt.m**-3
        upara = np.zeros((len(time_indices),) + shape) * unyt.m / unyt.s
        uperp = np.zeros((len(time_indices),) + shape) * unyt.m / unyt.s
        prs_para = np.zeros((len(time_indices),) + shape) * unyt.Pa
        prs_perp = np.zeros((len(time_indices),) + shape) * unyt.Pa
        prs_scalar = np.zeros((len(time_indices),) + shape) * unyt.Pa
        qpar = np.zeros((len(time_indices),) + shape) * unyt.W / unyt.m**2
        qperp = np.zeros((len(time_indices),) + shape) * unyt.W / unyt.m**2
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

            # === Compute velocity moments, pressures, and heat fluxes ===
            # Prepare distribution with spatial coords + momentum
            dist_spatial = dist5d.integrate(copy=True, **{k: np.s_[:] for k in ['charge']})
            
            # Build velocity grids from momentum grids
            ppa, ppe = np.meshgrid(dist_spatial.abscissa("ppar"), 
                                   dist_spatial.abscissa("pperp"))
            pnorm = np.sqrt(ppa.ravel()**2 + ppe.ravel()**2)
            vnorm = physlib.velocity_momentum(mass, pnorm).reshape(ppa.shape)
            
            # Component velocities (handle zero momentum)
            pnorm_2d = pnorm.reshape(ppa.shape)
            with np.errstate(divide='ignore', invalid='ignore'):
                vpa = np.where(pnorm_2d > 0, ppa * vnorm / pnorm_2d, 0)
                vpe = np.where(pnorm_2d > 0, ppe * vnorm / pnorm_2d, 0)
            
            # === First moments: fluid velocities ===
            # Number density in phase space: n = ∫ f d³p
            d = dist_spatial._copy()
            if not preserve_angles:
                d.integrate(phi=np.s_[:], theta=np.s_[:])
            d.integrate(ppar=np.s_[:], pperp=np.s_[:])
            n = d.histogram()
            
            # Parallel fluid velocity: upa = ∫ vpa * f d³p / n
            d = dist_spatial._copy()
            d._multiply(vpa.T, "ppar", "pperp")
            if not preserve_angles:
                d.integrate(phi=np.s_[:], theta=np.s_[:])
            d.integrate(ppar=np.s_[:], pperp=np.s_[:])
            vpafluid = d.histogram()
            upa_val = vpafluid.copy()
            upa_val[n > 0] = (vpafluid[n > 0] / n[n > 0]).to('m/s')
            upa_val[n == 0] = 0 * unyt.m / unyt.s
            upara[itime, ...] = upa_val
            
            # Perpendicular fluid velocity: upe = ∫ vpe * f d³p / n
            d = dist_spatial._copy()
            d._multiply(vpe.T, "ppar", "pperp")
            if not preserve_angles:
                d.integrate(phi=np.s_[:], theta=np.s_[:])
            d.integrate(ppar=np.s_[:], pperp=np.s_[:])
            vpefluid = d.histogram()
            upe_val = vpefluid.copy()
            upe_val[n > 0] = (vpefluid[n > 0] / n[n > 0]).to('m/s')
            upe_val[n == 0] = 0 * unyt.m / unyt.s
            uperp[itime, ...] = upe_val
            
            # === Second moments: pressures ===
            # <vpa²>
            d = dist_spatial._copy()
            d._multiply(vpa.T**2, "ppar", "pperp")
            if not preserve_angles:
                d.integrate(phi=np.s_[:], theta=np.s_[:])
            d.integrate(ppar=np.s_[:], pperp=np.s_[:])
            vpafluid2 = d.histogram()
            
            # <vpe²>
            d = dist_spatial._copy()
            d._multiply(vpe.T**2, "ppar", "pperp")
            if not preserve_angles:
                d.integrate(phi=np.s_[:], theta=np.s_[:])
            d.integrate(ppar=np.s_[:], pperp=np.s_[:])
            vpefluid2 = d.histogram()
            
            # Pressure components (in Joules, not yet normalized)
            Ppa = mass * (vpafluid2 - n * upa_val**2)  # Parallel with bulk flow correction
            Ppe = mass * vpefluid2 / 2  # Perpendicular (gyrotropic, no correction)
            Pscalar = (Ppa + 2*Ppe) / 3  # Scalar pressure
            
            # Normalize by volume and time to get pressure densities
            if preserve_angles:
                # Broadcast vol to 3D shape (nrho, ntheta, nphi)
                vol_broadcast = vol[:, np.newaxis, np.newaxis]
            else:
                vol_broadcast = vol
            
            prs_para[itime, ...] = (Ppa.to('J') / vol_broadcast / Dt).to('Pa')
            prs_perp[itime, ...] = (Ppe.to('J') / vol_broadcast / Dt).to('Pa')
            prs_scalar[itime, ...] = (Pscalar.to('J') / vol_broadcast / Dt).to('Pa')
            
            # === Third moments: heat fluxes ===
            # Parallel heat flux: q‖ = (m/2) * ∫ (v‖ - ū‖)[v‖² + v⊥²] f d³p
            # = (m/2) * [<v‖³> - ū‖<v‖²> + <v‖·v⊥²> - ū‖<v⊥²>]
            
            # <vpa³>
            d = dist_spatial._copy()
            d._multiply(vpa.T**3, "ppar", "pperp")
            if not preserve_angles:
                d.integrate(phi=np.s_[:], theta=np.s_[:])
            d.integrate(ppar=np.s_[:], pperp=np.s_[:])
            vpafluid3 = d.histogram()
            
            # <vpa·vpe²>
            d = dist_spatial._copy()
            d._multiply((vpa * vpe**2).T, "ppar", "pperp")
            if not preserve_angles:
                d.integrate(phi=np.s_[:], theta=np.s_[:])
            d.integrate(ppar=np.s_[:], pperp=np.s_[:])
            vpa_vpe2 = d.histogram()
            
            # Parallel heat flux (in J/s, not yet normalized)
            Qpar = (mass / 2) * (vpafluid3 - upa_val * vpafluid2 + 
                                 vpa_vpe2 - upa_val * vpefluid2)
            qpar[itime, ...] = (Qpar.to('J/s') / vol_broadcast / Dt).to('W/m**2')
            
            # Perpendicular heat flux: q⊥ = (m/2) * ∫ v⊥[v‖² + v⊥²] f d³p  
            # = (m/2) * [<v⊥³> - ū⊥<v⊥²> + <v⊥·v‖²> - ū⊥<v‖²>]
            
            # <vpe³>
            d = dist_spatial._copy()
            d._multiply(vpe.T**3, "ppar", "pperp")
            if not preserve_angles:
                d.integrate(phi=np.s_[:], theta=np.s_[:])
            d.integrate(ppar=np.s_[:], pperp=np.s_[:])
            vpefluid3 = d.histogram()
            
            # <vpe·vpa²>
            d = dist_spatial._copy()
            d._multiply((vpe * vpa**2).T, "ppar", "pperp")
            if not preserve_angles:
                d.integrate(phi=np.s_[:], theta=np.s_[:])
            d.integrate(ppar=np.s_[:], pperp=np.s_[:])
            vpe_vpa2 = d.histogram()
            
            # Perpendicular heat flux (in J/s, not yet normalized)
            Qperp = (mass / 2) * (vpefluid3 - upe_val * vpefluid2 + 
                                  vpe_vpa2 - upe_val * vpafluid2)
            qperp[itime, ...] = (Qperp.to('J/s') / vol_broadcast / Dt).to('W/m**2')
            
            del dist_spatial  # Clean up

            # We integrate all the spatial variables to get f(ppar, pperp)
            intrg = {ii: np.s_[:] for ii in ['r', 'phi', 'theta', 'charge']}
            fmom = dist5d.integrate(True, **intrg)
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
                "upara": upara,
                "uperp": uperp,
                "prs_para": prs_para,
                "prs_perp": prs_perp,
                "prs_scalar": prs_scalar,
                "qpar": qpar,
                "qperp": qperp,
                "fEpitch": fEpitch,
            }
            all_payloads = comm.gather(payload, root=0)

            if rank == 0:
                # Allocate global arrays
                density_all  = np.zeros((ntime,) + shape) * unyt.m**-3
                upara_all = np.zeros((ntime,) + shape) * unyt.m / unyt.s
                uperp_all = np.zeros((ntime,) + shape) * unyt.m / unyt.s
                prs_para_all = np.zeros((ntime,) + shape) * unyt.Pa
                prs_perp_all = np.zeros((ntime,) + shape) * unyt.Pa
                prs_scalar_all = np.zeros((ntime,) + shape) * unyt.Pa
                qpar_all = np.zeros((ntime,) + shape) * unyt.W / unyt.m**2
                qperp_all = np.zeros((ntime,) + shape) * unyt.W / unyt.m**2
                fEpitch_all  = np.zeros((ntime, npitch_bins, nenergy_bins)) \
                            * (1/unyt.eV) * (1/unyt.dimensionless)

                # Insert slices
                for payload in all_payloads:
                    ti = payload["time_indices"]
                    density_all[ti, ...]  = payload["density"]
                    upara_all[ti, ...] = payload["upara"]
                    uperp_all[ti, ...] = payload["uperp"]
                    prs_para_all[ti, ...] = payload["prs_para"]
                    prs_perp_all[ti, ...] = payload["prs_perp"]
                    prs_scalar_all[ti, ...] = payload["prs_scalar"]
                    qpar_all[ti, ...] = payload["qpar"]
                    qperp_all[ti, ...] = payload["qperp"]
                    fEpitch_all[ti, ...]  = payload["fEpitch"]
        else:
            density_all  = density
            upara_all = upara
            uperp_all = uperp
            prs_para_all = prs_para
            prs_perp_all = prs_perp
            prs_scalar_all = prs_scalar
            qpar_all = qpar
            qperp_all = qperp
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
            ds['upara'] = xr.DataArray(upara_all.value,
                                        dims=dims,
                                        coords=coords,
                                        attrs={'units': 'm/s',
                                                'description': 'Parallel fluid velocity'})
            ds['uperp'] = xr.DataArray(uperp_all.value,
                                        dims=dims,
                                        coords=coords,
                                        attrs={'units': 'm/s',
                                                'description': 'Perpendicular fluid velocity'})
            ds['prs_para'] = xr.DataArray(prs_para_all.value,
                                        dims=dims,
                                        coords=coords,
                                        attrs={'units': 'Pa',
                                                'description': 'Parallel pressure'})
            ds['prs_perp'] = xr.DataArray(prs_perp_all.value,
                                        dims=dims,
                                        coords=coords,
                                        attrs={'units': 'Pa',
                                                'description': 'Perpendicular pressure'})
            ds['pressure'] = xr.DataArray(prs_scalar_all.value,
                                        dims=dims,
                                        coords=coords,
                                        attrs={'units': 'Pa',
                                                'description': 'Scalar pressure (Ppa + 2*Ppe)/3'})
            ds['qpar'] = xr.DataArray(qpar_all.value,
                                        dims=dims,
                                        coords=coords,
                                        attrs={'units': 'W/m**2',
                                                'description': 'Parallel heat flux'})
            ds['qperp'] = xr.DataArray(qperp_all.value,
                                        dims=dims,
                                        coords=coords,
                                        attrs={'units': 'W/m**2',
                                                'description': 'Perpendicular heat flux'})
            
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
    
    def get_runtime(self, fn: str):
        """
        Provided the console output from ASCOT, this will read the file and 
        obtain the runtime of the simulation and the number of CPUs used.

        Parameters
        ----------
        fn : str
            File path to the console output.
        """
        if not os.path.isfile(fn):
            raise FileNotFoundError(f"File {fn} does not exist.")
        all_found = 0
        with open(fn, 'r') as fout:
            lines = fout.readlines()
            for line in lines:
                if 'Simulation finished in' in line:
                    # This will give us the runtime.
                    time_str = line.split('Simulation finished in')[-1].strip().split(' ')[0]
                    sim_time = float(time_str)
                    all_found += 1
                if 'Initialized MPI, rank' in line:
                    # From here typically we get the number of MPI processes used, but we need to be careful as sometimes it can print multiple times.
                    size_str = line.split('size')[-1].strip().split('.')[0]
                    mpi_size = int(size_str)
                    all_found += 1
                
                if 'Simulation begins;' in line:
                    # We read the number of threads, which can be useful to estimate the total number of CPUs used.
                    # The string is usually like "Simulation begins; 4 threads."
                    threads_str = line.split('threads')[0].strip().split(' ')[-1]
                    n_threads = int(threads_str)
                    all_found += 1
                
                if all_found == 3:
                    break
            
        if all_found < 3:
            logger.warning(f"Could not find all runtime information in the console output. Found {all_found}/3 items.")
            sim_time = -1.0
            mpi_size = -1
            n_threads = -1
        
        dset = xr.Dataset()
        dset.attrs['simulation_time_s'] = sim_time
        dset.attrs['mpi_size'] = mpi_size
        dset.attrs['n_threads'] = n_threads
                

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
        gitinfo = get_ascot_info('dict')

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