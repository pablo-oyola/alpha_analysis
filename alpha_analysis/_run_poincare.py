import a5py
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
from ._load import desc_field
from a5py.ascot5io.options import Opt
from a5py.physlib import parseunits
import unyt
import logging

logger = logging.getLogger('alpha_analysis')


class Poincare:
    """
    The class contains the generation of the inputs for the Poincare plot
    either with particles or using field line tracing with the ASCOT5 code.
    """
    def __init__(self, equ: str, nr: int=200, nz: int=200, nphi: int=320,
                 prefix: str='ascot'):
        """
        Parameters
        ----------
        equ : str
            Path to the DESC equilibrium file.
        nr : int
            Number of radial grid points.
        nz : int
            Number of vertical grid points.
        nphi : int
            Number of toroidal grid points.
        """
        self.equ = equ
        self.nr = nr
        self.nz = nz
        self.nphi = nphi
        self.prefix = prefix

        self.a5fn = f'{prefix}_{nphi}.h5'

        # We prepare the ASCOT5 equilibrium file.
        if os.path.exists(self.a5fn):
            logger.warning(f"Loading existing ASCOT5 file: {self.a5fn}")
            self.a5 = a5py.Ascot(self.a5fn, create=False)
            self.bsts = self.a5.data.bfield.active.read()
        else:
            self.bsts, self.lcfs = desc_field(equ, nphi=nphi, nr=nr, nz=nz)
            self.a5 = a5py.Ascot(self.a5fn, create=True)

            self.a5.data.create_input("B_STS", **self.bsts)
            self.a5.data.create_input("plasma_1D")
            self.a5.data.create_input("wall_rectangular")
            self.a5.data.create_input("E_TC")
            self.a5.data.create_input("N0_1D")
            self.a5.data.create_input("Boozer")
            self.a5.data.create_input("MHD_STAT")
            self.a5.data.create_input("asigma_loc")
            self.a5.data.create_input("RF2D")

        # Safety net: we can check how many Nphi are in the ASCOT5 file.
        self.a5.input_init(bfield=True)

        # Getting the axis values.

        # We will follow particles initialized at the outboard midplane 
        # along the "midplane" (z=x_axis).
        self.Raxis = self.bsts['axisr'][0]
        self.zaxis = self.bsts['axisz'][0]

        R_initial = np.linspace(self.Raxis, self.bsts['b_rmax'], 512).squeeze()
        rhop = self.a5.input_eval(R_initial*unyt.m, 0.0*unyt.rad, self.zaxis*unyt.m, 0*unyt.s, 'rho',
                            grid=False).squeeze()
        R_lcfs = np.interp(1.0, rhop, R_initial)

        self.rhop2R = xr.DataArray(R_initial, coords=[rhop], dims=['rhop'],
                                   attrs={'long_name': 'R major radius at midplane',
                                          'units': 'm',
                                          'R_LCFS': R_lcfs})

    @parseunits(energy='keV', pitch='dimensionless', strip=False)
    def run(self, npoincare: int=100, sim_mode: str='gc', ntorpasses: int=1000,
            phithreshold: float=1e-2, species: str='He4', 
            energy: float=100.0 * unyt.keV, pitch: float=1.0):
        """
        Run the Poincare plot.

        Starts particles equispaced in rhopol from the magnetic axis up to the
        separatrix. The particles/field line tracers are started at the outboard
        midplane and at phi=0.

        @todo Allow the user to change the poloidal plane for the Poincare section,
        and store that into the dataset attributes.

        Parameters
        ----------
        npoincare : int
            Number of points in the Poincare plot (number of particles/tracers).
        sim_mode : str
            Simulation mode: 'gc' for guiding center particles, 'fl' for field line tracing
        ntorpasses : int
            Number of toroidal passes to simulate.
        phithreshold : float
            Threshold in radians to consider a point close to phi=0 for the Poincare section.
        """
        opt = Opt.get_default()
        if sim_mode.lower() == 'gc':
            opt['SIM_MODE'] = 2 # 1= FO simulation; 2 = GC simulation. 4 = Field line tracing.
        elif sim_mode.lower() == 'fl':
            opt['SIM_MODE'] = 4 # Field line tracing.
        else:
            raise ValueError("sim_mode must be either 'gc' or 'fl'.")
        
        if npoincare <= 0:
            raise ValueError("npoincare must be a positive integer.")
        if ntorpasses <= 0:
            raise ValueError("ntorpasses must be a positive integer.")
        if phithreshold <= 0 or phithreshold >= np.pi:
            raise ValueError("phithreshold must be in the range (0, pi).")
        
        opt['ENABLE_ADAPTIVE'] = 1 # Enable/Disable adaptive method (1/0).
        opt['ENABLE_ORBIT_FOLLOWING'] = 1 # Enable orbit following.
        opt['ENABLE_COULOMB_COLLISIONS'] = 0 # Enable collisions.
        opt['ENABLE_MHD'] = 0 # Disable MHD.
        opt['ENABLE_DIST_5D'] = 0 # disable here.
        if hasattr(Opt, '_OPT_ENABLE_RF'):
            opt['ENABLE_RF'] = 0 # Disable RF.

        # Time step options.
        opt["FIXEDSTEP_USE_USERDEFINED"] = 0

        # Final end conditions.
        opt['ENDCOND_SIMTIMELIM'] = 0 # Flag for the simulation to finish at the MAX_MILEAGE.
        opt['ENDCOND_WALLHIT'] = 0 # Stop simulation when a particle hits the wall.
        opt['ENDCOND_RHOLIM']  = 1 # Stop the simulation with a given rho limit.
        opt['ENDCOND_ENERGYLIM'] = 0 # Disable energy limit end condition.
        opt['ENDCOND_MAXORBS'] = 2 # Simulation ends after a maximum number of passing the 
                                   # same toroidal angle (phi=0).

        # Setting up the limits.
        opt["ENDCOND_MIN_ENERGY"] = 0. # Energy in eV.
        opt["ENDCOND_MIN_THERMAL"] = 0.0 # Multiplier for determining the thermal threshold.
        opt['ENDCOND_LIM_SIMTIME'] = 0 # Disable simulation time limit.
        opt['ENDCOND_MAX_RHO'] = 0.9999 # Separatrix.
        opt['ENDCOND_MAX_TOROIDALORBS'] = ntorpasses
        opt['ENDCOND_MAX_POLOIDALORBS'] = 0 # We don't want to limit poloidal orbits.

        # Orbit writing options.
        opt['ENABLE_ORBITWRITE'] = 1
        opt['ORBITWRITE_MODE'] = 0
        opt['ORBITWRITE_NPOINT'] = opt['ENDCOND_MAX_TOROIDALORBS'] * 10 # How many points to write for the orbit.

        # We get the radius for equispaced rhop values.
        rhop_grid = np.linspace(0, 1.0, npoincare)
        R_grid = np.interp(rhop_grid, self.rhop2R.rhop.values, 
                           self.rhop2R.values)

        # Let's take a resonant particle.
        mrk = a5py.ascot5io.marker.Marker.generate(sim_mode.lower(), 
                                                   n=npoincare, 
                                                   species=species)
        mrk['r'][:] = R_grid * unyt.m
        mrk['z'][:] = self.zaxis * unyt.m
        mrk['phi'][:] = 0.0 * unyt.rad
        # We only need to fill in the energy and pitch for particle simulations.
        if sim_mode.lower() == 'gc':
            mrk['energy'][:] = energy
            mrk['pitch'][:] = pitch
            mrk['zeta'] = np.linspace(0, 2*np.pi, mrk['n'], endpoint=False) * unyt.rad

        self.a5.simulation_free()
        self.a5.simulation_initinputs()
        self.a5.simulation_initoptions(**opt)
        self.a5.simulation_initmarkers(**mrk)
        vrun = self.a5.simulation_run(printsummary=True)

        # From the Poincaré, we will only read the (R, z) positions close
        # to phi=0.
        r, z, phi = vrun.getorbit('r', 'z', 'phi')
        phimod = np.mod(phi.to('rad').value, 2*np.pi)
        flags = (phimod < phithreshold) * (np.abs(phimod - 2*np.pi) < phithreshold)
        # flags = np.ones_like(r.value, dtype=bool)

        r = r.to('m').value[flags]
        z = z.to('m').value[flags]
        phi = phi.to('rad').value[flags] # Just for quality assurance.

        dset = xr.Dataset()
        dset.attrs['description'] = f'Poincare plot data from ASCOT5 simulation ({sim_mode} mode).'
        dset.attrs['equilibrium_file'] = self.equ
        dset.attrs['n_r'] = self.nr
        dset.attrs['n_z'] = self.nz
        dset.attrs['n_phi'] = self.nphi
        dset.attrs['simulation_mode'] = sim_mode
        dset.attrs['ntorpasses'] = ntorpasses
        dset.attrs['phithreshold'] = phithreshold
        dset.attrs['species'] = species
        # dset.attrs['ascot_version'] = a5py.__version__

        dset['R'] = xr.DataArray(r, dims=['points'],
                                 attrs={'long_name': 'Major radius at midplane',
                                        'units': 'm'})
        dset['Z'] = xr.DataArray(z, dims=['points'],
                                 attrs={'long_name': 'Vertical position at midplane',
                                        'units': 'm'})
        dset['Phi'] = xr.DataArray(phi, dims=['points'],
                                   attrs={'long_name': 'Toroidal angle',
                                          'units': 'rad',
                                          'other_description': "Quality assurance variable."})
        return dset
        
    def plot(self, dset: xr.Dataset):
        """
        Plot the Poincare plot from the dataset generated with the `run` method.

        Parameters
        ----------
        dset : xr.Dataset
            Dataset containing the Poincare plot data.
        """
        # Building the grid.
        rgrid = np.linspace(self.bsts['b_rmin'], self.bsts['b_rmax'], 1024).squeeze()
        zgrid = np.linspace(self.bsts['b_zmin'], self.bsts['b_zmax'], 1023).squeeze()
        self.a5.input_init(bfield=True)
        rhop = self.a5.input_eval(rgrid*unyt.m, 0.0*unyt.rad, 
                                  zgrid*unyt.m, 0*unyt.s, 'rho',
                                  grid=True)
        
        fig, ax = plt.subplots(1)
        ax.contour(rgrid, zgrid, rhop.squeeze().T, levels=np.arange(0, 1.0, 0.1), colors='gray')
        ax.contour(rgrid, zgrid, rhop.squeeze().T, levels=[0.9999,], colors='red')
        ax.scatter(dset['R'].values, dset['Z'].values, s=1, color='blue', marker='.')
        ax.set_xlabel('R (m)')

        return fig, ax