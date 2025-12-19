import numpy as np
import unyt
from a5py.ascot5io.dist import DistData

try:
    from numba import njit, prange
except ImportError:
    def njit(**kwargs):
        def wrapper(func):
            return func
        return wrapper
    prange = range

@njit(parallel=True)
def _histogram2d(r: float, z: float, rhotor: float, 
                 data:float, rhomin: float=0.0, rhoout: float=1.0, 
                 nrho: int=100, n_samples: int=1000):
    """
    Transform a distribution in (R, z) into a distribution in rho.

    For each (R, z) cell, the code will sample n_samples at random
    points within (R_1, R_2), (z_1, z_2) and evaluate rho at those points.
    The distribution in rho is then constructed by accumulating the
    contributions from each (R, z) cell into the appropriate rho bins.

    Parameters
    ----------
    r : 1D array
        R grid points.
    z : 1D array
        Z grid points.
    rhotor : 2D array
        Rho values at each (R, z) grid point.
    data : 2D array
        Distribution values at each (R, z) grid point.
    rhomin : float
        Minimum rho value.
    rhoout : float
        Maximum rho value.
    nrho : int
        Number of rho bins.
    n_samples : int
        Number of random samples per (R, z) cell.
    """
    dr = r[1] - r[0]
    dz = z[1] - z[0]
    rho_edges = np.linspace(rhomin, rhoout, nrho + 1)
    distrho = np.zeros(nrho)

    for ir in prange(r.size - 1):
        for iz in prange(z.size - 1):
            r1 = r[ir]
            r2 = r[ir + 1]
            z1 = z[iz]
            z2 = z[iz + 1]
            cell_value = data[ir, iz]

            for isample in range(n_samples):
                rr = r1 + np.random.rand() * dr
                zz = z1 + np.random.rand() * dz

                # Bilinear interpolation of rhotor at (rr, zz)
                t = (rr - r1) / dr
                u = (zz - z1) / dz
                rho_sample = (
                    (1 - t) * (1 - u) * rhotor[ir, iz] +
                    t * (1 - u) * rhotor[ir + 1, iz] +
                    (1 - t) * u * rhotor[ir, iz + 1] +
                    t * u * rhotor[ir + 1, iz + 1]
                )

                # Find the appropriate rho bin
                if rhomin <= rho_sample < rhoout:
                    irho = int((rho_sample - rhomin) / (rhoout - rhomin) * nrho)
                    if irho < nrho:
                        distrho[irho] += cell_value / n_samples

    return distrho

def rho_from_Rz(a5, distRz: DistData, rhomin: float, 
                rhoout: float, nrho: int, n_samples=1000, phi: float=0.0):
    """
    Computes the distribution in rho by sampling points in (R, z) space
    and evaluating rho at those points.
    
    Parameters
    ----------
    a5 : a5py.Ascot
        ASCOT instance with the equilibrium loaded.
    distRz : DistData
        Distribution in (R, z) space.
    rhomin : float
        Minimum rho value.
    rhoout : float
        Maximum rho value.
    nrho : int
        Number of rho bins.
    n_samples : int
        Number of random samples per (R, z) cell.
    phi : float
        Toroidal angle at which to evaluate rho.
    """
    # We get the rhotor from the ASCOT instance into an array we can
    # easily linearly interpolate in numba.
    R  = distRz.abscissa('r').to('m').value
    Z  = distRz.abscissa('z').to('m').value
    rhotor_out = np.linspace(rhomin, rhoout, nrho + 1)

    rhotor_on_grid = a5.input_eval(R*unyt.m, phi*unyt.rad,
                                   Z*unyt.m, 0*unyt.s, 'rho', grid=True)
    rhotor_on_grid = rhotor_on_grid.to('dimensionless').value.squeeze()

    # We now compute the histogram in rho.
    distrho_vals = _histogram2d(R, Z, rhotor_on_grid, distRz.histogram().value,
                               rhomin, rhoout, nrho, n_samples=n_samples)
    
    # We now build a DistData object to return.
    distrho = DistData(distrho_vals, rho=rhotor_out * unyt.dimensionless)

    return distrho

