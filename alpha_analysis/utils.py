import numpy as np
import unyt
from a5py.ascot5io.dist import DistData
from numpy.polynomial.legendre import leggauss
from desc.grid import LinearGrid

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
    if data.ndim > 2:
        extradims = data.shape[2:]
        data = data.reshape(data.shape[0], data.shape[1], -1)
    else:
        extradims = ()
        data = data.reshape(data.shape[0], data.shape[1], 1)

    distrho = np.zeros((nrho, data.shape[2]))

    for ir in prange(r.size - 1):
        for iz in prange(z.size - 1):
            r1 = r[ir]
            z1 = z[iz]

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
                        for ifield in range(data.shape[2]):
                            distrho[irho, ifield] += data[ir, iz, ifield] / n_samples

    return distrho.reshape((nrho,) + extradims)

def distrz2distrho(a5, distRz: DistData, rhomin: float, 
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

    if 'phi' in distRz.abscissae:
        distRz = distRz.integrate(True, phi=np.s_[:])  # Integrate over phi if present.

    # We need to rearrange the axes to set R and z as the first two axes.
    axes_order = distRz.abscissae
    cur_raxis = axes_order.index('r')
    cur_zaxis = axes_order.index('z')
    new_order = [cur_raxis, cur_zaxis] + [i for i in range(len(axes_order)) if i not in (cur_raxis, cur_zaxis)]
    hist = distRz.histogram().value # This is now a numpy array.
    hist = np.transpose(hist, new_order)
    

    rhotor_on_grid = a5.input_eval(R*unyt.m, phi*unyt.rad,
                                   Z*unyt.m, 0*unyt.s, 'rho', grid=True)
    rhotor_on_grid = rhotor_on_grid.to('dimensionless').value.squeeze()

    # We now compute the histogram in rho.
    distrho_vals = _histogram2d(R, Z, rhotor_on_grid, hist,
                               rhomin, rhoout, nrho, n_samples=n_samples)
    
    # We need now to ensure the distribution also has space for the theta and phi
    # axes, that will have dimension 1.
    axes_order = ['rho', 'theta', 'phi'] + [ax for ax in distRz.abscissae if ax not in ('r', 'z', 'phi')]
    distrho_vals = distrho_vals.reshape((distrho_vals.shape[0], 1, 1) + distrho_vals.shape[1:])

    # Building the new abscissae list
    abscissae = {}
    abscissae['rho'] = rhotor_out * unyt.dimensionless
    abscissae['theta'] = np.array([0.0, 360.0]) * unyt.deg
    abscissae['phi'] = np.array([0.0, 360.0]) * unyt.deg
    for ax in distRz.abscissae:
        if ax not in ('r', 'z', 'phi'):
            abscissae[ax] = distRz.abscissa(ax)

    # We now build a DistData object to return.
    distrho = DistData(distrho_vals, **abscissae)

    return distrho

def compute_volume(eq, rho1: float, rho2: float, ntheta: int,
                   zeta1: float, zeta2: float, Nr: int=16,
                   Nz: int=16, Nq: int=6):
    """
    Compute per-cell plasma volumes in a ``(rho, zeta, theta)`` region.

    The routine integrates ``sqrt(g)`` (Jacobian determinant) from a DESC
    equilibrium over a finite-volume grid bounded by ``rho1 <= rho <= rho2``,
    ``0 <= theta <= 2*pi``, and ``zeta1 <= zeta <= zeta2``.

    Parameters
    ----------
    eq : desc.equilibrium.Equilibrium-like
        Equilibrium object exposing ``compute`` and compatible with
        ``desc.grid.LinearGrid`` evaluation.
    rho1 : float
        Lower radial flux coordinate bound.
    rho2 : float
        Upper radial flux coordinate bound.
    ntheta : int
        Number of coarse poloidal cells spanning ``[0, 2*pi]``.
    zeta1 : float
        Lower toroidal angle bound in radians.
    zeta2 : float
        Upper toroidal angle bound in radians.
    Nr : int, default=16
        Number of coarse cells in ``rho``.
    Nz : int, default=16
        Number of coarse cells in ``zeta``.
    Nq : int, default=6
        Gauss-Legendre points per coarse cell per dimension.

    Returns
    -------
    numpy.ndarray
        Array ``volumes`` with shape ``(Nr, Nz, ntheta)`` where each entry is
        the integrated physical volume of one coarse cell.

    Notes
    -----
    Method implemented:
    1. Build coarse cell edges in ``rho``, ``theta``, and ``zeta``.
    2. Map Gauss-Legendre nodes from ``[-1, 1]`` to each coarse cell,
       yielding concatenated quadrature nodes for each coordinate.
    3. Evaluate ``sqrt(g)`` on the tensor-product grid via DESC.
    4. Build mapped quadrature weights in each coordinate.
    5. For each coarse cell, contract local ``sqrt(g)`` values with the
       tensor-product weights to obtain the cell volume.
    """

    # ---------------------------------------------
    # 1) Coarse box edges
    # ---------------------------------------------

    rho_edges = np.linspace(rho1, rho2, Nr+1)
    theta_edges = np.linspace(0, 2*np.pi, ntheta+1)
    zeta_edges = np.linspace(zeta1, zeta2, Nz+1)

    xi, wi = leggauss(Nq)

    # ---------------------------------------------
    # 2) Build expanded 1D Gauss coordinates
    # ---------------------------------------------

    rho_nodes = np.concatenate([
        0.5*(rho_edges[i+1]-rho_edges[i])*xi
        + 0.5*(rho_edges[i+1]+rho_edges[i])
        for i in range(Nr)
    ])

    theta_nodes = np.concatenate([
        0.5*(theta_edges[k+1]-theta_edges[k])*xi
        + 0.5*(theta_edges[k+1]+theta_edges[k])
        for k in range(ntheta)
    ])

    zeta_nodes = np.concatenate([
        0.5*(zeta_edges[j+1]-zeta_edges[j])*xi
        + 0.5*(zeta_edges[j+1]+zeta_edges[j])
        for j in range(Nz)
    ])

    # ---------------------------------------------
    # 3) Single LinearGrid (tensor built internally)
    # ---------------------------------------------

    grid = LinearGrid(
        rho=rho_nodes,
        theta=theta_nodes,
        zeta=zeta_nodes
    )

    data = eq.compute(['sqrt(g)'], grid=grid)

    sqrtg = grid.meshgrid_reshape(
        data['sqrt(g)'], order='rtz'
    )

    # ---------------------------------------------
    # 4) Build Gauss weights expanded similarly
    # ---------------------------------------------

    rho_weights = np.concatenate([
        0.5*(rho_edges[i+1]-rho_edges[i])*wi
        for i in range(Nr)
    ])

    theta_weights = np.concatenate([
        0.5*(theta_edges[k+1]-theta_edges[k])*wi
        for k in range(ntheta)
    ])

    zeta_weights = np.concatenate([
        0.5*(zeta_edges[j+1]-zeta_edges[j])*wi
        for j in range(Nz)
    ])

    # reshape sqrtg to block structure
    sqrtg = sqrtg.reshape(
        Nr, Nq,
        ntheta, Nq,
        Nz, Nq
    )

    volumes = np.zeros((Nr, Nz, ntheta))

    # ---------------------------------------------
    # 5) Contract weights per box
    # ---------------------------------------------

    for i in range(Nr):
        for j in range(Nz):
            for k in range(ntheta):

                wr = rho_weights[i*Nq:(i+1)*Nq]
                wt = theta_weights[k*Nq:(k+1)*Nq]
                wz = zeta_weights[j*Nq:(j+1)*Nq]

                w = (
                    wr[:,None,None]
                    * wt[None,:,None]
                    * wz[None,None,:]
                )

                volumes[i,j,k] = np.sum(
                    sqrtg[i,:,k,:,j,:] * w
                )

    return volumes