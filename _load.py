import os
import numpy as np
from tqdm import tqdm
import desc.io as dscio
import desc.grid as dscg
from desc.compat import rescale
import time
from tqdm import tqdm
import unyt
from a5py.physlib import parseunits
from logging import getLogger
from scipy.interpolate import griddata, NearestNDInterpolator
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation, LinearTriInterpolator
logger = getLogger(__name__)

@parseunits(phimin='deg', phimax='deg', rescale_R='m', rescale_B='T', strip=False)
def desc_field(h5file: str, phimin: float=0, phimax: float=360, 
               nphi: int=361, ntheta: int=120, nr: int=100, nz: int=100, 
               psipad: float=0.000, waitingbar: bool=False,
               rescale_R: float=None, rescale_B: float=None):
    """Load magnetic field data from a DESC equilibrium.

    This will behave like a template for the DESC equilibrium for the ASCOT5
    code:
        - The first element to be returned is a dictionary with the 
        inputs for the write_hdf5 function in the B_STS ASCOT5 input.
        - The second is the last-closed flux surface (LCFS) in R,Z coordinates,
        which can be used as diagnostic.

    Parameters
    ----------
    h5file : str
        File path to DESC HDF5 output.
    phimin : float, optional
        Minimum toroidal angle phi (deg). Default = 0.
    phimax : float, optional
        Maximum toroidal angle phi (deg). Default = 360.
    nphi : int, optional
        Number of toroidal angle phi grid points. Default = 360.
    ntheta : int, optional
        Number of poloidal angle theta grid points. Default = 120.
    nr : int, optional
        Number of radial coordinate R grid points. Default = 100.
    nz : int, optional
        Number of vertical coordinate Z grid points. Default = 100.

    Returns
    -------
    out : dict
        Dictionary with the following items:
        - `'axis_nphi'`, `'b_nphi'`, `'psi_nphi'`: nphi
        - `'b_nr'`, `'psi_nr'`: nr
        - `'b_nz'`, `'psi_nz'`: nz
        - `'axis_phimin'`, `'b_phimin'`, `'psi_phimin'`: phimin (deg)
        - `'axis_phimax'`, `'b_phimax'`, `'psi_phimax'`: (phimax-phimin)*(nphi-1)/nphi (deg)
        - `'b_rmin'`, `'psi_rmin'`: minimum radial coordinate R of output grids (m)
        - `'b_rmax'`, `'psi_rmax'`: maximum radial coordinate R of output grids (m)
        - `'b_zmin'`, `'psi_zmin'`: minimum vertical coordinate Z of output grids (m)
        - `'b_zmax'`, `'psi_zmax'`: maximum vertical coordinate Z of output grids (m)
        - `'axis_r'`: R(phi) on the magnetic axis (m)
        - `'axis_z'`: Z(phi) on the magnetic axis (m)
        - `'psi0'`: toroidal magnetic flux on the magnetic axis (Wb)
        - `'psi1'`: toroidal magnetic flux through the last closed flux surface (Wb)
        - `'psi'`: toroidal magnetic flux psi(R,phi,Z) (Wb)
        - `'br'`: radial magnetic field B_R(R,phi,Z) (T)
        - `'bphi'`: toroidal magnetic field B_phi(R,phi,Z) (T)
        - `'bz'`: vertical magnetic field B_Z(R,phi,Z) (T)
    lcfs: dict
        - `'rlcfs'`: R coordinates of the last closed flux surface (m)
        - `'zlcfs'`: Z coordinates of the last closed flux surface (m)
    """
    if not os.path.isfile(h5file):
        raise FileNotFoundError(f"DESC file {h5file} not found.")

    fam = dscio.load(h5file, file_format="hdf5")
    try:  # if file is an EquilibriaFamily, use final Equilibrium
        eq = fam[-1]
    except:  # file is already an Equilibrium
        eq = fam

    if (rescale_R is not None) and (rescale_B is not None):
        eq = rescale(eq, L=("R0", rescale_R), B=("B0", rescale_B))
        logger.info("Rescaled equilibium to R0 = %s and B0 = %s", rescale_R, rescale_B)
    elif rescale_R is not None:
        eq = rescale(eq, L=("R0", rescale_R))
        logger.info("Rescaled equilibium to R0 = %s", rescale_R)
    elif rescale_B is not None:
        eq = rescale(eq, B=("B0", rescale_B))
        logger.info("Rescaled equilibium to B0 = %s", rescale_B)

    eq.resolution_summary()

    # toroidal angle array
    phi = np.deg2rad(np.linspace(phimin.to('deg').value, 
                                 phimax.to('deg').value,
                                 nphi, endpoint=True)) * unyt.deg
    # note: phi should start at 0 and end on 360, inclusive

    # magnetic axis
    grid_axis = dscg.LinearGrid(rho=0.0, zeta=nphi, NFP=1)
    data_axis = eq.compute(["R", "Z"], grid=grid_axis)
    axis_r = data_axis["R"]  # m
    axis_z = data_axis["Z"]  # m
    psi0 = 0  # Wb

    # boundary
    grid = dscg.LinearGrid(
        rho=1.0, theta=ntheta, zeta=nphi, NFP=1, sym=False, endpoint=True
    )
    data = eq.compute(["R", "Z"], grid=grid)
    bdry_r = data["R"].reshape((grid.num_zeta, grid.num_theta), order="C").T * unyt.m
    bdry_z = data["Z"].reshape((grid.num_zeta, grid.num_theta), order="C").T * unyt.m

    # boundaries
    rmin = np.min(bdry_r)  # m
    rmax = np.max(bdry_r)  # m
    zmin = np.min(bdry_z)  # m
    zmax = np.max(bdry_z)  # m
    psi1 = eq.Psi * unyt.Wb

    # output domain
    R_1d = np.linspace(rmin, rmax, nr)  # m
    Z_1d = np.linspace(zmin, zmax, nz)  # m
    Z_2d, R_2d = np.meshgrid(Z_1d, R_1d)

    # interpolate psi, B_R, B_phi, B_Z to cylindircal coordinates
    psi = np.zeros([nr, nz, nphi]) * unyt.Wb
    br = np.zeros([nr, nz, nphi])  * unyt.T
    bphi = np.zeros([nr, nz, nphi]) * unyt.T
    bz = np.zeros([nr, nz, nphi]) * unyt.T
    timings = {"compute": [], "griddata": [], "fill": []}

    # compute on concentric grid
    grid = dscg.ConcentricGrid(
        L=eq.L_grid, M=eq.M_grid, N=0, NFP=eq.NFP, node_pattern="linear"
    )

    # interpolate to cylindrical grid, iterate through toroidal angle
    # prepare timing containers to measure time spent in large blocks inside the nphi loop
    for k in tqdm(range(nphi), desc="Interpolating DESC field", total=nphi, disable=not waitingbar):        
        grid._nodes[:, 2] = phi[k].to('rad').value  # set toroidal angle
        t0 = time.perf_counter()
        data = eq.compute(["R", "Z", "psi", "B_R", "B_phi", "B_Z"], grid=grid)
        t1 = time.perf_counter()
        timings["compute"].append(t1 - t0)

        # We build a triangulation and use LinearTriInterpolator for better performance and accuracy
        points = np.vstack((data["R"], data["Z"])).T
        tri = Triangulation(points[:,0], points[:,1])
        psi_interp = LinearTriInterpolator(tri, data["psi"] * 2 * np.pi)  # DESC `psi` is normalized by 2 pi
        br_interp = LinearTriInterpolator(tri, data["B_R"])
        bphi_interp = LinearTriInterpolator(tri, data["B_phi"])
        bz_interp = LinearTriInterpolator(tri, data["B_Z"])

        # interpolate data inside DESC domain
        t0 = time.perf_counter()
        psi[:, :, k] = psi_interp(R_2d.to('m').value, Z_2d.to('m').value) * unyt.Wb
        br[:, :, k] = br_interp(R_2d.to('m').value, Z_2d.to('m').value) * unyt.T
        bphi[:, :, k] = bphi_interp(R_2d.to('m').value, Z_2d.to('m').value) * unyt.T
        bz[:, :, k] = bz_interp(R_2d.to('m').value, Z_2d.to('m').value) * unyt.T
        t1 = time.perf_counter()
        timings["griddata"].append(t1 - t0)

        # Replace br, bphi, bz NaN values outside LCFS with closest values
        t0 = time.perf_counter()
        data = br[:, :, k].value
        mask = np.where(~np.isnan(data))
        interp = NearestNDInterpolator(np.transpose(mask), data[mask])
        filled_data = interp(*np.indices(data.shape))
        br[:, :, k] = filled_data * unyt.T

        data = bz[:, :, k].value
        mask = np.where(~np.isnan(data))
        interp = NearestNDInterpolator(np.transpose(mask), data[mask])
        filled_data = interp(*np.indices(data.shape))
        bz[:, :, k] = filled_data * unyt.T

        
        data = bphi[:, :, k].value
        mask = np.where(~np.isnan(data))
        interp = NearestNDInterpolator(np.transpose(mask), data[mask])
        filled_data = interp(*np.indices(data.shape))
        bphi[:, :, k] = filled_data * unyt.T
        t1 = time.perf_counter()
        timings["fill"].append(t1 - t0)

    # Print a concise timing summary to help identify hotspots
    try:
        compute_total = sum(timings["compute"]) if timings["compute"] else 0.0
        griddata_total = sum(timings["griddata"]) if timings["griddata"] else 0.0
        fill_total = sum(timings["fill"]) if timings["fill"] else 0.0
        compute_mean = np.mean(timings["compute"]) if timings["compute"] else 0.0
        griddata_mean = np.mean(timings["griddata"]) if timings["griddata"] else 0.0
        fill_mean = np.mean(timings["fill"]) if timings["fill"] else 0.0
        print(
            "desc_field timing summary (s):\n >>> compute total={:.4f}, mean={:.4f}; \n >>> griddata total={:.4f}, mean={:.4f}; \n >>> fill total={:.4f}, mean={:.4f}".format(
                compute_total, compute_mean, griddata_total, griddata_mean, fill_total, fill_mean
            )
        )
    except Exception:
        # do not fail the main routine if timing summary has issues
        pass

    # change order from [R,Z,phiang] to [R,phiang,Z]
    psi = np.transpose(psi, (0, 2, 1))
    br = np.transpose(br, (0, 2, 1))
    bphi = np.transpose(bphi, (0, 2, 1))
    bz = np.transpose(bz, (0, 2, 1))

    # pad psi0 if needed
    if psipad != 0.0:
        print("Warning: Padding psi0 with", psipad)
        psi0 += psipad

    out = {
        "axis_phimin": phimin, 
        "axis_phimax": phimax, 
        "axis_nphi": nphi,
        "axisr": axis_r,  # m
        "axisz": axis_z,  # m
        "b_rmin": rmin,  # m
        "b_rmax": rmax,  # m
        "b_nr": nr,
        "b_zmin": zmin,  # m
        "b_zmax": zmax,  # m
        "b_nz": nz,
        "b_phimin": phimin,  # deg
        "b_phimax": np.rad2deg(phi[-1]),  # deg
        "b_nphi": nphi,
        "br": br,  # T
        "bphi": bphi,  # T
        "bz": bz,  # T
        "psi": psi,  # Wb
        "psi0": psi0,  # Wb
        "psi1": psi1,  # Wb
        "psi_rmin": rmin,  # m
        "psi_rmax": rmax,  # m
        "psi_nr": nr,
        "psi_zmin": zmin,  # m
        "psi_zmax": zmax,  # m
        "psi_nz": nz,
        "psi_phimin": phimin,  # deg
        "psi_phimax": np.rad2deg(phi[-1]),  # deg
        "psi_nphi": nphi,
    }
    lcfs = {
        "rlcfs": bdry_r,  # m
        "zlcfs": bdry_z,  # m
    }

    return out, lcfs