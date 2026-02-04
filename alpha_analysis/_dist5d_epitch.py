"""
Common ground to transform both the distributions from the ASCOT
and ORBIT-RF into a distribution in (E, pitch; others...) starting from:
- a 5D distribution with (ppara, pperp; others...)
- a 6D distribution with (vR, vPhi, vZ; others...)

The results will be returned as a DistData object from the ASCOT.
"""

import numpy as np
import xarray as xr
import os
from a5py.ascot5io.dist import DistData 
from a5py.physlib import parseunits
import logging
import sys
import unyt
from typing import Union

import logging
logger = logging.getLogger("transform2Epitch")

# Define custom formatter for colored output
class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "[%(asctime)s - %(funcName)s] %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)

# Configure the logger
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(ColoredFormatter())
logger.addHandler(ch)
logger.setLevel(logging.INFO)



try:
    from numba import njit, prange, jit
except ImportError:
    logger.warning('Numba not installed. The code will run without numba acceleration.')
    def njit(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    def prange(*args, **kwargs):
        return range(*args, **kwargs)

# @njit(cache=True, fastmath=True)
# def _pparapperp_to_Epitch(dist: np.ndarray, mass: float,
#                           pparamin: float, pparamax: float, npara: int,
#                           pperpmin: float, pperpmax: float, nperp: int,
#                           energymin: float, energymax: float, nenergy: int,
#                           pitchmin: float, pitchmax: float, npitch: int,
#                           Nmc: int=1000):
#     """
#     Transforms the distribution from (ppara, pperp) to (E, pitch).

#     This technique uses a random Montecarlo sampling to transform
#     the velocity-space of the distribution between the (ppara, pperp)
#     into the (E, pitch), automatically accounting for the Jacobian
#     of the transformation.
#     """
#     if dist.ndim < 2:
#         raise ValueError('The distribution must be at least 2D.')
#     if dist.shape[0] != npara:
#         raise ValueError('The first dimension of the distribution ' \
#                          'must be the ppara dimension.')
    
#     if dist.shape[1] != nperp:
#         raise ValueError('The second dimension of the distribution ' \
#                          'must be the pperp dimension.')
    
#     # Preparing the output.
#     out = np.zeros((nenergy, npitch, *dist.shape[2:]))

#     dppara = (pparamax - pparamin) / npara
#     dpperp = (pperpmax - pperpmin) / nperp
#     dE = (energymax - energymin) / nenergy
#     dpitch = (pitchmax - pitchmin) / npitch

#     for ivpara in prange(npara):
#         for ivperp in prange(nperp):
#             tmp = np.random.rand(Nmc*2) - 0.5
#             # Getting the ppara and pperp values.
#             ppara = pparamin + (ivpara + tmp[:Nmc]) * dppara
#             pperp = pperpmin + (ivperp + tmp[Nmc:]) * dpperp

#             # Getting the energy and pitch values.
#             E = 0.5 * (ppara**2 + pperp**2) / mass
#             pitch = ppara / np.sqrt(ppara**2 + pperp**2)

#             # Getting the indices of the energy and pitch values.
#             for ii in range(Nmc):
#                 idx = int((E[ii] - energymin) / dE)
#                 idy = int((pitch[ii] - pitchmin) / dpitch)
#                 if((idx < 0) or (idx >= nenergy) or (idy < 0) or (idy >= npitch)):
#                     continue
#                 out[idx, idy, ...] += dist[ivpara, ivperp, ...]/Nmc
#     return out

@njit(cache=True, fastmath=True)
def _compute_transformation_weights(
        pparamin, pparamax, npara,
        pperpmin, pperpmax, nperp,
        energymin, energymax, nenergy,
        pitchmin, pitchmax, npitch,
        mass, Nmc):
    """
    Pre-computes the sparse mapping (weights) from velocity space to Energy-Pitch space.
    Returns Coordinate List (COO) format vectors: sources, targets, weights.
    """
    
    # 1. Setup Grids
    dppara = (pparamax - pparamin) / npara
    dpperp = (pperpmax - pperpmin) / nperp
    dE = (energymax - energymin) / nenergy
    dpitch = (pitchmax - pitchmin) / npitch

    # Estimate buffer size. 
    # A single (ppara, pperp) bin spreads to a small cluster of (E, pitch) bins.
    # We estimate max 20 target bins per source bin to be safe.
    max_entries = npara * nperp * 50 
    
    src_indices = np.empty(max_entries, dtype=np.int32)
    dst_indices = np.empty(max_entries, dtype=np.int32)
    weights = np.empty(max_entries, dtype=np.float64)
    
    count = 0
    
    # Temporary local grid for binning MC samples of a single pixel
    local_grid = np.zeros((nenergy, npitch), dtype=np.float64)
    
    # 2. Iterate over input velocity grid
    for ivpara in range(npara):
        for ivperp in range(nperp):
            
            # --- Monte Carlo Sampling ---
            # Reset local grid
            local_grid[:] = 0.0
            
            # Generate all randoms at once for this pixel
            tmp = np.random.rand(Nmc * 2) - 0.5
            ppara_vals = pparamin + (ivpara + tmp[:Nmc]) * dppara
            pperp_vals = pperpmin + (ivperp + tmp[Nmc:]) * dpperp

            E_vals = 0.5 * (ppara_vals**2 + pperp_vals**2) / mass
            # Avoid division by zero if necessary, though pperp usually > 0
            pitch_vals = ppara_vals / np.sqrt(ppara_vals**2 + pperp_vals**2)
            
            # Binning
            found_any = False
            for k in range(Nmc):
                idx = int((E_vals[k] - energymin) / dE)
                idy = int((pitch_vals[k] - pitchmin) / dpitch)
                
                if (idx >= 0 and idx < nenergy and idy >= 0 and idy < npitch):
                    local_grid[idx, idy] += 1.0
                    found_any = True
            
            if not found_any:
                continue

            # --- Compress and Store Weights ---
            # We look for non-zero bins in local_grid and store them
            # This makes the operation a Sparse Matrix multiplication later
            flat_src_idx = ivpara * nperp + ivperp
            
            for r in range(nenergy):
                for c in range(npitch):
                    val = local_grid[r, c]
                    if val > 0:
                        if count >= max_entries:
                            raise RuntimeError("Weight buffer overflow. Increase estimate.")
                        
                        src_indices[count] = flat_src_idx
                        dst_indices[count] = r * npitch + c
                        weights[count] = val / Nmc
                        count += 1

    return src_indices[:count], dst_indices[:count], weights[:count]

@njit('void(float64[:, ::1], float64[:, ::1], int32[::1], int32[::1], float64[::1])',
      parallel=True, cache=True, fastmath=True)
def _apply_weights_parallel(dist_flat, out_flat, src, dst, w):
    """
    Applies the weights to the high-dimensional array.
    Parallelized over the SPATIAL dimensions (last dim of flattened inputs).
    """
    n_interactions = len(src)
    n_spatial = dist_flat.shape[1]

    # Parallel loop over the spatial points (chunks of the big matrix)
    # This avoids Race Conditions because every thread writes to a unique spatial index 'i'.
    for i in prange(n_spatial):
        for k in range(n_interactions):
            s = src[k]
            t = dst[k]
            val = dist_flat[s, i]
            
            # Accumulate result
            out_flat[t, i] += val * w[k]

def _pparapperp_to_Epitch(dist: np.ndarray, mass: float,
                              pparamin: float, pparamax: float, npara: int,
                              pperpmin: float, pperpmax: float, nperp: int,
                              energymin: float, energymax: float, nenergy: int,
                              pitchmin: float, pitchmax: float, npitch: int,
                              Nmc: int=1000):
    
    if dist.ndim < 2:
        raise ValueError('Dist must be at least 2D')

    # 1. Compute Weights (Geometry)
    # This is fast and done once, regardless of how many spatial points you have.
    src, dst, w = _compute_transformation_weights(
        pparamin, pparamax, npara,
        pperpmin, pperpmax, nperp,
        energymin, energymax, nenergy,
        pitchmin, pitchmax, npitch,
        mass, Nmc
    )


    # 2. Reshape Data for Parallel Processing
    # Combine (npara, nperp) into one dimension: n_vel_in
    # Combine (nenergy, npitch) into one dimension: n_vel_out
    # Combine (x, y, z, ...) into one dimension: n_spatial
    
    spatial_shape = dist.shape[2:]
    n_spatial = 1
    for s in spatial_shape:
        n_spatial *= s
        
    # Flatten input: (npara*nperp, n_spatial)
    dist_reshaped = dist.reshape((npara * nperp, n_spatial))

    # Ensure memory is contiguous before passing to Numba
    dist_reshaped = np.ascontiguousarray(dist_reshaped)
    
    # Prepare output: (nenergy*npitch, n_spatial)
    out_reshaped = np.zeros((nenergy * npitch, n_spatial), dtype=dist.dtype)

    # 3. Parallel Application
    _apply_weights_parallel(dist_reshaped, out_reshaped, src, dst, w)

    # 4. Reshape back to original dimensions
    return out_reshaped.reshape((nenergy, npitch, *spatial_shape))

@njit(cache=True, fastmath=True)
def _pvec_to_E(dist: np.ndarray, mass: float,
               prmin: float, prmax: float, npr: int,
               pphimin: float, ppimax: float, npphi: int,
               pzmin: float, pzmax: float, npz: int,
               energymin: float, energymax: float, nenergy: int,
               Nmc: int=1000):
    """
    Transforms the distribution from (vR, vPhi, vZ) to (E).
    This technique uses a random Montecarlo sampling to transform
    the velocity-space of the distribution between the (vR, vPhi, vZ)
    into the (E), automatically accounting for the Jacobian
    of the transformation.
    """
    if dist.ndim < 3:
        raise ValueError('The distribution must be at least 3D.')
    if dist.shape[0] != npr:
        raise ValueError('The first dimension of the distribution ' \
                         'must be the pR dimension.')
    if dist.shape[1] != npphi:
        raise ValueError('The second dimension of the distribution ' \
                         'must be the pPhi dimension.')
    if dist.shape[2] != npz:
        raise ValueError('The third dimension of the distribution ' \
                         'must be the pZ dimension.')
    
    # Preparing the output.
    out = np.zeros((nenergy, *dist.shape[3:]))
    dpr = (prmax - prmin) / npr
    dpphi = (ppimax - pphimin) / npphi
    dpz = (pzmax - pzmin) / npz
    dE = (energymax - energymin) / nenergy
    for ivr in prange(npr):
        for ivphi in prange(npphi):
            for ivz in prange(npz):
                tmp = np.random.rand(Nmc*3) - 0.5
                # Getting the pR, pPhi and pZ values.
                pR = prmin + (ivr + tmp[:Nmc]) * dpr
                pPhi = pphimin + (ivphi + tmp[Nmc:2*Nmc]) * dpphi
                pZ = pzmin + (ivz + tmp[2*Nmc:]) * dpz

                # Getting the energy values.
                E = 0.5 * (pR**2 + pPhi**2 + pZ**2) / mass

                # Getting the indices of the energy values.
                for ii in range(Nmc):
                    idx = int((E[ii] - energymin) / dE)
                    if(idx < 0 or idx >= nenergy):
                        continue
                    out[idx, ...] += dist[ivr, ivphi, ivz, ...]/Nmc
    return out

@parseunits(Emin='eV', Emax='eV', mass='amu', strip=False)
def transform2E(dist: DistData,
                Ebins: int, mass: float=2.014*unyt.amu,
                Emin: float=1.0*unyt.eV, Emax: float=None,
                Nmc: int=1000):
    """
    Perfoms the transformation of the distribution to (E) space.
    This will simply act as a wrapper to the corresponding MC sampling
    numba-accelerated functions, by looking at the input type.
    :param dist: The distribution to be transformed. It must be a DistData object
        from the ASCOT simulation.
    :param Ebins: The number of bins in the energy axis.
    :param mass: The mass of the particles in the distribution. It must be a
        unyt quantity with units of mass.
    :param Emin: The minimum energy to be considered in the distribution. When
        not provided, it will be set to a default value of 1.0 eV.
    :param Emax: The maximum energy to be considered in the distribution. When
        not provided, it will be tried to be numerically determined from the
        distribution.
    :param Nmc: The number of Montecarlo samples to be used in the transformation.
    """
    # Checking the distribution.
    if not isinstance(dist, DistData):
        raise ValueError('The distribution must be a DistData object from the ASCOT simulation.')
    if not set(('pr', 'pphi', 'pz')).issubset(dist.abscissae):
        raise ValueError('The distribution must have pr, pphi and pz as abscissae.')
    
    # Checking the inputs.
    if mass <= 0:
        raise ValueError('The mass must be greater than 0. Got %f' % mass)
    if Ebins <= 2:
        raise ValueError('The number of bins in the energy axis must ' \
                         'be greater than 0. Got %d' % Ebins)

    coords = {ii: dist.abscissa(ii).value for ii in dist.abscissae}
    hist = dist.histogram()
    histunits = hist.units
    hist = xr.DataArray(hist.value, dims=dist.abscissae,
                        coords=coords,
                        attrs={'units': dist._distribution.units})
    for ii in coords:
        hist[ii].attrs['units'] = dist.abscissa(ii).units

    if Emin is None:
        Emin = 1.0 * unyt.eV
    if Emax is None:
        pr = hist.pr.values.max() * unyt.Unit('kg*m/s')
        pphi = hist.pphi.values.max() * unyt.Unit('kg*m/s')
        pz = hist.pz.values.max() * unyt.Unit('kg*m/s')
        Emax = 0.5 * (pr**2 + pphi**2 + pz**2) / mass
        logger.info('The maximum energy is %f eV' % Emax)
    if Emax <= Emin:
        raise ValueError('The maximum energy must be greater than the minimum. ' +
                         'Got %f and %f' % (Emax, Emin))
    if Emax <= 0:
        raise ValueError('The maximum energy must be greater than 0. Got %f' % Emax)

    # Getting the pr, pphi and pz values.
    pr = hist.pr.values
    pphi = hist.pphi.values
    pz = hist.pz.values
    npr = len(pr)
    npphi = len(pphi)
    npz = len(pz)
    prmin = pr.min()
    prmax = pr.max()
    pphimin = pphi.min()
    ppimax = pphi.max()
    pzmin = pz.min()
    pzmax = pz.max()

    coords = [ii for ii in hist.dims if ii not in ['pr', 'pphi', 'pz']]
    coords = ['pr', 'pphi', 'pz', *coords]
    hist = hist.transpose(*coords)
    # Getting the energy values.
    distout = _pvec_to_E(hist.values, mass.to('kg').value,
                         prmin, prmax, npr,
                         pphimin, ppimax, npphi,
                         pzmin, pzmax, npz,
                         Emin.to('J').value, Emax.to('J').value, Ebins - 1,
                         Nmc=Nmc)
    # Generating the output dataset and DistData.
    coords = {ii: hist[ii].values for ii in hist.dims if ii not in ['pr', 'pphi', 'pz']}
    coords['E'] = np.linspace(Emin.to('eV').value, Emax.to('eV').value, Ebins - 1)
    outx = xr.DataArray(distout, dims=['E', *hist.dims[3:]],
                        coords=coords,
                        attrs=hist.attrs)
    
    # We need to modify the units of the outx.
    outx.attrs['units'] = str(histunits)
    for ii in coords:
        if ii == 'E':
            outx[ii].attrs['units'] = 'eV'
        else:
            outx[ii].attrs['units'] = hist[ii].attrs['units']
        outx.attrs['units'] += '/' + str(outx[ii].attrs['units'])
    # Generating the DistData.
    abscissae = {'E' : np.linspace(Emin.to('eV').value, Emax.to('eV').value, Ebins) * unyt.eV}
    for ii in hist.dims[3:]:
        abscissae[ii] = hist[ii].values * unyt.Unit(hist[ii].attrs['units'])

    out = DistData(outx.values * unyt.particles, **abscissae)
    return out, outx


@parseunits(Emin='eV', Emax='eV', mass='amu', strip=False)
def transform2Epitch(dist: Union[DistData, str, xr.Dataset],
                     Ebins: int, pitchbins: int,
                     mass: float=2.014*unyt.amu, 
                     Emin: float=1.0*unyt.eV, Emax: float=None, 
                     pitchmin: float=None, pitchmax: float=None,
                     Nmc: int=1000):
    """
    Perfoms the transformation of the distribution to (E, pitch) space.

    This will simply act as a wrapper to the corresponding MC sampling
    numba-accelerated functions, by looking at the input type.

    :param dist: The distribution to be transformed. It can be a DistData object
        from the ASCOT simulation, the path to the results of an ORBIT-RF simulation,
        or an xarray dataset with the distribution.

    :param Ebins: The number of bins in the energy axis.
    :param pitchbins: The number of bins in the pitch angle axis.
    :param Emin: The minimum energy to be considered in the distribution. When
        not provided, it will be set to a default value of 1.0 eV.
    :param Emax: The maximum energy to be considered in the distribution. When
        not provided, it will be tried to be numerically determined from the 
        distribution.
    :param pitchmin: The minimum pitch angle to be considered in the distribution.
        When not provided, it will be set to a default value of -1.0.
    :param pitchmax: The maximum pitch angle to be considered in the distribution.
        When not provided, it will be set to a default value of 1.0.
    """
    # Checking input type for the distribution.
    if isinstance(dist, DistData):
        hist = dist.histogram() 
        if 'ppar' in dist.abscissae:

            coords = {ii: dist.abscissa(ii).value for ii in dist.abscissae}

            hist = xr.DataArray(hist, dims=dist.abscissae,
                                coords=coords,
                                attrs={'units': dist._distribution.units})
            
            for ii in coords:
                hist[ii].attrs['units'] = dist.abscissa(ii).units

        else:
            raise ValueError('The distribution must have ppara and pperp.')
    elif isinstance(dist, str):
        # Getting the path to the distribution.
        if not os.path.exists(dist):
            raise ValueError('The path to the distribution does not exist: %s' % dist)
        
        hist0 = xr.load_dataset(dist)
        if 'vpar' not in hist0.dims:
            raise ValueError('The distribution must have vpar as a dimension.')
        if 'vperp' not in hist0.dims:
            raise ValueError('The distribution must have vperp as a dimension.')
        
        # We change the scale of the histogram to momenta.
        coords ={ii: hist0[ii].values for ii in hist0.dims if ii not in ['vpar', 'vperp']}
        coords['ppar'] = hist0.vpar.values * mass.to('kg').value
        coords['pperp'] = hist0.vperp.values * mass.to('kg').value
        hist = xr.DataArray(hist0.values,
                            dims=hist0.dims,
                            coords=coords,
                            attrs=hist0.attrs)
        
        for ii in coords.keys():
            hist[ii].attrs['units'] = hist0[ii].attrs['units']
    
    elif isinstance(dist, xr.Dataset):
        hist = dist
        if 'vpar' not in hist.dims:
            raise ValueError('The distribution must have vpar as a dimension.')
        if 'vperp' not in hist.dims:
            raise ValueError('The distribution must have vperp as a dimension.')
    else:
        raise ValueError('The distribution must be either a DistData object, ' \
                         'a path to an ORBIT-RF distribution or an xarray dataset.')
    
    # Getting the mass of the distribution.
    if mass <= 0:
        raise ValueError('The mass must be greater than 0. Got %f' % mass)
    
    # checking the binning.
    if Ebins <= 2:
        raise ValueError('The number of bins in the energy axis must ' \
                         'be greater than 0. Got %d' % Ebins)
    if pitchbins <= 2:
        raise ValueError('The number of bins in the pitch axis must be greater than 0. ' \
                         'Got %d' % pitchbins)
    
    if pitchmin is None:
        pitchmin = -1.0
    if pitchmax is None:
        pitchmax = 1.0
    if pitchmax <= pitchmin:
        raise ValueError('The maximum pitch angle must be greater than the minimum. ' +
                         'Got %f and %f' % (pitchmax, pitchmin))
    
    if Emin is None:
        Emin = 1.0 * unyt.eV
    
    # Getting the maximum energy.
    if Emax is None:
        ppara = hist.ppar.values.max() * unyt.Unit('kg*m/s')
        pperp = hist.pperp.values.max() * unyt.Unit('kg*m/s')
        Emax = 0.5 * (ppara**2 + pperp**2) / mass
        logger.info('The maximum energy is %f eV' % Emax)
    
    if Emax <= Emin:
        raise ValueError('The maximum energy must be greater than the minimum. ' +
                         'Got %f and %f' % (Emax, Emin))
    if Emax <= 0:
        raise ValueError('The maximum energy must be greater than 0. Got %f' % Emax)
    
    # Getting the ppara and pperp values.
    ppara = hist.ppar.values
    pperp = hist.pperp.values
    npara = len(ppara)
    nperp = len(pperp)
    pparamin = ppara.min()
    pparamax = ppara.max()
    pperpmin = pperp.min()
    pperpmax = pperp.max()

    coords = [ii for ii in hist.dims if ii not in ['ppar', 'pperp']]
    coords = ['ppar', 'pperp', *coords]
    hist = hist.transpose(*coords)

    # Getting the energy and pitch values.
    distout = _pparapperp_to_Epitch(hist.values, mass.to('kg').value,
                                    pparamin, pparamax, npara,
                                    pperpmin, pperpmax, nperp,
                                    Emin.to('J').value, Emax.to('J').value, Ebins -1,
                                    pitchmin, pitchmax, pitchbins - 1, 
                                    Nmc=Nmc)

    # Generating the output dataset and DistData.
    coords = {ii: hist[ii].values for ii in hist.dims if ii not in ['ppar', 'pperp']}
    coords['E'] = np.linspace(Emin.to('eV').value, Emax.to('eV').value, Ebins - 1)
    coords['pitch'] = np.linspace(pitchmin, pitchmax, pitchbins - 1)
    
    outx = xr.DataArray(distout, dims=['E', 'pitch', *hist.dims[2:]],
                        coords=coords,
                        attrs=hist.attrs)
    
    for ii in coords:
        if ii == 'E':
            outx[ii].attrs['units'] = 'eV'
        elif ii == 'pitch':
            outx[ii].attrs['units'] = 'dimensionless'
        else:
            outx[ii].attrs['units'] = hist[ii].attrs['units']
    
    # Generating the DistData.
    abscissae = {'E' : np.linspace(Emin.to('eV').value, Emax.to('eV').value, Ebins) * unyt.eV,
                 'pitch' : np.linspace(pitchmin, pitchmax, pitchbins) * unyt.dimensionless}
    for ii in hist.dims[2:]:
        if len(hist[ii].values) > 1:
            dx = hist[ii].values[1] - hist[ii].values[0]
            edges = np.concatenate(([[hist[ii].values[0] - dx/2,],
                                     hist[ii].values + dx/2]))
            abscissae[ii] = edges * unyt.Unit(hist[ii].attrs['units'])
        else:
            abscissae[ii] = hist[ii].values * unyt.Unit(hist[ii].attrs['units'])
    out = DistData(outx.values * unyt.particles, **abscissae)

    return out, outx
