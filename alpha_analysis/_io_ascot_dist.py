import numpy as np
import xarray as xr
import os
from a5py.ascot5io.dist import Dist, DistData
import unyt


def dump_ascot_dist(dist: Dist, fn: str, compress: bool=False, 
                    dset: xr.Dataset=None) -> xr.Dataset:
    """
    Saves to a netCDF file the contents of the distribution function as generated
    by the ascot code and stored in a Dist object.

    Parameters
    ----------
    dist : Dist
        The distribution function object to be saved.
    fn : str
        The filename where to save the distribution.
    compress : bool, optional
        Whether to compress the data in the netCDF file, by default False.
    dset : xr.Dataset, optional
        An existing xarray Dataset to which to add the distribution data, by default None.
    Returns
    -------
    xr.Dataset
        The xarray Dataset containing the distribution data.    
    """
    if dset is None:
        dset = xr.Dataset()
    elif not isinstance(dset, xr.Dataset):
        raise TypeError("dset must be an xarray.Dataset instance.")
    dset['dist'] = xr.DataArray(dist._distribution.value,
                                dims=dist.abscissae,
                                coords={k: dist.abscissa(k).value for k in dist.abscissae})
    dset['dist'].attrs['units'] = str(dist._distribution.units)
    dset['dist'].attrs['description'] = 'Distribution function'
    
    for k in dist.abscissae:
        dset[k].attrs = { 'units': str(dist.abscissa(k).units),
                          'dx': dist.abscissa_edges(k).value[1] - dist.abscissa_edges(k).value[0],}

    dset['phasespacevol'] = xr.DataArray(dist.phasespacevolume().value,
                                         dims=dist.abscissae)
    dset['phasespacevol'].attrs['units'] = str(dist.phasespacevolume().units)
    dset['phasespacevol'].attrs['description'] = 'Phasespace volume'

    if compress:
        enconding = {'dist': {'zlib': True, 'complevel': 5},
                     'phasespacevol': {'zlib': True, 'complevel': 5}}
    else:
        enconding = None

    # Saving the dataset to file.
    dset.to_netcdf(fn, mode='w', encoding=enconding)

    return dset

def load_ascot_dist(fn: str) -> tuple[xr.Dataset, DistData]:
    """
    Loads the distribution function in 5D from file. Returns
    both a xarray Dataset and a DistData object.

    Parameters
    ----------
    fn : str
        The filename from which to load the distribution.
    
    Returns
    -------
    dset : xr.Dataset
        The xarray Dataset containing the distribution data.
    dist_obj : DistData
        The DistData object reconstructed from the data.
    """
    if not os.path.exists(fn):
        raise FileNotFoundError(f"File {fn} does not exist.")
    
    dset = xr.open_dataset(fn)
    
    if 'dist' not in dset or 'phasespacevol' not in dset:
        raise KeyError("The dataset does not contain the required keys: 'dist' and 'phasespacevol'.")
    

    # Let's get the histogram distribution.
    dist = dset['dist'].values * unyt.Unit(dset['dist'].attrs['units'])
    psvol = dset['phasespacevol'].values * unyt.Unit(dset['phasespacevol'].attrs['units'])
    hist = dist * psvol 

    # Getting the abscissae
    abscissae = {k: dset[k].values * unyt.Unit(dset[k].attrs['units']) for k in dset.coords if k != 'nmarkers'}
    abscissae_edges = dict()
    for k in abscissae:
        dx = dset[k].attrs['dx'] #* unyt.Unit(dset[k].attrs['units'])
        edges = np.linspace(abscissae[k].value[0] - dx / 2, abscissae[k].value[-1] + dx / 2, len(abscissae[k].value) + 1)
        abscissae_edges[k] = edges * unyt.Unit(dset[k].attrs['units'])
    
    # Reconstructing the Dist object.
    dist_obj = DistData(hist, **abscissae_edges)

    return dset, dist_obj