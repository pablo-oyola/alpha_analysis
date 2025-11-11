import a5py
import alpha_analysis as aa
import matplotlib.pyplot as plt
import xarray as xr
import os

plt.ion()

# %% Configure and run Poincare plot
equ = '/home/pablooyola/Desktop/Pablo/PPPL/Thea-INFUSE/equilibria/equil_Helios_E0956_R80_B60_DESC_fixed.h5'
nr = 200
nz = 200
nphi = 600
ascot_database = '/home/pablooyola/Desktop/Pablo/PPPL/Thea-INFUSE/ascot5_database_inputs/'
npoincare = 10
ntorpasses = 100
sim_mode = 'gc'

# %% Run and plot.
fn = os.path.basename(equ).replace('.h5', '')
db_path = os.path.join(ascot_database, fn)
if not os.path.exists(db_path):
    os.makedirs(db_path, exist_ok=True)

os.chdir(db_path)
poincare = aa.Poincare(equ=equ, nr=nr, nz=nz, nphi=nphi,
                        prefix='ascot')

dset = poincare.run(npoincare=npoincare, sim_mode=sim_mode,
                    ntorpasses=ntorpasses)

fn_actual_database = os.path.join(db_path, f'poincare_db.nc')
if not os.path.exists(fn_actual_database):
    dtree = xr.DataTree()
else:
    dtree = xr.open_datatree(fn_actual_database)

# We look up for the current number of poincares saved to the datatree.
# The entries are labeled as poincare_0, poincare_1, ...
existing_keys = [key for key in dtree.data_vars.keys() if key.startswith('poincare_')]
if existing_keys:
    existing_indices = [int(key.split('_')[1]) for key in existing_keys]
    next_index = max(existing_indices) + 1
else:
    next_index = 0

dtree[f'poincare_{next_index}'] = dset
dtree.to_netcdf(fn_actual_database, mode='a')

poincare.plot(dset)
    