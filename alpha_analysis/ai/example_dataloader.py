from dataloader import build_simulation_dataloader
from glob import glob

# folders = [
#     "/global/cfs/cdirs/m5300/results/G1600/G1600_00000",
#     "/global/cfs/cdirs/m5300/results/G1600/G1600_00001",
# ]
folders = glob("/global/cfs/cdirs/m5300/results/G1600/*/")

loader = build_simulation_dataloader(folders, batch_size=4, shuffle=True)
batch = next(iter(loader))

inputs = batch["inputs"]                  # [B, 2, ...] -> qpar, qperp
context_r = batch["context"]["R_lmn"]     # [B, max_R]
context_z = batch["context"]["Z_lmn"]     # [B, max_Z]
target = batch["target"]                  # [B, max_target_len]
target_mask = batch["target_mask"]        # valid entries in target
