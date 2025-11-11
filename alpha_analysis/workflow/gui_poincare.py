#!/usr/bin/env python3

"""
Matplotlib-only GUI for generating and saving Poincaré plots with ASCOT5.

Features:
- Parses CLI kwargs (equilibrium path, grid params, etc.)
- Prompts for equilibrium path if not provided via CLI
- Loads alpha_analysis.Poincare
- Inputs for number of Poincaré points
- Mode selection: field line (fl) or guiding center (gc)
- Energy (keV) and pitch enabled only when gc is selected
- Button to run and plot
- Button to save results to a NetCDF file

Note: Requires a functional ASCOT/a5py installation and DESC dependencies.
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import xarray as xr
import unyt
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, RadioButtons, Button

# Use a lightweight stdlib file dialog for visual file selection
try:
    import tkinter as _tk
    from tkinter import filedialog as _filedialog
except Exception:  # pragma: no cover
    _tk = None
    _filedialog = None

try:
    from alpha_analysis import Poincare
except Exception as e:  # pragma: no cover
    print("Failed to import alpha_analysis.Poincare: ", e, file=sys.stderr)
    raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matplotlib GUI for Poincaré plots")
    p.add_argument("--equ", "--equilibrium", dest="equ", default=None,
                   help="Path to DESC equilibrium file (HDF5). If omitted, use the GUI Browse+Load.")
    p.add_argument("--nr", type=int, default=200)
    p.add_argument("--nz", type=int, default=200)
    p.add_argument("--nphi", type=int, default=320)
    p.add_argument("--prefix", type=str, default="ascot")

    # Simulation defaults
    p.add_argument("--npoincare", type=int, default=100,
                   help="Number of points in the Poincaré section")
    p.add_argument("--mode", choices=["gc", "fl"], default="gc",
                   help="Simulation mode: 'gc' (guiding center) or 'fl' (field line)")
    p.add_argument("--ntorpasses", type=int, default=1000)
    p.add_argument("--phithreshold", type=float, default=1e-2)
    p.add_argument("--species", type=str, default="He4")
    p.add_argument("--energy", type=float, default=100.0,
                   help="Energy in keV (used only for gc mode)")
    p.add_argument("--pitch", type=float, default=1.0,
                   help="Pitch (dimensionless, used only for gc mode)")

    p.add_argument("--out", type=str, default="poincare.nc",
                   help="Output dataset file (NetCDF)")

    return p.parse_args(argv)


class PoincareGUI:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dset: xr.Dataset | None = None

        # Defer equilibrium selection and Poincare construction to GUI controls
        self.equ = None
        self.pc = None

        # Build figure and UI
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
        plt.subplots_adjust(left=0.3, bottom=0.25)
        self.ax.set_title("Poincaré GUI")
        self.ax.set_xlabel("R (m)")
        self.ax.set_ylabel("Z (m)")

        # Controls panel (left side)
        axcolor = "lightgoldenrodyellow"

        # Equilibrium selector (visual): path box + Browse + Load
        self.ax_equ = plt.axes([0.05, 0.90, 0.2, 0.05], facecolor=axcolor)
        self.tb_equ = TextBox(self.ax_equ, "Equ file", initial=str(args.equ or ""))

        self.ax_browse = plt.axes([0.26, 0.90, 0.09, 0.05])
        self.bt_browse = Button(self.ax_browse, "Browse")

        self.ax_load = plt.axes([0.26, 0.83, 0.09, 0.05])
        self.bt_load = Button(self.ax_load, "Load")

        self.ax_npoincare = plt.axes([0.05, 0.76, 0.2, 0.05], facecolor=axcolor)
        self.tb_npoincare = TextBox(self.ax_npoincare, "N points", initial=str(args.npoincare))

        self.ax_mode = plt.axes([0.05, 0.56, 0.2, 0.16], facecolor=axcolor)
        self.rb_mode = RadioButtons(self.ax_mode, ("gc", "fl"), active=0 if args.mode == "gc" else 1)

        self.ax_energy = plt.axes([0.05, 0.46, 0.2, 0.05], facecolor=axcolor)
        self.tb_energy = TextBox(self.ax_energy, "Energy [keV]", initial=str(args.energy))

        self.ax_pitch = plt.axes([0.05, 0.39, 0.2, 0.05], facecolor=axcolor)
        self.tb_pitch = TextBox(self.ax_pitch, "Pitch", initial=str(args.pitch))

        self.ax_ntor = plt.axes([0.05, 0.32, 0.2, 0.05], facecolor=axcolor)
        self.tb_ntor = TextBox(self.ax_ntor, "N tor passes", initial=str(args.ntorpasses))

        self.ax_phi = plt.axes([0.05, 0.25, 0.2, 0.05], facecolor=axcolor)
        self.tb_phi = TextBox(self.ax_phi, "phi thresh [rad]", initial=str(args.phithreshold))

        self.ax_species = plt.axes([0.05, 0.18, 0.2, 0.05], facecolor=axcolor)
        self.tb_species = TextBox(self.ax_species, "Species", initial=str(args.species))

        # Buttons bottom
        self.ax_run = plt.axes([0.05, 0.10, 0.09, 0.06])
        self.bt_run = Button(self.ax_run, "Run")

        self.ax_save = plt.axes([0.16, 0.10, 0.09, 0.06])
        self.bt_save = Button(self.ax_save, "Save")

        self.ax_out = plt.axes([0.05, 0.03, 0.2, 0.05], facecolor=axcolor)
        self.tb_out = TextBox(self.ax_out, "Out file", initial=args.out)

        # Connect events
        self.rb_mode.on_clicked(self.on_mode_change)
        self.bt_browse.on_clicked(self.on_browse)
        self.bt_load.on_clicked(self.on_load)
        self.bt_run.on_clicked(self.on_run)
        self.bt_save.on_clicked(self.on_save)

        # Apply initial mode state
        self.apply_mode_state(args.mode)

        # Auto-load if --equ provided
        if args.equ:
            try:
                self.on_load()
            except Exception as e:
                print(f"[warn] Auto-load failed: {e}")

    def apply_mode_state(self, mode: str) -> None:
        # Simulate enable/disable by changing label color; TextBox has no disable API
        is_gc = (mode == "gc")
        lbl_energy = self.tb_energy.label
        lbl_pitch = self.tb_pitch.label
        color = "black" if is_gc else "gray"
        lbl_energy.set_color(color)
        lbl_pitch.set_color(color)
        self.fig.canvas.draw_idle()

    def on_mode_change(self, label: str) -> None:
        mode = label.strip()
        self.apply_mode_state(mode)

    def on_browse(self, _event=None) -> None:
        # Open a file dialog to pick the equilibrium file
        if _filedialog is None:
            print("File dialog not available on this platform.", file=sys.stderr)
            return
        # Create a transient Tk root only for the dialog
        root = _tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        fn = _filedialog.askopenfilename(title="Select DESC equilibrium file",
                                         filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")])
        root.destroy()
        if fn:
            self.tb_equ.set_val(fn)

    def on_load(self, _event=None) -> None:
        # Validate and construct Poincare object; draw background
        equ = (self.tb_equ.text or "").strip()
        if not equ or not os.path.isfile(equ):
            print("Please select a valid equilibrium file.", file=sys.stderr)
            return
        self.equ = equ
        try:
            self.pc = Poincare(equ=self.equ, nr=self.args.nr, nz=self.args.nz, nphi=self.args.nphi,
                               prefix=self.args.prefix)
            # Draw background context
            self.draw_background()
        except Exception as e:
            print("Load failed:", e, file=sys.stderr)
            self.pc = None

    def read_params(self) -> dict:
        # Safely parse controls with fallbacks
        def as_int(tb: TextBox, dv: int) -> int:
            try:
                return int(float(tb.text))  # allow "100.0"
            except Exception:
                return dv

        def as_float(tb: TextBox, dv: float) -> float:
            try:
                return float(tb.text)
            except Exception:
                return dv

        mode = self.rb_mode.value_selected
        params = {
            "npoincare": as_int(self.tb_npoincare, self.args.npoincare),
            "sim_mode": mode,
            "ntorpasses": as_int(self.tb_ntor, self.args.ntorpasses),
            "phithreshold": as_float(self.tb_phi, self.args.phithreshold),
            "species": self.tb_species.text.strip() or self.args.species,
        }
        if mode == "gc":
            params["energy"] = as_float(self.tb_energy, self.args.energy)
            params["pitch"] = as_float(self.tb_pitch, self.args.pitch)
        return params

    def on_run(self, _event=None) -> None:
        if self.pc is None:
            print("Load an equilibrium first (Browse -> Load).", file=sys.stderr)
            return
        params = self.read_params()
        try:
            dset = self.pc.run(**params)
            self.dset = dset
            self.redraw_results(dset)
        except Exception as e:
            print("Run failed:", e, file=sys.stderr)

    def on_save(self, _event=None) -> None:
        if self.dset is None:
            print("No dataset to save yet.", file=sys.stderr)
            return
        out = self.tb_out.text.strip() or self.args.out
        try:
            # Save as NetCDF
            self.dset.to_netcdf(out)
            print(f"Saved: {out}")
        except Exception as e:
            print("Save failed:", e, file=sys.stderr)

    def draw_background(self) -> None:
        # Draw rho contours to provide context for the Poincaré points
        if self.pc is None:
            return
        rgrid = np.linspace(self.pc.bsts['b_rmin'], self.pc.bsts['b_rmax'], 256).squeeze()
        zgrid = np.linspace(self.pc.bsts['b_zmin'], self.pc.bsts['b_zmax'], 255).squeeze()
        self.pc.a5.input_init(bfield=True)
        rhop = self.pc.a5.input_eval(rgrid*unyt.m, 0.0*unyt.rad,
                                     zgrid*unyt.m, 0*unyt.s, 'rho', grid=True)
        cs = self.ax.contour(rgrid, zgrid, rhop.squeeze().T, levels=np.arange(0, 1.0, 0.1), colors='gray', linewidths=0.8)
        self.ax.contour(rgrid, zgrid, rhop.squeeze().T, levels=[0.9999], colors='red', linewidths=1.5)
        self.ax.clabel(cs, inline=True, fmt="%.1f", fontsize=8)

    def redraw_results(self, dset: xr.Dataset) -> None:
        # Overlay results
        self.ax.scatter(dset['R'].values, dset['Z'].values, s=6, color='blue', marker='.', alpha=0.8, label='Poincaré')
        self.ax.legend(loc='best', fontsize=8)
        self.fig.canvas.draw_idle()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gui = PoincareGUI(args)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
