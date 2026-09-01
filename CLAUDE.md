# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

- **Run Reconstruction**: `python3 main.py <input_csv>`
  - Reconstructs reaction vertices from proton tracks provided in a CSV file.
  - Input CSV format: `event_id,hit1x,hit1y,hit1z,hit2x,hit2y,hit2z,hit3x,hit3y,hit3z,hit4x,hit4y,hit4z`

## Architecture & Structure

The project is a vertex reconstruction tool for the ExPRT tracker.

- `main.py`: Entry point. Manages the high-level workflow:
    - Reads input CSV via `utilities`.
    - Splits events into chunks for parallel processing using `multiprocessing.Pool`.
    - Calls `proton_reco.vertex_reco` for each event.
    - Aggregates results and filters them by `target_thickness`.
- `proton_reco.py`: Implements the core reconstruction algorithms.
    - `vertex_reco`: Dispatches to specific fitting functions based on the number of hits (2, 3, or 4).
    - `Fit2Hits`, `Fit3Hits`, `FitGroupsAdv`: Use `scipy.optimize.least_squares` to minimize residuals between extrapolated proton tracks and the z-axis, incorporating a Gaussian beam spot penalty.
- `utilities.py`: Provides shared utilities:
    - `Point3D`: A helper class for 3D coordinates.
    - File I/O helpers for reading and counting lines in large CSVs.
    - Coordinate shifting for simulation smears.
- `analysis.py`: Contains tools for analyzing and visualizing the reconstructed vertices:
    - `vertex3D`: Plots vertices in 3D space.
    - `analyse_vertex`: Performs Gaussian fits on vertex distributions to determine resolution ($\sigma$).
    - `save_resolution_efficiency`: Logs resolution and efficiency results to a CSV.

## Key Parameters

These are primarily configured in `main.py`'s `if __name__ == "__main__":` block:
- `target_thickness`: Boundary limit for the z-axis reconstruction (e.g., LH2 full length).
- `beam_spot`: The $\sigma$ (in mm) of the assumed 2D Gaussian beam profile, used as a penalty in the fit.
- `nthreads`: Number of CPU threads used for parallel reconstruction.
- `simresx`, `simresy`, `simresz`: Gaussian smear parameters for simulating position resolutions.
