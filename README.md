# Scattering
Scripts for X-Ray scattering analysis

Python 3.10 or newer is recommended

These scripts are research analysis tools and require dataset specific configuration.
A successful optimizer termination message indicates numerical convergence, not necessarily a physically correct model.
Parameter uncertainties are local estimates and may be unreliable when parameters are strongly correlated or constrained by bounds.
Verify masks and calibration visually before interpreting orientation, spacing, or fit parameters.


unified_fit.py — Fits SAXS data using the multilevel Beaucage Unified model. It supports single file and batch processing, adjustable parameter bounds, fixed or fitted parameters, mass-fractal levels, hierarchical cutoffs, correlation corrections, and CSV/plot output.

-Accepted input files are `.dat`, `.txt`, or `.csv` files containing: q  intensity  [sigma]
-The uncertainty column is optional. Whitespace- and comma-separated files are supported, and text headers or         comment lines are skipped. Outputs include:
- A fit plot and residual plot for each sample
- A per-sample CSV containing the data, total fit, residual, and individual level contributions
- `fit_results.csv` containing fitted parameters and estimated uncertainties
- Before running, edit `FIT_CONFIG` in the script. Parameters use this form: Parameter(value, fit, lower, upper)
- Single file analysis: python unified_fit.py --file "path/to/sample.dat" --out-dir "path/to/results"
- Batch analysis: python unified_fit.py --folder "path/to/data" --out-dir "path/to/results"
- Before running, edit `FIT_CONFIG` in the script. Parameters use this form:

ubg_penalties.py — Fits SAXS data with a customized Unified Born–Green model that includes structural correlation terms. Penalty functions constrain the optimization toward physically reasonable parameters, with support for batch processing, uncertainty estimates, residual plots, and CSV output.

 - Input files must be `.dat` files with a header followed by at least two numeric columns: q  intensity
 - The starting values, bounds, q ranges, and penalty strengths are currently set inside `fit_with_penalties()`and in the calls near the bottom of the script. Review these values for each material system before interpreting fitted parameters.
 - Run one file: python ubg_penalties.py "path/to/sample.dat" --show-plot
 - Save single file plot: python ubg_penalties.py "path/to/sample.dat" --save-plot --output "path/to/sample_ubg_fit.png"
 - Batch analysis:python ubg_penalties.py --folder "path/to/data" --out-dir "path/to/results" --save-plot


HOPpy2.py — Calculates the Herman's orientation function from 2D SAXS detector images. It supports NPZ and TIFF data, detector masks, background subtraction, azimuthal reconstruction using SAXS symmetry, calibration handling, diagnostic plots, and batch analysis.

 - Supported formats: .npz files containing image and reciprocal space axes, and .tiff or .tif detector images
 - The mask must be a grayscale or black-and-white PNG. White pixels are treated as valid detector pixels and black pixels are excluded. The `flip_vertical`, `flip_horizontal`, and `rotate_deg` settings align the mask with the detector image. Check the diagnostic overlay before trusting the result. A blank image can be supplied through `blank_path`; use `None` to skip background subtraction.
 - NPZ files can provide `x_axis` and `y_axis`, allowing the script to locate q = 0 and calculate the reciprocal-space scale automatically. For TIFF data, provide either a measured `center_pixel=(row, col)` or a calibration file.
 - single file analysis: set PROCESS_SINGLE=TRUE
 - Batch analysis: set PROCESS_SINGLE=False
 - Configure (set these parameters to your analysis):
    file_path = r"path/to/sample.npz"
    mask_path = r"path/to/mask.png"
    blank_path = None
    calibration_path = None
    center_pixel = None
    q_rad = 135
    radial_half_width = 30
    baseline_percentile = 5.0
    reference_angle_deg = 90.0
   - Run: python HOPpy2.py

ellipsetrajectory.py — Analyzes butterfly or eyebrow-shaped SAXS patterns from tilted lamellar stacks using elliptic coordinates based on Grubb and Murthy. It estimates pattern rotation, lamellar spacing, tilt geometry, and center-streak dimensions, then exports fitted overlays and tabulated results.

 - reads .npz files containing an image and reciprocal space axes
 - change dat_directory to where you saved your data
 - The mask must be a grayscale or black-and-white PNG. White pixels are treated as valid detector pixels and black pixels are excluded. The `flip_vertical`, `flip_horizontal`, and `rotate_deg` settings align the mask with the detector image. Check the diagnostic overlay before trusting the result. A blank image can be supplied through `blank_path`; use `None` to skip background subtraction.
 - 
