import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from PIL import Image
import ast
import os
import glob
import re
from pathlib import Path
import pandas as pd

def interpolate_masked_azimuths(phis, intensity):
    """Reconstruct a SAXS azimuthal profile using 180-degree symmetry."""
    reconstructed = np.asarray(intensity, dtype=float).copy()
    if not np.any(np.isfinite(reconstructed)):
        raise ValueError("no valid intensity samples found -- check mask/parameters")

    # SAXS intensity is centrosymmetric: I(phi) = I(phi + pi). Average valid
    # opposite bins, or copy the valid member when its counterpart is masked.
    if reconstructed.size % 2 == 0:
        half_turn = reconstructed.size // 2
        for i in range(half_turn):
            j = i + half_turn
            i_valid = np.isfinite(reconstructed[i])
            j_valid = np.isfinite(reconstructed[j])
            if i_valid and j_valid:
                pair_mean = 0.5 * (reconstructed[i] + reconstructed[j])
                reconstructed[i] = pair_mean
                reconstructed[j] = pair_mean
            elif i_valid:
                reconstructed[j] = reconstructed[i]
            elif j_valid:
                reconstructed[i] = reconstructed[j]

    # Only use periodic linear interpolation if both members of an opposite
    # pair were unavailable.
    valid = np.isfinite(reconstructed)
    if not np.any(valid):
        raise ValueError("no reconstructable intensity samples found")
    if not np.all(valid):
        reconstructed = np.interp(
            phis, phis[valid], reconstructed[valid], period=2 * np.pi
        )
    return reconstructed

def calculate_hof(phis, intensity, reference_angle_deg=90.0):
    """
    Calculate the conventional Hermans orientation function.

    The full detector profile is folded into an unsigned angle theta from 0
    to 90 degrees relative to the chosen reference axis, then evaluated as:

        <cos^2(theta)> = sum(I cos^2(theta) sin(theta)) / sum(I sin(theta))
        HOF = (3 <cos^2(theta)> - 1) / 2

    Non-negative intensity makes the result intrinsically bounded by -0.5
    and 1.0. The default 90-degree reference is the vertical tangential-flow
    direction for this disk experiment.
    """
    phis = np.asarray(phis, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    finite = np.isfinite(phis) & np.isfinite(intensity)
    if not np.any(finite):
        raise ValueError("azimuthal profile contains no finite intensity")
    if np.any(intensity[finite] < 0):
        raise ValueError("HOF requires non-negative azimuthal intensity")

    reference_angle = np.deg2rad(reference_angle_deg)
    # abs(cos) treats opposite directions along the same physical axis as
    # equivalent, as appropriate for a centrosymmetric SAXS pattern.
    theta = np.arccos(
        np.clip(np.abs(np.cos(phis[finite] - reference_angle)), 0.0, 1.0)
    )
    orientation_weight = np.sin(theta)
    weighted_intensity = intensity[finite] * orientation_weight
    denominator = np.sum(weighted_intensity)
    if not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        raise ValueError("weighted azimuthal intensity is zero -- cannot calculate HOF")

    avg_cos2 = np.sum(weighted_intensity * np.cos(theta)**2) / denominator
    hof = 0.5 * (3.0 * avg_cos2 - 1.0)
    if hof < -0.5000001 or hof > 1.0000001:
        raise ValueError(f"calculated HOF {hof} is outside its physical bounds")
    return float(np.clip(hof, -0.5, 1.0))

def baseline_correct_azimuthal_profile(intensity, percentile=5.0):
    """Subtract a robust constant baseline from an azimuthal profile."""
    if not 0 <= percentile < 100:
        raise ValueError("baseline percentile must be between 0 and 100")
    intensity = np.asarray(intensity, dtype=float)
    finite = np.isfinite(intensity)
    if not np.any(finite):
        raise ValueError("cannot estimate a baseline from an empty profile")

    baseline = float(np.nanpercentile(intensity, percentile))
    corrected = np.maximum(intensity - baseline, 0.0)
    return corrected, baseline

def get_beam_center_and_q_scale(npz_data, requested_center=None):
    """Get beam center from q-axis zero crossings and q-per-pixel scale."""
    def zero_crossing_pixel(axis):
        axis = np.asarray(axis, dtype=float)
        pixel_indices = np.arange(axis.size, dtype=float)
        finite = np.isfinite(axis)
        axis = axis[finite]
        pixel_indices = pixel_indices[finite]
        if axis.size < 2:
            raise ValueError("q axis does not contain enough finite coordinates")
        if axis[0] > axis[-1]:
            axis = axis[::-1]
            pixel_indices = pixel_indices[::-1]
        if axis[0] <= 0 <= axis[-1]:
            return float(np.interp(0.0, axis, pixel_indices))
        return float(pixel_indices[np.argmin(np.abs(axis))])

    has_axes = 'x_axis' in npz_data and 'y_axis' in npz_data
    if requested_center is None:
        if not has_axes:
            raise ValueError(
                "automatic beam centering requires x_axis and y_axis in the NPZ; "
                "provide center_pixel=(row, col) instead"
            )
        x_axis = np.asarray(npz_data['x_axis'])
        y_axis = np.asarray(npz_data['y_axis'])
        center = (zero_crossing_pixel(y_axis), zero_crossing_pixel(x_axis))
        center_source = 'NPZ q-axis zero crossings'
    else:
        center = tuple(requested_center)
        center_source = 'user supplied'

    q_per_pixel = None
    if has_axes:
        x_axis = np.asarray(npz_data['x_axis'])
        y_axis = np.asarray(npz_data['y_axis'])
        x_step = np.nanmedian(np.abs(np.diff(x_axis)))
        y_step = np.nanmedian(np.abs(np.diff(y_axis)))
        if np.isfinite(x_step) and np.isfinite(y_step):
            q_per_pixel = float(0.5 * (x_step + y_step))

    return center, q_per_pixel, center_source

def load_detector_calibration(calibration_path):
    """Read the detector geometry needed for stitched TIFF analysis."""
    wanted_keys = {
        'wavelength_A', 'image_size', 'pixel_size_um',
        'beam_position', 'distance'
    }
    values = {}
    with open(calibration_path, 'r', encoding='utf-8') as calibration_file:
        for raw_line in calibration_file:
            line = raw_line.split('#', 1)[0].strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            if key in wanted_keys:
                values[key] = ast.literal_eval(value.strip())

    missing = wanted_keys.difference(values)
    if missing:
        raise ValueError(
            f"calibration file is missing: {', '.join(sorted(missing))}"
        )

    # The YAML stores image size and beam position as (x, y), whereas NumPy
    # arrays and this script use (row/y, col/x).
    image_size = tuple(int(value) for value in values['image_size'])
    beam_x, beam_y = (float(value) for value in values['beam_position'])
    center = (beam_y, beam_x)

    wavelength = float(values['wavelength_A'])
    pixel_size_m = float(values['pixel_size_um']) * 1e-6
    distance_m = float(values['distance'])
    pixel_angle = np.arctan(pixel_size_m / distance_m)
    q_per_pixel = float(
        (4 * np.pi / wavelength) * np.sin(0.5 * pixel_angle)
    )

    return {
        'image_size': image_size,
        'center': center,
        'q_per_pixel': q_per_pixel,
    }

def load_image_array(file_path):
    """Load a 2D SAXS image from an NPZ, TIFF, or TIF file."""
    suffix = Path(file_path).suffix.lower()
    if suffix == '.npz':
        with np.load(file_path) as data:
            if 'image' in data:
                image = data['image']
            else:
                image = data[data.files[0]]
    elif suffix in ('.tif', '.tiff'):
        with Image.open(file_path) as tiff_image:
            image = np.asarray(tiff_image)
    else:
        raise ValueError(
            f"unsupported file type '{suffix}'; use .npz, .tif, or .tiff"
        )

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            f"expected a 2D SAXS image, but {file_path} has shape {image.shape}"
        )
    return image.astype(float)

def load_saxs_data(file_path, requested_center=None, calibration_path=None):
    """Load a SAXS image and determine its center and reciprocal-space scale."""
    suffix = Path(file_path).suffix.lower()
    image = load_image_array(file_path)

    if suffix == '.npz':
        with np.load(file_path) as data:
            center, q_per_pixel, center_source = get_beam_center_and_q_scale(
                data, requested_center
            )
        return image, center, q_per_pixel, center_source

    calibration = None
    if calibration_path is not None:
        calibration = load_detector_calibration(calibration_path)
        expected_shape = (
            calibration['image_size'][1], calibration['image_size'][0]
        )
        if image.shape != expected_shape:
            raise ValueError(
                f"TIFF shape {image.shape} does not match calibration image "
                f"shape {expected_shape}"
            )

    if requested_center is not None:
        center = tuple(float(value) for value in requested_center)
        center_source = 'user supplied'
    elif calibration is not None:
        center = calibration['center']
        center_source = f"calibration file {os.path.basename(calibration_path)}"
    else:
        raise ValueError(
            "TIFF files do not contain q axes; provide center_pixel=(row, col) "
            "or a calibration_path"
        )

    q_per_pixel = calibration['q_per_pixel'] if calibration is not None else None
    return image, center, q_per_pixel, center_source

def calculate_hermans(image_data, center, q_radius, radial_half_width=45,
                      baseline_percentile=5.0,
                      reference_angle_deg=90.0):
    """
    Calculates Hermans Orientation Parameter (f) from a 2D array.
    center: (row, col) in pixels
    q_radius: radial distance from center to the peak of interest
    
    Returns:
        hof_raw: Hermans orientation function from the full profile
        hof_corrected: HOF after subtracting the azimuthal baseline
        baseline: subtracted intensity value
        phis: azimuthal angles (radians)
        i_phi: intensity profile as function of azimuthal angle
               (masked angles filled by periodic linear interpolation)
        coords: (xi, yi) integration path coordinates
    """
    # Define 360 unique azimuthal bins. Do not duplicate 0 degrees at 360.
    phis = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    cy, cx = center
    
    # Extract I(phi) from a radial band centered on the selected radius.
    if radial_half_width < 0 or q_radius - radial_half_width < 1:
        raise ValueError(
            "q_radius must be greater than the non-negative radial_half_width"
        )
    r_band = np.arange(
        q_radius - radial_half_width,
        q_radius + radial_half_width + 1,
    )
    r_grid, phi_grid = np.meshgrid(r_band, phis)
    
    # Convert polar to Cartesian for image sampling
    xi = cx + r_grid * np.cos(phi_grid)
    yi = cy + r_grid * np.sin(phi_grid)
    
    # Sample the image at these coordinates
    polar_samples = map_coordinates(
        image_data, [yi, xi], order=1, mode='constant', cval=np.nan
    )
    # Average the five-pixel radial band using only unmasked samples. If an
    # entire azimuth bin is masked, fill it from neighboring angles on the
    # circular profile rather than treating it as zero or omitting the gap.
    valid_counts = np.sum(np.isfinite(polar_samples), axis=1)
    i_phi = np.divide(
        np.nansum(polar_samples, axis=1),
        valid_counts,
        out=np.full(valid_counts.shape, np.nan, dtype=float),
        where=valid_counts > 0,
    )
    i_phi = interpolate_masked_azimuths(phis, i_phi)

    # Calculate HOF relative to the physical reference axis. Here 90 degrees
    # is the vertical, tangential-flow direction of the parallel-plate disk.
    hof_raw = calculate_hof(
        phis, i_phi, reference_angle_deg=reference_angle_deg
    )
    corrected_i_phi, baseline = baseline_correct_azimuthal_profile(
        i_phi, percentile=baseline_percentile
    )
    try:
        hof_corrected = calculate_hof(
            phis, corrected_i_phi,
            reference_angle_deg=reference_angle_deg,
        )
    except ValueError:
        # A perfectly flat profile has no intensity left after subtraction,
        # so a corrected orientation is undefined rather than zero.
        hof_corrected = np.nan
    
    # Return coordinates for verification plotting
    coords = (xi.flatten(), yi.flatten())
    
    return (
        hof_raw, hof_corrected, baseline, phis, i_phi, corrected_i_phi, coords
    )

def calculate_radial_diagnostics(image_data, center, radial_half_width=2,
                                 n_angles=360, minimum_valid_fraction=0.20,
                                 reference_angle_deg=90.0):
    """Calculate radial I(r) and raw HOF as functions of radius."""
    cy, cx = center
    height, width = image_data.shape
    max_radius = int(min(cy, cx, height - 1 - cy, width - 1 - cx))
    if max_radius < 1:
        raise ValueError("beam center is outside or too close to the image boundary")

    radii = np.arange(1, max_radius + 1)
    phis = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    r_grid, phi_grid = np.meshgrid(radii, phis)
    xi = cx + r_grid * np.cos(phi_grid)
    yi = cy + r_grid * np.sin(phi_grid)
    polar = map_coordinates(
        image_data, [yi, xi], order=1, mode='constant', cval=np.nan
    )

    radial_counts = np.sum(np.isfinite(polar), axis=0)
    radial_intensity = np.divide(
        np.nansum(polar, axis=0),
        radial_counts,
        out=np.full(radii.shape, np.nan, dtype=float),
        where=radial_counts > 0,
    )

    hof_by_radius = np.full(radii.shape, np.nan, dtype=float)
    valid_fraction = np.full(radii.shape, np.nan, dtype=float)
    for radius_index in range(radii.size):
        lo = max(0, radius_index - radial_half_width)
        hi = min(radii.size, radius_index + radial_half_width + 1)
        band = polar[:, lo:hi]
        angle_counts = np.sum(np.isfinite(band), axis=1)
        valid_fraction[radius_index] = np.mean(angle_counts > 0)
        if valid_fraction[radius_index] < minimum_valid_fraction:
            continue
        profile = np.divide(
            np.nansum(band, axis=1),
            angle_counts,
            out=np.full(n_angles, np.nan, dtype=float),
            where=angle_counts > 0,
        )
        try:
            profile = interpolate_masked_azimuths(phis, profile)
            hof_by_radius[radius_index] = calculate_hof(
                phis, profile, reference_angle_deg=reference_angle_deg
            )
        except ValueError:
            pass

    return radii, radial_intensity, hof_by_radius, valid_fraction

def plot_hermans_verification(img, center, q_radius, hof_raw, hof_corrected,
                              baseline, angles, intensity, corrected_intensity,
                              coords, radial_diagnostics=None, q_per_pixel=None,
                              radial_half_width=20, baseline_percentile=5.0,
                              reference_angle_deg=90.0):
    """
    Create visualization plots for Hermans orientation analysis.
    
    Parameters:
        img: 2D scattering image
        center: beam center (row, col)
        q_radius: integration radius in pixels
        hof_raw: Hermans orientation function from the full profile
        hof_corrected: HOF after constant-baseline subtraction
        baseline: baseline intensity subtracted from the profile
        angles: azimuthal angles (radians)
        intensity: intensity profile I(phi)
        coords: (xi, yi) integration path coordinates
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
     
    # Plot 1: Show where the integration is happening on the 2D image
    ax1.imshow(np.log1p(img), cmap='jet', origin='lower')
    cx, cy = center[1], center[0]  # convert to (x, y) indexing
    
    # Mark only the inner and outer boundaries of the integration band so the
    # scattering lobes remain visible underneath the annotation.
    boundary_angles = np.linspace(0, 2 * np.pi, 720)
    inner_radius = q_radius - radial_half_width
    outer_radius = q_radius + radial_half_width
    for boundary_index, radius in enumerate((inner_radius, outer_radius)):
        boundary_x = cx + radius * np.cos(boundary_angles)
        boundary_y = cy + radius * np.sin(boundary_angles)
        label = (
            f'Integration boundaries (r={inner_radius}, {outer_radius})'
            if boundary_index == 0 else '_nolegend_'
        )
        ax1.plot(
            boundary_x, boundary_y,
            color='white', alpha=1.0, linewidth=1.2,
            linestyle=(0, (5, 4)), label=label,
        )

    # Relate detector coordinates to the parallel-plate geometry. The sample
    # was scanned horizontally from the disk center toward its edge, making
    # detector x radial and detector y the local tangential-flow direction.
    geometry_axis_length = outer_radius + 60
    ax1.plot(
        [cx - geometry_axis_length, cx + geometry_axis_length], [cy, cy],
        color='red', alpha=0.70, linewidth=1.1,
        label='PLA disc radial / scan axis (0°)',
    )
    ax1.plot(
        [cx, cx], [cy - geometry_axis_length, cy + geometry_axis_length],
        color='red', alpha=0.70, linewidth=1.1,
        label=f'Tangential flow / HOF reference ({reference_angle_deg:g}°)',
    )
    axis_label_box = dict(
        facecolor='white', edgecolor='none', alpha=0.65, pad=1.5
    )
    ax1.text(
        cx + geometry_axis_length + 8, cy, 'radial $r$',
        color='red', fontsize=8, va='center', ha='left',
        bbox=axis_label_box,
    )
    ax1.text(
        cx, cy + geometry_axis_length + 8, 'tangential flow',
        color='red', fontsize=8, va='bottom', ha='center',
        bbox=axis_label_box,
    )
    ax1.plot(cx, cy, 'r+', markersize=7, markeredgewidth=2, label='Beam center (q=0)')
    # Empty-data handle: show the out-of-plane plate direction in the legend
    # without drawing another annotation over the SAXS pattern.
    ax1.plot(
        [], [], linestyle='none', marker='o', markersize=6,
        markerfacecolor='none', markeredgecolor='red',
        label='Plate normal / velocity gradient\n(out of detector plane)',
    )
    
    ax1.set_title(f"Integration Region: $q_{{radius}}$ = {q_radius} pixels")
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")
    ax1.legend(loc='lower left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Azimuthal intensity profile
    angle_degrees = np.degrees(angles)
    ax2.plot(angle_degrees, intensity, 'b-', linewidth=1.5,
             label='Raw intensity')
    ax2.axhline(
        baseline, color='white', linestyle='--', linewidth=1.2,
        label=f'{baseline_percentile:g}th-percentile baseline = {baseline:.2f}'
    )
    # Shade only the intensity remaining after baseline correction. The
    # corrected profile itself equals the vertical distance above this line.
    ax2.fill_between(
        angle_degrees, baseline, intensity,
        where=corrected_intensity > 0,
        color='tab:blue', alpha=0.25,
        label='Baseline-corrected intensity',
    )
    ax2.set_xlabel("Azimuthal Angle (degrees)")
    ax2.set_ylabel("Intensity")
    ax2.set_title(
        "Azimuthal Profile I(φ)\n"
        f"Raw HOF = {hof_raw:.4f}; "
        f"baseline-corrected HOF = {hof_corrected:.4f} "
        f"(reference = {reference_angle_deg:g}°)"
    )
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 360)
    ax2.legend(fontsize=8)

    # Plots 3 and 4: radial intensity and orientation as functions of q/radius.
    if radial_diagnostics is not None:
        radii, radial_intensity, hof_by_radius, valid_fraction = radial_diagnostics
        if q_per_pixel is not None:
            radial_axis = radii * q_per_pixel
            selected_position = q_radius * q_per_pixel
            half_width = radial_half_width * q_per_pixel
            radial_label = r"q ($\AA^{-1}$)"
        else:
            radial_axis = radii
            selected_position = q_radius
            half_width = radial_half_width
            radial_label = "Radius (pixels)"

        ax3.plot(radial_axis, radial_intensity, color='darkgreen', linewidth=1.2)
        ax3.axvspan(selected_position - half_width,
                    selected_position + half_width,
                    color='orange', alpha=0.25, label='Selected integration band')
        ax3.axvline(selected_position, color='darkorange', linestyle='--', linewidth=1)
        positive_intensity = radial_intensity[np.isfinite(radial_intensity) &
                                              (radial_intensity > 0)]
        if positive_intensity.size:
            ax3.set_yscale('log')
        ax3.set_xlabel(radial_label)
        ax3.set_ylabel("Mean intensity")
        ax3.set_title("Radial Intensity Profile I(q)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)

        reliable = valid_fraction >= 0.20
        ax4.plot(radial_axis[reliable], hof_by_radius[reliable],
                 color='purple', linewidth=1.2)
        ax4.axvspan(selected_position - half_width,
                    selected_position + half_width,
                    color='orange', alpha=0.25)
        ax4.axvline(selected_position, color='darkorange', linestyle='--', linewidth=1)
        ax4.axhline(0, color='black', linewidth=0.8, alpha=0.6)
        ax4.set_xlabel(radial_label)
        ax4.set_ylabel("HOF")
        ax4.set_title(
            f"Raw Hermans Orientation Function vs q "
            f"(reference = {reference_angle_deg:g}°)"
        )
        ax4.set_ylim(-0.5, 1.0)
        ax4.grid(True, alpha=0.3)
    else:
        ax3.set_visible(False)
        ax4.set_visible(False)
    
    plt.tight_layout()
    return fig

def apply_saxs_mask(image_data, mask_path, nan_masked=True,
                    flip_vertical=False, flip_horizontal=False,
                    rotate_deg=0):
    """
    Apply a SAXS mask from a PNG file to the image data.
    
    Parameters:
        image_data: 2D numpy array of SAXS intensity data
        mask_path: path to black and white PNG mask file
                   White (255) = valid data, Black (0) = masked (invalid)
        nan_masked: if True, set masked pixels to NaN; if False, set to 0
        flip_vertical: if True, flip the mask top-to-bottom before applying
        flip_horizontal: if True, flip mask left-to-right
        rotate_deg: rotate mask by multiples of 90 degrees clockwise
                    (allowed values: 0, 90, 180, 270)
    Returns:
        masked_image: image_data with mask applied
    """
    # Load the mask PNG file
    mask_img = Image.open(mask_path)
    
    # Convert to grayscale if needed
    if mask_img.mode != 'L':
        mask_img = mask_img.convert('L')

    # apply geometric transforms before resizing
    if rotate_deg not in (0, 90, 180, 270):
        raise ValueError("rotate_deg must be one of 0, 90, 180, 270")

    # perform flips first so that a following rotation doesn't reverse
    # their apparent effect
    if flip_vertical:
        mask_img = mask_img.transpose(Image.FLIP_TOP_BOTTOM)
    if flip_horizontal:
        mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)
    if rotate_deg:
        mask_img = mask_img.rotate(-rotate_deg, expand=True)

        
    # Resize mask to match image dimensions if needed
    image_height, image_width = image_data.shape[:2]
    mask_width, mask_height = mask_img.size
    
    if (mask_height, mask_width) != (image_height, image_width):
        print(f"Resizing mask from {mask_width}x{mask_height} to {image_width}x{image_height}")
        mask_img = mask_img.resize((image_width, image_height), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    mask_array = np.array(mask_img)
    
    # Create boolean mask: True where data is valid (white = 255).
    # Threshold at 128 to handle gray variations. Some remapped q-space NPZ
    # images also use negative values (commonly -1) as an invalid-pixel fill
    # value, so exclude those even when the PNG mask is white there.
    png_valid_mask = mask_array > 128
    source_invalid_mask = ~np.isfinite(image_data) | (image_data < 0)
    valid_mask = png_valid_mask & ~source_invalid_mask
    
    # Make a copy to avoid modifying the original
    masked_image = image_data.astype(float)
    
    # Apply mask by setting invalid pixels to NaN or 0
    if nan_masked:
        masked_image[~valid_mask] = np.nan
    else:
        masked_image[~valid_mask] = 0
    
    masked_count = np.sum(~valid_mask)
    source_invalid_count = np.sum(source_invalid_mask & png_valid_mask)
    print(
        f"Mask applied: {masked_count} pixels masked "
        f"({100*masked_count/valid_mask.size:.2f}%); "
        f"{source_invalid_count} invalid source pixels included"
    )
    
    return masked_image

def process_single_file(file_path, blank_path, mask_path, center_pixel, q_rad,
                        flip_vertical=True, flip_horizontal=False, rotate_deg=0,
                        save_plots=False, output_dir=None, show_plots=False,
                        calibration_path=None, radial_half_width=45,
                        baseline_percentile=5.0,
                        reference_angle_deg=90.0):
    """
    Process a single SAXS NPZ or TIFF file and calculate the orientation parameter.
    
    Parameters:
        file_path: path to sample NPZ, TIF, or TIFF file
        blank_path: path to a matching blank/background image (or None to skip)
        mask_path: path to SAXS mask PNG
        center_pixel: beam center (row, col), or None for automatic metadata
        q_rad: integration radius in pixels
        flip_vertical, flip_horizontal, rotate_deg: mask orientation parameters
        save_plots: if True, save plots to output_dir
        output_dir: directory to save plots (if save_plots=True)
        show_plots: if True, display plots during processing
        calibration_path: detector YAML used for TIFF center and q scale
        radial_half_width: pixels on either side of q_rad to integrate
        baseline_percentile: low percentile used as the constant I(phi) baseline
        reference_angle_deg: detector angle of the HOF reference direction
    
    Returns:
        dict containing raw and baseline-corrected HOF results
    """
    result = {
        'filename': os.path.basename(file_path),
        'hof_raw': None,
        'hof_baseline_corrected': None,
        'azimuthal_baseline': None,
        'baseline_percentile': baseline_percentile,
        'reference_angle_deg': reference_angle_deg,
        'reference_direction': 'vertical tangential flow',
        'q_radius': q_rad,
        'center': center_pixel,
        'status': 'failed',
        'error_msg': None
    }
    
    try:
        # Load sample data and its detector geometry.
        img, actual_center, q_per_pixel, center_source = load_saxs_data(
            file_path, center_pixel, calibration_path
        )
        result['center'] = actual_center
        result['center_source'] = center_source
        result['q_per_pixel'] = q_per_pixel
        result['q_value'] = q_rad * q_per_pixel if q_per_pixel is not None else None
        print(f"  Beam center: {actual_center} ({center_source})")
        
        # Background subtraction
        if blank_path is not None:
            blank_img = load_image_array(blank_path)
            
            # Handle shape mismatch
            if blank_img.shape != img.shape:
                from PIL import Image as PILImage
                h, w = img.shape
                blank_pil = PILImage.fromarray(blank_img.astype(np.uint16))
                blank_pil = blank_pil.resize((w, h), PILImage.Resampling.LANCZOS)
                blank_img = np.array(blank_pil)
            
            img = img - blank_img.astype(float)
            img = np.maximum(img, 0)
        
        # Apply SAXS mask
        img = apply_saxs_mask(img, mask_path, nan_masked=True,
                             flip_vertical=flip_vertical, flip_horizontal=flip_horizontal,
                             rotate_deg=rotate_deg)
        
        # Calculate Hermans parameter
        (hof_raw, hof_corrected, baseline, angles, intensity,
         corrected_intensity, coords) = calculate_hermans(
            img, actual_center, q_radius=q_rad,
            radial_half_width=radial_half_width,
            baseline_percentile=baseline_percentile,
            reference_angle_deg=reference_angle_deg,
        )
        radial_diagnostics = calculate_radial_diagnostics(
            img, actual_center,
            reference_angle_deg=reference_angle_deg,
        )
        
        result['hof_raw'] = hof_raw
        result['hof_baseline_corrected'] = hof_corrected
        result['azimuthal_baseline'] = baseline
        result['status'] = 'success'
        
        # Create and optionally save plots
        fig = plot_hermans_verification(
            img, actual_center, q_rad, hof_raw, hof_corrected, baseline,
            angles, intensity, corrected_intensity, coords,
            radial_diagnostics=radial_diagnostics, q_per_pixel=q_per_pixel,
            radial_half_width=radial_half_width,
            baseline_percentile=baseline_percentile,
            reference_angle_deg=reference_angle_deg,
        )
        
        if save_plots and output_dir:
            output_name = f"{Path(result['filename']).stem}_hermans.png"
            output_path = os.path.join(output_dir, output_name)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved plot: {output_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
    except Exception as e:
        result['error_msg'] = str(e)
        result['status'] = 'failed'
    
    return result

def batch_process_folder(folder_path, blank_path, mask_path, center_pixel, q_rad,
                        file_pattern='*_saxs.npz', flip_vertical=True, flip_horizontal=False,
                        rotate_deg=0, save_plots=True, save_results=True, show_plots=False,
                        calibration_path=None, radial_half_width=45,
                        baseline_percentile=5.0,
                        reference_angle_deg=90.0):
    """
    Process all matching NPZ or TIFF files in a folder.
    
    
    Parameters:
        folder_path: folder containing NPZ files
        blank_path: path to a matching blank/background image (or None to skip)
        mask_path: path to SAXS mask PNG
        center_pixel: beam center (row, col), or None for automatic metadata
        q_rad: integration radius in pixels
        file_pattern: glob pattern to match input files
        flip_vertical, flip_horizontal, rotate_deg: mask orientation parameters
        save_plots: if True, save plots to a subfolder
        save_results: if True, save results to CSV
        show_plots: if True, display each plot during processing (will block script)
    
    Returns:
        list of result dicts for each file processed
    """
    # Find all matching files
    search_pattern = os.path.join(folder_path, file_pattern)
    input_files = sorted(
        glob.glob(search_pattern),
        key=lambda path: [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', os.path.basename(path))
        ],
    )
    
    if not input_files:
        print(f"No files matching pattern '{file_pattern}' found in {folder_path}")
        return []
    
    print(f"Found {len(input_files)} files to process\n")
    
    # Create output directory for plots if needed
    if save_plots:
        plot_dir = os.path.join(folder_path, 'hermans_plots')
        os.makedirs(plot_dir, exist_ok=True)
    else:
        plot_dir = None
    
    # Process each file
    results = []
    for i, file_path in enumerate(input_files, 1):
        print(f"[{i}/{len(input_files)}] Processing: {os.path.basename(file_path)}")
        
        result = process_single_file(file_path, blank_path, mask_path, center_pixel, q_rad,
                                     flip_vertical=flip_vertical, flip_horizontal=flip_horizontal,
                                     rotate_deg=rotate_deg, save_plots=save_plots,
                                     output_dir=plot_dir, show_plots=show_plots,
                                     calibration_path=calibration_path,
                                     radial_half_width=radial_half_width,
                                     baseline_percentile=baseline_percentile,
                                     reference_angle_deg=reference_angle_deg)
        
        results.append(result)
        
        if result['status'] == 'success':
            print(
                f"  ✓ raw HOF = {result['hof_raw']:.4f}; "
                f"corrected HOF = {result['hof_baseline_corrected']:.4f}; "
                f"baseline = {result['azimuthal_baseline']:.2f}"
            )
        else:
            print(f"  ✗ Error: {result['error_msg']}")
        print()
    
    # Save results to CSV
    if save_results:
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(folder_path, 'hermans_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        print("\nSummary:")
        print(results_df[[
            'filename', 'hof_raw', 'hof_baseline_corrected',
            'azimuthal_baseline', 'reference_direction', 'status'
        ]])
    
    return results

# === OPTION 1: SINGLE FILE PROCESSING ===
PROCESS_SINGLE_FILE = False  # Set to True to process a single file

if PROCESS_SINGLE_FILE:
    file_path = r'd:\Brookhaven\MD_TD_data\saxs\analysis\q_image\M6210OCSTD_stitched_x0.000_y-0.200_5.00s_2263413_000000_saxs.npz'
    
    # Path to your SAXS mask PNG file
    mask_path = r'd:\Brookhaven\MD_TD_data\saxs\analysis\q_image\SAXS_mask.png'
    
    # Path to blank/background scan NPZ file (set to None to skip background subtraction)
    blank_path = None  # [ADJUST THIS TO YOUR BLANK SCAN FILE PATH]

    # Required for automatic TIFF centering; optional for NPZ files.
    calibration_path = None
    
    # === USER-ADJUSTABLE PARAMETERS ===
    # None = automatically use the q=0 coordinates stored in the NPZ axes.
    # If axes are unavailable, replace None with a measured (row, col) tuple.
    center_pixel = None
    q_rad = 135  # [ADJUST THIS based on your peak position]
    radial_half_width = 30  # Integrate q_rad ± this many pixels
    baseline_percentile = 5.0  # Constant baseline from the lowest intensities
    reference_angle_deg = 90.0  # Vertical tangential-flow direction
    
    result = process_single_file(file_path, blank_path, mask_path, center_pixel, q_rad,
                                flip_vertical=True, flip_horizontal=False, rotate_deg=0,
                                save_plots=False, show_plots=True,
                                calibration_path=calibration_path,
                                radial_half_width=radial_half_width,
                                baseline_percentile=baseline_percentile,
                                reference_angle_deg=reference_angle_deg)
    
    print(f"Raw HOF: {result['hof_raw']:.4f}")
    print(
        "Baseline-corrected HOF: "
        f"{result['hof_baseline_corrected']:.4f}"
    )

# === OPTION 2: BATCH PROCESSING ===
else:
    # Folder containing the stitched SAXS TIFF files.
    sample_folder = r"C:\Users\kduff\OneDrive\Documents\Analysis for others\Ejajul shear aligned PLA\saxs\stitched"
    
    # Path to your SAXS mask PNG file
    mask_path = r"C:\Users\kduff\OneDrive\Documents\Analysis for others\Ejajul shear aligned PLA\data\combined_mask_SAXS.png"  # [ADJUST THIS]
    
    # Path to a stitched blank/background TIFF (None skips subtraction).
    blank_path = None

    # Stitched detector geometry. The YAML supplies center=(1081, 743) in
    # NumPy (row, col) order and supplies the q-per-pixel conversion.
    calibration_path = r"C:\Users\kduff\OneDrive\Documents\Analysis for others\Ejajul shear aligned PLA\data\caliXS.yaml"
    
    # === USER-ADJUSTABLE PARAMETERS ===
    # None uses the stitched detector center in caliXS.yaml. Use (row, col)
    # here only if you later obtain a better measured beam center.
    center_pixel = None
    q_rad = 135  # [ADJUST THIS based on your peak position]
    radial_half_width = 30  # Integrate q_rad ± this many pixels
    baseline_percentile = 5.0  # Constant baseline from the lowest intensities
    reference_angle_deg = 90.0  # Vertical tangential-flow direction
    
    # Pattern matching the 30 leftoverSample stitched files.
    file_pattern = '*.tiff'
    
    # Run batch processing
    results = batch_process_folder(sample_folder, blank_path, mask_path,
                                   center_pixel, q_rad, file_pattern=file_pattern,
                                   flip_vertical=False, flip_horizontal=False, rotate_deg=0,
                                   save_plots=True, save_results=True,
                                   calibration_path=calibration_path,
                                   radial_half_width=radial_half_width,
                                   baseline_percentile=baseline_percentile,
                                   reference_angle_deg=reference_angle_deg)
