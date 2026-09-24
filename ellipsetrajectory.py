from importlib_metadata import metadata
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.integrate import simpson
from scipy.optimize import minimize, curve_fit
from PIL import Image
from pathlib import Path
import pandas as pd
from skimage.measure import EllipseModel
from matplotlib.colors import PowerNorm

"""
Code based on Grubb & Murthy, J. Appl. Cryst. (1997). 30, 259-266

This python script works with 2D SAXS patterns of lamellar stacks with tilted layers (butterfly/eyebrow geometry) 
to extract the lamellar spacing and tilt distribution. It is currently design to import .npz files from Brookhaven 
National Lab's SAXS software, but can be adapted to work with other formats.

The Brookhaven NSLS II 11-BM CMS beamline .png Mask is used here as well. Apply your own mask as needed.

If the pattern is rotated, the streak at the origin is used to determine the rotation angle and correct for it before fitting alpha.

Lm = 2π / u_peak , the periodicity in the lamellar stack

The 2D SAXS patterns are in qx and qy coordinates and the center streak is 
used to rotate the image so an accurate alpha can be detected.

Alpha is detected by finding the intensity peak in qx for each slice of qy and fitting a line to the peak locations.


"""

def load_saxs_npz(filename, beam_center=(549, 738)):
    """
    Load SAXS data from NPZ file with x_axis and y_axis and convert to proper q values in Å⁻¹.
    
    Parameters:
    -----------
    filename : str
        Path to the NPZ file
    beam_center : tuple
        (x, y) pixel coordinates of the beam center
    
    Returns:
    --------
    image : 2D numpy array
        The 2D SAXS pattern
    q_values : 2D numpy array
        Array of q values for each pixel in Å⁻¹
    metadata : dict
        Additional metadata from the file
    """
    data = np.load(filename)
    
    # Load image data
    image = data['image'].astype(float)
    
    # Load axis information
    x_axis = data['x_axis']  # These are already in pixels
    y_axis = data['y_axis']  # These are already in pixels
    x_scale = float(data['x_scale'])  # Conversion factor from pixels to Å⁻¹
    y_scale = float(data['y_scale'])  # Conversion factor from pixels to Å⁻¹
    
    # Get image dimensions
    ny, nx = image.shape
    
    # Create pixel coordinate grids
    pixel_x, pixel_y = np.meshgrid(np.arange(nx), np.arange(ny))
    
    # Calculate pixel distances from beam center
    # beam_center[0] is x-coordinate, beam_center[1] is y-coordinate
    dx_pixels = pixel_x - beam_center[0]
    dy_pixels = pixel_y - beam_center[1]
    
    # Convert pixel distances to q-space distances using scale factors
    # The scale factors convert pixel coordinates to q values
    # Positive x in pixel space corresponds to positive q_x
    # Positive y in pixel space corresponds to positive q_y
    qx = dx_pixels * x_scale  # Now in Å⁻¹
    qy = dy_pixels * y_scale  # Now in Å⁻¹
    
    # Calculate total q = sqrt(qx² + qy²)
    q_values = np.sqrt(qx**2 + qy**2)
    
    # Calculate angles in degrees (0° = positive x-axis, 90° = positive y-axis)
    # arctan2(qy, qx) gives angle from positive x-axis
    angles = np.degrees(np.arctan2(qy, qx))
    angles = (angles + 360) % 360  # Convert to 0-360 range
    
    # Store metadata
    metadata = {
        'x_scale': x_scale,
        'y_scale': y_scale,
        'beam_center': beam_center,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'shape': image.shape,
        'angles': angles,
        'qx': qx,
        'qy': qy,
        'dx_pixels': dx_pixels,
        'dy_pixels': dy_pixels
    }
   
    return image, q_values, metadata

def apply_saxs_mask(image_data, mask_path, nan_masked=True,
                    flip_vertical=True, flip_horizontal=False,
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
         mask_img = mask_img.resize((image_width, image_height), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    mask_array = np.array(mask_img)
    
    # Create boolean mask: True where data is valid (white = 255)
    # Threshold at 128 to handle gray variations
    valid_mask = mask_array > 128
    
    # Make a copy to avoid modifying the original
    masked_image = image_data.astype(float)
    
    # Apply mask by setting invalid pixels to NaN or 0
    if nan_masked:
        masked_image[~valid_mask] = np.nan
    else:
        masked_image[~valid_mask] = 0
    
    return masked_image

def compute_murthy_coordinates(qx, qy, A_focal, alpha_deg):
    """
    Murthy 1997 elliptic cylindrical coordinates for TD beam geometry
    where peaks appear as on the qx/MD axis.
    
       
    Parameters:
    -----------
    qx, qy : 2D arrays in Å⁻¹
        qx along MD (draw direction, horizontal)
        qy along ND (normal direction, vertical)
    A_focal : float
        Focal parameter in Å⁻¹.
        A=0 recovers polar coordinates.
        Controls how elliptic the coordinate curves are.
        Physically: encodes the elongation of the scattering 
        object along MD.
    alpha_deg : float
        Tilt of the whole pattern from horizontal.
        Encodes butterfly/eyebrow stack rotation angle.
    
    Returns:
    --------
    u : 2D array in Å⁻¹
        Semi-minor axis coordinate.
        At v=90°: u = qx (MD component of q).
        Lamellar spacing directly: L = 2π/u_peak
        u-width encodes lamellar spacing distribution 
        and stack height (feeds into UBG).
    v : 2D array in radians [0, 2π]
        Angular coordinate (confocal hyperbolic).
        v = 90°  → MD axis, where your || peaks sit
        v = 0°   → ND axis (equatorial direction)
        v-position of peak encodes lamellar tilt φ from MD.
        v-width encodes distribution of φ (spread of tilt angles).
    """
    
    alpha = np.radians(alpha_deg)
    
    # Rotate for butterfly/eyebrow tilt
    xp =  qx * np.cos(alpha) - qy * np.sin(alpha) # qx is along MD, so it contributes to xp (which controls sin(v))
    yp =  qx * np.sin(alpha) + qy * np.cos(alpha) # qy is along ND, so it contributes to yp (which controls cos(v))
    
    #   xp (MD) = u * sin(v)            [Murthy's y = u*sin(v)]
    #   yp (ND) = sqrt(A²+u²) * cos(v)  [Murthy's x = sqrt(A²+u²)*cos(v)]
    # Find the discriminant to solve for u given xp, yp, and A_focal
    # Solve quadratic for s = u²:
    # xp²/u² + yp²/(A²+u²) = 1 
    # xp²*(A²+u²) + yp²*u² = u²*(A²+u²)
    # A²*xp² + xp²*u² + yp²*u² = A²*u² + u⁴
    # u⁴ + u²*(A² - xp² - yp²) - A²*xp² = 0
    
    b_coeff = A_focal**2 - xp**2 - yp**2 # This is the coefficient of u² in the quadratic equation
    c_coeff = -A_focal**2 * xp**2      # xp because MD = u*sin(v) role
    
    discriminant = b_coeff**2 - 4.0*c_coeff #b^2-4ac with a=1, to solve for u²
    s = (-b_coeff + np.sqrt(np.maximum(discriminant, 0.0))) / 2.0 # We take the positive root because u² must be positive
    u = np.sqrt(np.maximum(s, 0.0))
    
    denom_u = np.where(u > 1e-10, u, 1e-10)
    denom_a = np.sqrt(A_focal**2 + u**2 + 1e-30)
    
    sin_v = np.clip(xp / denom_u, -1.0, 1.0)
    cos_v = np.clip(yp / denom_a, -1.0, 1.0)
    
    v = np.arctan2(sin_v, cos_v)
    v = (v + 2*np.pi) % (2*np.pi)
    
    return u, v

def pattern_rotation():
    """
    Use the streak at the origin to determine how much the whole 2D saxs pattern is rotated. 
    The center streak should be oriented on and parallel to the y axis.
    Subtract the difference that center streak is in degrees from the fitted alpha_deg
    
    Returns:
    --------
    streak_angle_deg : float
        Angle of the streak in degrees.
    x_axis_degs : float
        Angle of the rotated x-axis in degrees.
    """
    
    origin_radius = 0.02
    mask_origin = (np.sqrt(qx**2+qy**2)<origin_radius) & np.isfinite(img) & (img>0)
    qy_vals = qy[mask_origin].ravel()
    qx_vals = qx[mask_origin].ravel()
    I_vals = img[mask_origin].ravel()

    # Restrict to window around 90 ± 15 degrees
    # Calculate angle from positive x-axis (0°) to each (qx, qy) point
    angles_deg = (np.degrees(np.arctan2(qy_vals, qx_vals)) + 360) % 360
    # Window: 75° to 105° (90 ± 15°)
    angle_mask = ((angles_deg >= 75) & (angles_deg <= 105))

    qx_vals = qx_vals[angle_mask]
    qy_vals = qy_vals[angle_mask]
    I_vals = I_vals[angle_mask]

    if len(qx_vals)>30:
        threshold = np.percentile(I_vals, 80)
        mask_bright = I_vals > threshold
        qx_bright = qx_vals[mask_bright]
        qy_bright = qy_vals[mask_bright]
        I_bright = I_vals[mask_bright]

        coords = np.vstack([qx_bright,qy_bright])
        coords_mean = np.mean(coords, axis=1, keepdims=True)
        coords_centered = coords - coords_mean
        cov = np.cov(coords_centered, aweights=I_bright)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_vec = eigvecs[:,np.argmax(eigvals)]

        #A_mat = np.vstack([qx_vals, np.ones_like(qx_vals)]).T
        #m, b = np.linalg.lstsq(A_mat*np.sqrt(I_vals[:,None]),qy_vals * np.sqrt(I_vals), rcond=None)[0]
        streak_angle_rad = np.arctan2(principal_vec[1], principal_vec[0])
        streak_angle_deg = np.degrees(streak_angle_rad)%180 - 90
        x_axis_degrees = streak_angle_deg - 90
        #if streak_angle_deg<0: #for negative streak angles, subtract the absolute value from alpha_deg to correct for rotation
        #    alpha_deg_corrected = alpha_deg + (np.abs(streak_angle_deg))
        #else:
        #    alpha_deg_corrected = alpha_deg - streak_angle_deg
        print(f"\nStreak angle: {streak_angle_deg:.2f}° ")
       # print(f"\nCorrected alpha_deg: {alpha_deg_corrected:.2f}°")
        return streak_angle_deg, x_axis_degrees        
    else:
        print(f"\nNot enough points to determine streak angle.")
    return streak_angle_deg, x_axis_degrees


def center_streak_length(qx, qy, image, streak_angle_deg, qx_width=0.01, qy_max=0.15, intensity_pct=80, min_points=30):
    """
    Estimate the center streak length along q_y after rotating the pattern by streak_angle_deg.

    The center streak is aligned to the y-axis by the streak angle. This function
    selects pixels near the rotated q_x = 0 line, thresholds them by intensity,
    and returns the q_y extent of the bright streak.

    Returns:
    --------
    streak_length : float
        Total length of the center streak in Å⁻¹.
    qy_min : float
        Lower q_y endpoint of the bright streak region.
    qy_max : float
        Upper q_y endpoint of the bright streak region.
    """
    if np.isnan(streak_angle_deg):
        return np.nan, np.nan, np.nan

    if streak_angle_deg < 0:
        theta = -np.radians(-streak_angle_deg)
    else:
        theta = np.radians(streak_angle_deg)

    qx_rot = qx * np.cos(theta) - qy * np.sin(theta)
    qy_rot = qx * np.sin(theta) + qy * np.cos(theta)

    valid = np.isfinite(image) & (image > 0)
    mask = valid & (np.abs(qx_rot) <= qx_width) & (np.abs(qy_rot) <= qy_max)

    if np.count_nonzero(mask) < min_points:
        return np.nan, np.nan, np.nan

    threshold = np.percentile(image[mask], intensity_pct)
    bright = mask & (image > threshold)
    if np.count_nonzero(bright) < min_points:
        return np.nan, np.nan, np.nan

    qy_vals = qy_rot[bright]
    qy_min = qy_vals.min()
    qy_max = qy_vals.max()
    streak_length = qy_max - qy_min

    return streak_length, qy_min, qy_max


def alpha(filename, u_peak_approx, beam_center=(549, 738)):
    image, q_values, metadata = load_saxs_npz(filename, beam_center=beam_center)
    streak_angle_deg, x_axis_degs = pattern_rotation()
    lamwidth =  0.04
   
    qx = metadata['qx']
    qy= metadata['qy']
    
    I_vals = image
    qx_min = u_peak_approx -0.011
    qx_max = u_peak_approx + 0.011
    qy_min = 0
    qy_max = 0.05

    mask = (qx > qx_min) & (qx < qx_max) & (qy > qy_min) & (qy < qy_max)
    #rotate masked qx and qy by streak_angle_deg to align the streak with the y-axis
    if streak_angle_deg<0:
        theta = -np.radians(-streak_angle_deg)
    else:
        theta = np.radians(streak_angle_deg)
    qx_rotated = qx[mask] * np.cos(theta) - qy[mask] * np.sin(theta)
    qy_rotated = qx[mask] * np.sin(theta) + qy[mask] * np.cos(theta)
    q_values_masked = q_values[mask]
    I_vals_masked = I_vals[mask]
    lammask = (np.sqrt(qx_rotated**2 + qy_rotated**2) < lamwidth) & np.isfinite(I_vals_masked) & (I_vals_masked > 0)
    qx_rotated = qx_rotated[lammask]
    qy_rotated = qy_rotated[lammask]
    q_values_masked = q_values_masked[lammask]
    I_vals_masked = I_vals_masked[lammask]

     
    if len(qx_rotated) > 30:
        threshold = np.percentile(I_vals_masked, 80)
        mask_bright = I_vals_masked > threshold
        qx_bright = qx_rotated[mask_bright]
        qy_bright = qy_rotated[mask_bright]
        I_slice = I_vals_masked[mask_bright]

        if np.sum(I_slice) <= 0:
            I_slice = np.ones_like(I_slice)

        coords = np.vstack([qx_bright, qy_bright])
        coords_mean = np.mean(coords, axis=1, keepdims=True)
        coords_centered = coords - coords_mean
        cov = np.cov(coords_centered, aweights=I_slice)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        lam1, lam2 = eigvals[order]
        principal_vec = eigvecs[:, order[0]]

        lamangle_rad = np.arctan2(principal_vec[1], principal_vec[0])
        lamangledeg = np.degrees(lamangle_rad) % 180 - 90

        weights = I_slice.astype(float)
        weights = weights / np.sum(weights)
        neff = 1.0 / np.sum(weights**2)
        alpha_err_rad = np.sqrt(np.maximum(lam2 / (neff * (lam1 + 1e-30)), 0.0))
        alpha_err = np.degrees(alpha_err_rad)
        return lamangledeg, alpha_err
    else:
        print(f"\nNot enough points to determine alpha angle.")
        return np.nan, np.nan

def A_peak(filename, beam_center):
    #estimate the peak position in qx and qy by binning the data in q magnitude and finding the maximum intensity
    image, _, metadata = load_saxs_npz(filename, beam_center=beam_center)
    qx = metadata['qx']
    qy = metadata['qy']
    img = image
    near_qx = np.abs(qy) < 0.01 # focus on near-horizontal region where peaks are expected
    q_mag = np.abs(qx) # Calculate magnitude of q for valid pixels
    valid= near_qx & np.isfinite(img) & (img > 0) & (q_mag > 0.015) & (q_mag < 0.15) # Restrict to range where peaks are expected and avoid beamstop/equatorial streak
    q_bins = np.linspace(0.002, 0.15, 1000) # finer bins for smoother peak position
    I_binned = np.zeros(len(q_bins)-1)
    for i in range(len(q_bins)-1):
        in_bin = valid & (q_mag >= q_bins[i]) & (q_mag < q_bins[i+1])
        if np.sum(in_bin) > 0:
            I_binned[i] = np.mean(img[in_bin])

    q_centers = (q_bins[:-1] + q_bins[1:]) / 2
    max_bin = np.argmax(I_binned)
    u_peak_approx = q_centers[max_bin]
    h = q_centers[1] - q_centers[0]

    # Refine peak location with a quadratic fit around the maximum bin
    if 0 < max_bin < len(q_centers) - 1:
        y0 = I_binned[max_bin - 1]
        y1 = I_binned[max_bin]
        y2 = I_binned[max_bin + 1]
        denom = (y0 - 2*y1 + y2)
        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom
            u_peak_approx = u_peak_approx + delta * h

    # estimate peak intensity from pixels in the max bin if available
    in_bin_max = valid & (q_mag >= q_bins[max_bin]) & (q_mag < q_bins[max_bin+1])
    if np.sum(in_bin_max) > 0:
        I_peak = np.mean(img[in_bin_max])
    else:
        I_peak = I_binned[max_bin]

    # Estimate q-peak uncertainty from half-height width and bin spacing
    q_peak_err = h / 2
    if I_peak > 0:
        half_max = 0.5 * I_peak
        left_mask = (q_centers < u_peak_approx) & (I_binned >= half_max)
        right_mask = (q_centers > u_peak_approx) & (I_binned >= half_max)
        if left_mask.any() and right_mask.any():
            q_left = q_centers[left_mask][-1]
            q_right = q_centers[right_mask][0]
            q_peak_err = max((u_peak_approx - q_left) / 2, (q_right - u_peak_approx) / 2, h / 2)
        elif left_mask.any():
            q_left = q_centers[left_mask][-1]
            q_peak_err = max((u_peak_approx - q_left) / 2, h / 2)
        elif right_mask.any():
            q_right = q_centers[right_mask][0]
            q_peak_err = max((q_right - u_peak_approx) / 2, h / 2)

    Lm = 2*np.pi / u_peak_approx
    Lm_err = 2*np.pi * q_peak_err / (u_peak_approx**2 + 1e-30)

    print(f"Estimated peak position in q: {u_peak_approx:.5f} ± {q_peak_err:.5f} Å⁻¹")
    print(f"Approximate long period L = 2π / q_peak ≈ {Lm:.1f} ± {Lm_err:.1f} Å")
    q_min_restrict = 0.7 * u_peak_approx
    q_max_restrict = 2 * u_peak_approx

    print(f"Estimated long period Lm = 2π / q_peak ≈ {Lm:.1f} Å")
    return Lm, u_peak_approx, q_peak_err, I_peak, Lm_err

def ellipse_geometry(u_peak_approx, alpha_deg, image, metadata, streak_angle_deg):
    # Calculate ellipse parameters based on Lm and alpha_deg
    ellipse_minor = u_peak_approx
    #estimate the major axis length using the size of the lamellar streak (not the center streak)
    I_vals = image
    
    qx = metadata['qx']
    qy = metadata['qy']
    qx_min = 0.01
    qx_max = 0.15
    qy_min = 0
    qy_max = 0.5

    mask = (qx > qx_min) & (qx < qx_max) & (qy > qy_min) & (qy < qy_max)
    #rotate masked qx and qy by streak_angle_deg to align the streak with the y-axis
    if streak_angle_deg<0:
        theta = -np.radians(-streak_angle_deg)
    else:
        theta = np.radians(streak_angle_deg)
    qx_rotated = qx[mask] * np.cos(theta) - qy[mask] * np.sin(theta)
    qy_rotated = qx[mask] * np.sin(theta) + qy[mask] * np.cos(theta)
   
    I_vals_masked = I_vals[mask]


    #Find the qy value in which intensity starts to die off
    peak_qx_list = []
    peak_qy_list = []
    peak_I_list = []
    for qy_bin_start in np.arange(qy_min, qy_max, 0.01):
        qy_bin_end = qy_bin_start + 0.01
        slice_mask = (qy_rotated > qy_bin_start) & (qy_rotated < qy_bin_end)
        qx_slice = qx_rotated[slice_mask]
        qy_slice_values = qy_rotated[slice_mask]
        I_slice = I_vals_masked[slice_mask]
        valid_slice = np.isfinite(I_slice) & (I_slice > 0)
        if np.sum(valid_slice) > 0:
            peak_index = np.argmax(I_slice[valid_slice])
            valid_idx = np.flatnonzero(valid_slice)[peak_index]
            peak_qx = qx_slice[valid_idx]
            peak_qy = qy_slice_values[valid_idx]
            peak_I = I_slice[valid_idx]
            peak_qx_list.append(peak_qx)
            peak_qy_list.append(peak_qy)
            peak_I_list.append(peak_I)

    peak_qx_array = np.array(peak_qx_list)
    peak_qy_array = np.array(peak_qy_list)
    peak_I_array = np.array(peak_I_list)

    ref_tolerance = 0.002
    ref_mask = (
        (np.abs(qx - 0.04) <= ref_tolerance)
        & (np.abs(qy - 0.14) <= ref_tolerance)
        & np.isfinite(image)
        & (image > 0)
    )
    I_ref_vals = image[ref_mask]
    if I_ref_vals.size > 0:
        I_ref = np.mean(I_ref_vals)
    else:
        nearby_mask = (
            (np.abs(qx - 0.04) <= 0.01)
            & (np.abs(qy - 0.14) <= 0.01)
            & np.isfinite(image)
            & (image > 0)
        )
        if np.any(nearby_mask):
            I_ref = np.mean(image[nearby_mask])
        else:
            I_ref = np.nanmean(image[np.isfinite(image) & (image > 0)])
    print(f"Reference intensity at (~0.04, 0.14): {I_ref:.3f}")

    qy_streak_end = None
    if I_ref > 0 and peak_I_array.size > 0:
        threshold = 0.4 * I_ref
        for i in range(1, len(peak_I_array)):
            if peak_I_array[i] <= threshold:
                qy_streak_end = peak_qy_array[i]
                break
    if qy_streak_end is None:
        qy_streak_end = peak_qy_array[-1] if peak_qy_array.size > 0 else qy_max
    ellipse_major = 0.75*qy_streak_end

    #ellipse geometry parameters
    u, v = compute_murthy_coordinates(qx, qy, A_focal=ellipse_minor, alpha_deg=alpha_deg)
    print(f"Estimated ellipse minor axis length: {ellipse_minor:.4f} Å⁻¹")
    print(f"Estimated ellipse major axis length: {ellipse_major:.4f} Å⁻¹")
    return ellipse_minor, ellipse_major, u, v
            
def plot_ellipse(image, metadata, ellipse_minor, Lm_err, ellipse_major, streak_angle_deg, alpha_deg, alpha_err, A_focal, sample_number=None, twoD_output_directory='.', f=''):
    qx = metadata['qx']
    qy = metadata['qy']
    extent = [qx.min(), qx.max(), qy.min(), qy.max()]
    u, v = compute_murthy_coordinates(qx, qy, A_focal, alpha_deg)
    v_low = 0
    v_high = 80
    vlorad = np.radians(v_low)
    vhighrad= np.radians(v_high)
    mask = (v > vlorad) & (v < vhighrad)
    u_vals = u[mask].ravel()
    int_vals = image[mask].ravel()
    valid = np.isfinite(u_vals) & np.isfinite(int_vals)
    u_vals = u_vals[valid]
    int_vals = int_vals[valid]
    if len(u_vals) == 0:
        print("Cant plot.")
        return
    u_min, u_max = u_vals.min(), u_vals.max()
    alpha_rad = np.radians(alpha_deg)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    #main image

    im = ax.imshow(np.log10(image + 1), cmap='rainbow', origin='lower', extent=extent, alpha=0.8)

    #draw refrence ellipse at u=u_peak
    if ellipse_minor is not None:
        theta = np.linspace(0,2*np.pi, 300)
        qx_ellipse = ellipse_minor * np.sin(theta)
        qy_ellipse = ellipse_major * np.cos(theta)
        #apply alpha rotation to the ellipse
        qx_ring = qx_ellipse * np.cos(alpha_rad) + qy_ellipse * np.sin(alpha_rad)
        qy_ring =  -qx_ellipse * np.sin(alpha_rad) + qy_ellipse * np.cos(alpha_rad)
        ax.plot(qx_ring, qy_ring, 'r-', linewidth=1.5,
                label=f'u = {ellipse_minor:.3f} Å⁻¹  (L = {2*np.pi/ellipse_minor:.1f} Å ± {Lm_err:.1f} Å)')
    v_lines = np.linspace(vlorad, vhighrad, 5)
    u_param = np.linspace(1e-4, ellipse_minor*2.5, 300)
    for i, v_val in enumerate(v_lines):
        qx_hyp = u_param * np.sin(v_val)
        qy_hyp = np.sqrt(ellipse_minor**2 + u_param**2) * np.cos(v_val)
        qx_rot =  qx_hyp * np.cos(alpha_rad) + qy_hyp * np.sin(alpha_rad)
        qy_rot =  -qx_hyp * np.sin(alpha_rad) + qy_hyp * np.cos(alpha_rad)
        #label = f'v = {np.degrees(v_val):.1f}°' if i == 0 else None
        ax.plot(qx_rot, qy_rot, 'w--', linewidth=0.8, alpha=0.7) #, label=label)
        ax.plot(-qx_rot, -qy_rot, 'w--', linewidth=0.8, alpha=0.7)
    #green box where the lamellar peak angle is being evaluated
    qx_min = ellipse_minor - 0.011
    qx_max = ellipse_minor + 0.011
    qy_min = 0
    qy_max = 0.05

    # Draw a green rectangle outline in rotated q-space
    if streak_angle_deg<0:
        theta = -np.radians(-streak_angle_deg)
    else:
        theta = np.radians(streak_angle_deg)
    rect_corners = np.array([
        [qx_min, qy_min],
        [qx_min, qy_max],
        [qx_max, qy_max],
        [qx_max, qy_min],
        [qx_min, qy_min],
    ])
    qx_rect = rect_corners[:, 0] * np.cos(theta) - rect_corners[:, 1] * np.sin(theta)
    qy_rect = rect_corners[:, 0] * np.sin(theta) + rect_corners[:, 1] * np.cos(theta)
    ax.plot(qx_rect, qy_rect, color='green', linestyle='-', linewidth=1.5, alpha=0.8, label='Evaluated Lamellar Peak Region')
    #Lamellar Streak Marker
    if ellipse_minor is not None:
        line_length =0.2
        x0, y0 = ellipse_minor, 0
        dx = np.sin(alpha_rad)* line_length
        dy = np.cos(alpha_rad)* line_length
        x1 = x0 + dx
        y1 = y0 + dy
        ax.plot([x0, x1], [y0, y1], color='white', linewidth=1, label=f"Lamellar Streak at α = {alpha_deg:.3f}± {alpha_err:.3f}°")
    #Center Streak Marker
    streak_line_length = 0.2
    p0x, p0y = 0, 0
    pdx = -np.sin(np.radians(streak_angle_deg)) * streak_line_length
    pdy = np.cos(np.radians(streak_angle_deg)) * streak_line_length
    px1 = p0x + pdx
    py1 = p0y + pdy
    ax.plot([p0x, px1], [p0y, py1], color='lightblue', linewidth=1, label=f"SAXS pattern off center by {streak_angle_deg:.3f}°") 

    # Center streak length line from beam center along the detected streak
    streak_length, streak_qy_min, streak_qy_max = center_streak_length(qx, qy, image, streak_angle_deg)
    if np.isfinite(streak_length):
        if streak_angle_deg < 0:
            theta = -np.radians(-streak_angle_deg)
        else:
            theta = np.radians(streak_angle_deg)
        axis_dir = np.array([-np.sin(theta), np.cos(theta)])
        endpoint_min = axis_dir * streak_qy_min
        endpoint_max = axis_dir * streak_qy_max
        ax.plot([endpoint_min[0], endpoint_max[0]], [endpoint_min[1], endpoint_max[1]],
                color='green', linestyle=':', linewidth=1.5,
                label=f'Center streak length = {streak_length:.4f} Å⁻¹')
        ax.scatter([endpoint_min[0], endpoint_max[0]], [endpoint_min[1], endpoint_max[1]],
                   color='cyan', s=25, marker='o')

    #Beam center marker 
    ax.scatter(0, 0, color='blue', s=80, marker='+',
               linewidths=2, label='Beam center (0, 0)')
    ax.set_xlabel('q_x (Å⁻¹)')
    ax.set_ylabel('q_y (Å⁻¹)')
    sample_text = f"Sample: {sample_number} - " if sample_number is not None else ""
    plt.title(f"\n{sample_text} Elliptical Trajectory Overlay\n")
    ax.legend()
    fig.colorbar(im, ax=ax, label='log₁₀(Intensity)')
    output_file = os.path.join(twoD_output_directory, os.path.basename(f).replace('.npz', '_2D_pattern.png'))
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    #plt.show()

if __name__ == "__main__":
    
    dat_directory = r'C:\Users\kduff\OneDrive\Documents\MD_TD_data\saxs\analysis\TD MDO q image'
    output_directory = r'C:\Users\kduff\OneDrive\Documents\fit calcs\Elliptical Fits 2\June 1\streak'
    twoD_output_directory = r'C:\Users\kduff\OneDrive\Documents\fit calcs\Elliptical Fits 2\June 1\streak'
    mask_path = r'C:\Users\kduff\OneDrive\Documents\MD_TD_data\data\SAXS_mask.png'
    beam_center = (738, 549)  # (x, y) pixel coordinates of beam center
    
    dat_files = sorted(glob.glob(os.path.join(dat_directory, "*.npz")))
    
    import time
    results_list = []
    for f in dat_files:
        try:
            print(f"\nFitting: {f}")
            img, _, metadata = load_saxs_npz(f, beam_center=beam_center)
            qx = metadata['qx']
            qy = metadata['qy']
            
            img = apply_saxs_mask(img, mask_path, nan_masked=True,
                             flip_vertical=True, flip_horizontal=False,
                             rotate_deg=0)
            
            streak_angle_deg, x_axis_degrees = pattern_rotation()
            Lm, u_peak_approx, u_peak_err, I_peak, Lm_err = A_peak(f, beam_center=beam_center)
            alpha_deg, alpha_err = alpha(f, u_peak_approx, beam_center=beam_center)
            alpha_deg = np.abs(alpha_deg)  # Ensure alpha is positive, remove for   
            alpha_err = np.abs(alpha_err)
            streak_length, streak_qy_min, streak_qy_max = center_streak_length(qx, qy, img, streak_angle_deg)
            print(f"Estimated alpha angle: {alpha_deg:.3f}° ± {alpha_err:.3f}°")
            print(f"Estimated ellipse minor: {u_peak_approx:.5f} ± {u_peak_err:.5f} Å⁻¹")
            print(f"Estimated long period Lm: {Lm:.1f} ± {Lm_err:.1f} Å")
            print(f"Estimated center streak length: {streak_length:.5f} Å⁻¹ (qy range {streak_qy_min:.5f} to {streak_qy_max:.5f})")
            ellipse_minor, ellipse_major, u, v = ellipse_geometry(u_peak_approx, alpha_deg, img, metadata, streak_angle_deg)

            file_base = os.path.basename(f)
            match = re.search(r'\d{4}', file_base)
            sample_number = match.group(0) if match else file_base[:4]
            
        
            plot_ellipse(img, metadata, ellipse_minor, Lm_err, ellipse_major, streak_angle_deg, alpha_deg, alpha_err, A_focal=ellipse_minor, sample_number=sample_number, twoD_output_directory=twoD_output_directory, f=f)
            # diagnostics
            x_scale = metadata.get('x_scale', np.nan)
            y_scale = metadata.get('y_scale', np.nan)
            beam_center_used = metadata.get('beam_center', beam_center)
            valid_pixel_count = int(np.sum(np.isfinite(img) & (img > 0)))
            # sample reference intensity near (qx,qy) = (0.04, 0.14)
            qx = metadata['qx']
            qy = metadata['qy']
            ref_tolerance = 0.002
            ref_mask = (
                (np.abs(qx - 0.04) <= ref_tolerance)
                & (np.abs(qy - 0.14) <= ref_tolerance)
                & np.isfinite(img)
                & (img > 0)
            )
            I_ref_vals = img[ref_mask]
            I_ref = float(np.nan) if I_ref_vals.size == 0 else float(np.mean(I_ref_vals))

            results_list.append({
                'filename' : os.path.basename(f),
                'streak_angle_deg' : streak_angle_deg,
                'streak_length' : streak_length,
                #'x_axis_degrees' : x_axis_degrees,
                'alpha_deg' : alpha_deg,
                'alpha_err' : alpha_err,
                'center_streak_qy_min' : streak_qy_min,
                'center_streak_qy_max' : streak_qy_max,
                'ellipse_minor' : ellipse_minor,
                'ellipse_minor_err' : u_peak_err,
                'ellipse_major' : ellipse_major,
                'Lm' : Lm,
                'Lm_err' : Lm_err,
                #'u_peak_approx' : u_peak_approx,
                #'I_peak' : I_peak,
                #'I_ref' : I_ref,
                #'x_scale' : x_scale,
                #'y_scale' : y_scale,
                #'beam_center' : str(beam_center_used),
                #'valid_pixel_count' : valid_pixel_count
            })
        except Exception as e:
            import traceback
            print(f"\nFailed. Exception: {e}")
            traceback.print_exc()
        time.sleep(0.1)

    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(output_directory, "streaklength_results.csv"), index=False)
    print(f"\nProcessing complete! Output saved to '{output_directory}/'")
