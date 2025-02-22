import numpy as np
from numba import njit, prange
import cv2
from scipy import ndimage


@njit
def linear_stretching_3d(volume, l=256):
    """Apply linear stretching to normalize 3D volume intensity values."""
    vol_min = np.min(volume)
    vol_max = np.max(volume)

    # Avoid division by zero
    if vol_max == vol_min:
        return volume.astype(np.uint8)

    return ((l - 1) * (volume - vol_min) / (vol_max - vol_min)).astype(np.uint8)


@njit
def compute_histogram_3d(volume, l=256):
    """Compute normalized histogram of a 3D volume."""
    hist = np.zeros(l, dtype=np.float32)
    total_voxels = volume.shape[0] * volume.shape[1] * volume.shape[2]

    for z in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for x in range(volume.shape[2]):
                hist[volume[z, y, x]] += 1

    return hist / total_voxels


@njit
def compute_mapping(hist, l=256):
    """Compute mapping function based on modified histogram."""
    cum_sum = 0
    t = np.zeros(l, dtype=np.int32)

    for i in range(l):
        cum_sum += hist[i]
        t[i] = int((l - 1) * cum_sum + 0.5)

    return t


@njit
def apply_mapping_3d(volume, mapping):
    """Apply mapping function to 3D volume."""
    result = np.zeros_like(volume, dtype=np.uint8)

    for z in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for x in range(volume.shape[2]):
                result[z, y, x] = mapping[volume[z, y, x]]

    return result


@njit
def compute_tone_distortion(mapping, hist, l=256):
    """Compute tone distortion of a mapping function."""
    max_distortion = 0

    for i in range(l):
        if hist[i] == 0:
            continue

        for j in range(i):
            if hist[j] == 0:
                continue

            if mapping[i] == mapping[j]:
                distortion = i - j
                max_distortion = max(max_distortion, distortion)

    return max_distortion


def optimize_lambda(volume_u8, lambda_range=np.linspace(0.1, 10, 20)):
    """Find optimal lambda for global enhancement by minimizing tone distortion."""
    hist = compute_histogram_3d(volume_u8)
    uniform_hist = np.ones(256) / 256

    min_distortion = float('inf')
    optimal_lambda = 1.0

    for lam in lambda_range:
        modified_hist = (1.0 / (1.0 + lam)) * hist + (lam / (1.0 + lam)) * uniform_hist
        mapping = compute_mapping(modified_hist)
        distortion = compute_tone_distortion(mapping, hist)

        if distortion < min_distortion:
            min_distortion = distortion
            optimal_lambda = lam

    return optimal_lambda


def global_contrast_adaptive_enhancement_3d(volume, lambda_param=None):
    """Apply global contrast adaptive enhancement to 3D volume."""
    # Convert input to uint8 if not already
    if volume.dtype != np.uint8:
        volume_u8 = linear_stretching_3d(volume)
    else:
        volume_u8 = volume.copy()

    # Find optimal lambda if not provided
    if lambda_param is None:
        lambda_param = optimize_lambda(volume_u8)

    # Compute histogram
    hist = compute_histogram_3d(volume_u8)

    # Create uniform histogram
    uniform_hist = np.ones(256) / 256

    # Compute modified histogram (weighted sum)
    modified_hist = (1.0 / (1.0 + lambda_param)) * hist + (lambda_param / (1.0 + lambda_param)) * uniform_hist

    # Compute mapping function
    mapping = compute_mapping(modified_hist)

    # Apply mapping to volume
    result = apply_mapping_3d(volume_u8, mapping)

    return result


@njit
def compute_3d_intensity(r_volume, g_volume, b_volume):
    """Compute intensity channel from RGB volumes."""
    depth, height, width = r_volume.shape
    intensity = np.zeros((depth, height, width), dtype=np.uint8)

    for z in range(depth):
        for y in range(height):
            for x in range(width):
                # Standard RGB to intensity conversion
                intensity[z, y, x] = int(0.299 * r_volume[z, y, x] +
                                         0.587 * g_volume[z, y, x] +
                                         0.114 * b_volume[z, y, x])

    return intensity


@njit
def hue_preservation_3d(enhanced_intensity, original_intensity, original_channel):
    """Apply hue-preserving transformation to a 3D color channel."""
    depth, height, width = enhanced_intensity.shape
    result = np.zeros_like(original_channel, dtype=np.uint8)

    for z in range(depth):
        for y in range(height):
            for x in range(width):
                # Handle divisions by zero and edge cases
                if enhanced_intensity[z, y, x] <= original_intensity[z, y, x]:
                    if original_intensity[z, y, x] == 0:
                        result[z, y, x] = 0
                    else:
                        ratio = enhanced_intensity[z, y, x] / original_intensity[z, y, x]
                        result[z, y, x] = min(255, max(0, int(ratio * original_channel[z, y, x])))
                else:
                    if 255 - original_intensity[z, y, x] == 0:
                        result[z, y, x] = original_channel[z, y, x]
                    else:
                        ratio = (255 - enhanced_intensity[z, y, x]) / (255 - original_intensity[z, y, x])
                        delta = original_channel[z, y, x] - original_intensity[z, y, x]
                        result[z, y, x] = min(255, max(0, int(ratio * delta + enhanced_intensity[z, y, x])))

    return result


def local_contrast_adaptive_enhancement_3d(volume, kernel_size=3, clip_limit=2.0):
    """Apply local contrast adaptive enhancement to 3D volume using local processing."""
    if volume.dtype != np.uint8:
        volume_u8 = linear_stretching_3d(volume)
    else:
        volume_u8 = volume.copy()

    # Process each slice using CLAHE
    result = np.zeros_like(volume_u8, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    for z in range(volume_u8.shape[0]):
        result[z] = clahe.apply(volume_u8[z])

    return result


@njit
def compute_3d_laplacian(volume):
    """Compute 3D Laplacian filter response for contrast measurement."""
    depth, height, width = volume.shape
    laplacian = np.zeros_like(volume, dtype=np.float32)

    # 3D Laplacian kernel weights
    center_weight = -6.0
    neighbor_weight = 1.0

    for z in range(1, depth - 1):
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Center voxel
                center_val = volume[z, y, x]

                # 6-connected neighbors
                neighbors_sum = (
                        volume[z - 1, y, x] + volume[z + 1, y, x] +
                        volume[z, y - 1, x] + volume[z, y + 1, x] +
                        volume[z, y, x - 1] + volume[z, y, x + 1]
                )

                # Compute Laplacian response
                laplacian[z, y, x] = abs(center_weight * center_val + neighbor_weight * neighbors_sum)

    return laplacian


def compute_weights_3d(volume, sigma=0.2):
    """Compute weights for fusion based on contrast and brightness."""
    # Calculate contrast using 3D Laplacian filter
    contrast = compute_3d_laplacian(volume)

    # Normalize contrast to [0, 1]
    max_contrast = np.max(contrast)
    if max_contrast > 0:
        contrast = contrast / max_contrast

    # Calculate brightness weight using Gaussian curve
    volume_norm = volume / 255.0  # Normalize to [0, 1]
    brightness = np.exp(-((volume_norm - 0.5) ** 2) / (2 * sigma ** 2))

    # Final weight is minimum of contrast and brightness
    weight = np.minimum(contrast, brightness)
    return weight


def fusion_blend_3d(global_enhanced, local_enhanced, w_global_norm):
    """Blend two 3D volumes using direct weighting."""
    w_local_norm = 1.0 - w_global_norm
    result = global_enhanced * w_global_norm + local_enhanced * w_local_norm
    return np.clip(result, 0, 255).astype(np.uint8)


def global_and_local_contrast_enhancement_3d_rgb(r_volume, g_volume, b_volume, lambda_param=None, clip_limit=2.0):
    """
    Apply global and local contrast adaptive enhancement to 3D RGB volumes.

    Args:
        r_volume: 3D numpy array for red channel (uint8 or other)
        g_volume: 3D numpy array for green channel (uint8 or other)
        b_volume: 3D numpy array for blue channel (uint8 or other)
        lambda_param: Parameter for global enhancement (higher = more uniform histogram)
                      If None, will be optimized automatically
        clip_limit: Threshold for contrast limiting in CLAHE

    Returns:
        Tuple of enhanced 3D volumes (r, g, b) as uint8 numpy arrays
    """
    # Ensure all inputs are uint8
    if r_volume.dtype != np.uint8:
        r_u8 = linear_stretching_3d(r_volume)
    else:
        r_u8 = r_volume.copy()

    if g_volume.dtype != np.uint8:
        g_u8 = linear_stretching_3d(g_volume)
    else:
        g_u8 = g_volume.copy()

    if b_volume.dtype != np.uint8:
        b_u8 = linear_stretching_3d(b_volume)
    else:
        b_u8 = b_volume.copy()

    # Compute intensity volume
    intensity = compute_3d_intensity(r_u8, g_u8, b_u8)

    # Global enhancement of intensity
    global_enhanced_i = global_contrast_adaptive_enhancement_3d(intensity, lambda_param)

    # Apply hue preservation to get color channels
    global_enhanced_r = hue_preservation_3d(global_enhanced_i, intensity, r_u8)
    global_enhanced_g = hue_preservation_3d(global_enhanced_i, intensity, g_u8)
    global_enhanced_b = hue_preservation_3d(global_enhanced_i, intensity, b_u8)

    # Local enhancement of intensity
    local_enhanced_i = local_contrast_adaptive_enhancement_3d(intensity, clip_limit=clip_limit)

    # Apply hue preservation to get color channels
    local_enhanced_r = hue_preservation_3d(local_enhanced_i, intensity, r_u8)
    local_enhanced_g = hue_preservation_3d(local_enhanced_i, intensity, g_u8)
    local_enhanced_b = hue_preservation_3d(local_enhanced_i, intensity, b_u8)

    # Compute fusion weights based on intensity
    w_global = compute_weights_3d(global_enhanced_i)
    w_local = compute_weights_3d(local_enhanced_i)

    # Normalize weights
    sum_weights = w_global + w_local
    sum_weights[sum_weights == 0] = 1  # Avoid division by zero
    w_global_norm = w_global / sum_weights

    # Fusion for each channel
    r_result = fusion_blend_3d(global_enhanced_r, local_enhanced_r, w_global_norm)
    g_result = fusion_blend_3d(global_enhanced_g, local_enhanced_g, w_global_norm)
    b_result = fusion_blend_3d(global_enhanced_b, local_enhanced_b, w_global_norm)

    return r_result, g_result, b_result


def global_and_local_contrast_enhancement_3d(volume, lambda_param=None, clip_limit=2.0):
    """
    Apply global and local contrast adaptive enhancement to a single 3D grayscale volume.

    Args:
        volume: Input 3D numpy array (uint8 or other)
        lambda_param: Parameter for global enhancement (higher = more uniform histogram)
                      If None, will be optimized automatically
        clip_limit: Threshold for contrast limiting in CLAHE

    Returns:
        Enhanced 3D volume as uint8 numpy array
    """
    # Ensure input is uint8
    if volume.dtype != np.uint8:
        volume_u8 = linear_stretching_3d(volume)
    else:
        volume_u8 = volume.copy()

    # Global enhancement
    global_enhanced = global_contrast_adaptive_enhancement_3d(volume_u8, lambda_param)

    # Local enhancement
    local_enhanced = local_contrast_adaptive_enhancement_3d(volume_u8, clip_limit=clip_limit)

    # Compute fusion weights
    w_global = compute_weights_3d(global_enhanced)
    w_local = compute_weights_3d(local_enhanced)

    # Normalize weights
    sum_weights = w_global + w_local
    sum_weights[sum_weights == 0] = 1  # Avoid division by zero
    w_global_norm = w_global / sum_weights

    # Fusion
    result = fusion_blend_3d(global_enhanced, local_enhanced, w_global_norm)

    return result

