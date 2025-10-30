# module1/enhancement.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from scipy import fftpack


# =========================
# Histogram Processing
# =========================
def histogram_equalization(img):
    """
    Apply histogram equalization.
    - Grayscale: cv2.equalizeHist directly.
    - Color: equalize Y channel in YCrCb to preserve color balance.
    """
    if img.ndim == 2:
        return cv2.equalizeHist(img)

    # Color image → equalize luminance only
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    out = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def histogram_matching(source, reference):
    """
    Match the histogram of 'source' to 'reference'.
    Works for grayscale and color images.
    """
    if source.ndim == 2 and reference.ndim == 2:
        matched = exposure.match_histograms(source, reference, channel_axis=None)
    else:
        # Ensure both are 3-channel BGR for consistent behavior
        if source.ndim == 2:
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        if reference.ndim == 2:
            reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
        matched = exposure.match_histograms(source, reference, channel_axis=-1)

    matched = np.clip(matched, 0, 255)
    return matched.astype(np.uint8)


def adjust_brightness_contrast(img, brightness=0, contrast=0):
    """
    Adjust brightness and contrast using a linear gray-level transform.
    out = img * (contrast/127 + 1) - contrast + brightness
    - brightness: [-255, 255]
    - contrast:   [-127, 127]
    """
    img16 = img.astype(np.int16)
    out = img16 * (contrast / 127.0 + 1.0) - contrast + brightness
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# =========================
# Spatial Filtering
# =========================
def mean_filter(img, kernel_size=3):
    """Mean (average) filter for smoothing / noise reduction."""
    return cv2.blur(img, (kernel_size, kernel_size))


def median_filter(img, kernel_size=3):
    """Median filter for salt-and-pepper noise reduction."""
    # kernel_size must be odd and >= 3
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    k = max(3, k)
    return cv2.medianBlur(img, k)


def laplacian_sharpen(img, alpha=0.7):
    """
    Sharpen via Laplacian: sharp = img - alpha * Laplacian(img)
    Handles both grayscale and color images.
    """
    if img.ndim == 2:
        lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
        sharp = img.astype(np.float32) - alpha * lap
        return np.clip(sharp, 0, 255).astype(np.uint8)

    # Color: apply per channel
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    sharp = img.astype(np.float32) - alpha * lap
    return np.clip(sharp, 0, 255).astype(np.uint8)


def high_pass_filter(img):
    """
    3x3 high-pass kernel to enhance edges.
    Works on grayscale or color.
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=np.float32)
    return cv2.filter2D(img, ddepth=-1, kernel=kernel)


# =========================
# Frequency Domain Filtering
# =========================
def _radial_mask(shape, kind='low', cutoff=30, bandwidth=10):
    """
    Build an ideal (binary) radial mask:
      - 'low'  : pass D <= cutoff
      - 'high' : pass D >= cutoff
      - 'band' : pass (cutoff - bw/2) <= D <= (cutoff + bw/2)
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    y = np.arange(rows) - crow
    x = np.arange(cols) - ccol
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)

    if kind == 'low':
        return (D <= cutoff).astype(np.float32)
    elif kind == 'high':
        return (D >= cutoff).astype(np.float32)
    elif kind == 'band':
        low = max(0, cutoff - bandwidth / 2.0)
        high = cutoff + bandwidth / 2.0
        return ((D >= low) & (D <= high)).astype(np.float32)
    else:
        raise ValueError("filter_type must be 'low', 'high', or 'band'.")


def frequency_filter(img, filter_type='low', cutoff=30, bandwidth=10):
    """
    Fourier-based filtering (ideal radial masks).
      - filter_type: 'low', 'high', or 'band'
      - cutoff:      radius in pixels (for 'band', center frequency)
      - bandwidth:   band thickness (used when filter_type='band')

    Steps: FFT → shift → multiply by mask → inverse FFT → normalize to [0,255].
    Operates on grayscale; color images are converted to grayscale first.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_f = img.astype(np.float32)
    F = fftpack.fft2(img_f)
    Fshift = fftpack.fftshift(F)

    M = _radial_mask(img.shape, kind=filter_type, cutoff=cutoff, bandwidth=bandwidth)
    Ffilt = Fshift * M

    ishift = fftpack.ifftshift(Ffilt)
    img_back = np.real(fftpack.ifft2(ishift))

    # Normalize to 0-255 for display
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back.astype(np.uint8)


# =========================
# Display Utility
# =========================
def show_comparison(original, processed, title1="Original", title2="Processed"):
    """
    Display two images side-by-side (handles gray or color).
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    if original.ndim == 2:
        plt.imshow(original, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if processed.ndim == 2:
        plt.imshow(processed, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    plt.title(title2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
