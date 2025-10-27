import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from scipy import fftpack

# ---------- Histogram Processing ----------
def histogram_equalization(img):
    """Apply histogram equalization to grayscale or color images."""
    if len(img.shape) == 3:  # Convert to grayscale first
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)
    else:
        return cv2.equalizeHist(img)

def histogram_matching(source, reference):
    """Match the histogram of the source image to the reference image."""
    matched = exposure.match_histograms(source, reference, channel_axis=None)
    return np.uint8(matched)

def adjust_brightness_contrast(img, brightness=0, contrast=0):
    """Adjust brightness and contrast."""
    img = np.int16(img)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    return np.uint8(img)

# ---------- Spatial Filtering ----------
def mean_filter(img, kernel_size=3):
    return cv2.blur(img, (kernel_size, kernel_size))

def median_filter(img, kernel_size=3):
    return cv2.medianBlur(img, kernel_size)

def laplacian_sharpen(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    sharp = img - 0.7 * lap
    sharp = np.clip(sharp, 0, 255)
    return np.uint8(sharp)

def high_pass_filter(img):
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

# ---------- Frequency Domain Filtering ----------
def frequency_filter(img, filter_type='low', cutoff=30):
    """Apply frequency-domain filter: low, high, or band."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = fftpack.fft2(img)
    fshift = fftpack.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.uint8)
    if filter_type == 'low':
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
    elif filter_type == 'high':
        mask[:] = 1
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0
    elif filter_type == 'band':
        mask[crow - cutoff - 10:crow + cutoff + 10, ccol - cutoff - 10:ccol + cutoff + 10] = 1
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0

    fshift_filtered = fshift * mask
    img_back = np.abs(fftpack.ifft2(fftpack.ifftshift(fshift_filtered)))
    img_back = np.uint8(np.clip(img_back, 0, 255))
    return img_back

# ---------- Display Utility ----------
def show_comparison(original, processed, title1="Original", title2="Processed"):
    """Display two images side by side."""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    if len(processed.shape) == 2:
        plt.imshow(processed, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    plt.title(title2)
    plt.axis('off')
    plt.show()
