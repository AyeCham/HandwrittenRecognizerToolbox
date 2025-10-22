# enhancement.py
import numpy as np
import cv2
from skimage import exposure

def hist_equalize(img):
    """Histogram equalization for grayscale or RGB."""
    if img is None: 
        return None
    if len(img.shape) == 3 and img.shape[2] == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return out
    else:
        return cv2.equalizeHist(img)

def hist_match(source, template):
    """Histogram matching (skimage exposure.match_histograms)"""
    matched = exposure.match_histograms(source, template, multichannel=(len(source.shape) == 3))
    # skimage returns float in [0,1] for floats - convert to uint8
    if matched.dtype != np.uint8:
        matched = np.clip(matched*255, 0, 255).astype(np.uint8)
    return matched

def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    """Linear transform: out = alpha * img + beta. alpha contrast, beta brightness"""
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out

def gamma_correction(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** invGamma) * 255
                       for i in np.arange(0, 256)])).astype("uint8")
    return cv2.LUT(img, table)

def mean_filter(img, k=3):
    return cv2.blur(img, (k,k))

def median_filter(img, k=3):
    return cv2.medianBlur(img, k)

def laplacian_sharpen(img):
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharp = cv2.convertScaleAbs(gray - 0.5*lap)
    if len(img.shape) == 3:
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    return sharp

def frequency_filter(img, kind='low', cutoff=30, order=2):
    """Simple gaussian low/high-pass in freq domain for grayscale images."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    # create Gaussian mask
    u = np.arange(-crow, crow)
    v = np.arange(-ccol, ccol)
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    H_low = np.exp(-(D**2) / (2*(cutoff**2)))
    if kind == 'low':
        H = H_low
    elif kind == 'high':
        H = 1 - H_low
    elif kind == 'band':
        # band: exclude low < r < high (we interpret cutoff as tuple)
        rl, rh = cutoff if isinstance(cutoff, (list, tuple)) else (cutoff, cutoff*2)
        H = np.logical_and(D>=rl, D<=rh).astype(float)
    else:
        H = H_low
    G = fshift * H
    img_back = np.fft.ifftshift(G)
    inv = np.fft.ifft2(img_back)
    inv = np.abs(inv)
    inv = np.uint8(255 * (inv - inv.min()) / (inv.max() - inv.min() + 1e-8))
    return inv
