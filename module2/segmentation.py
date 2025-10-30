import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- Thresholding ----------
def global_threshold(img, thresh_value=127):
    """Apply simple global thresholding."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)
    return binary

def adaptive_threshold(img):
    """Apply adaptive Gaussian thresholding."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return binary

def otsu_threshold(img):
    """Automatically find optimal threshold using Otsu’s method."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# ---------- Edge Detection ----------
def sobel_edge(img):
    """Detect edges using Sobel operator."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(grad_x, grad_y)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    return sobel

def prewitt_edge(img):
    """Detect edges using Prewitt operator."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Prewitt kernels
    kernelx = np.array([[ -1,  0,  1],
                        [ -1,  0,  1],
                        [ -1,  0,  1]])
    
    kernely = np.array([[ -1, -1, -1],
                        [  0,  0,  0],
                        [  1,  1,  1]])


    grad_x = cv2.filter2D(img, -1, kernelx)
    grad_y = cv2.filter2D(img, -1, kernely)

    prewitt = cv2.magnitude(grad_x.astype(float), grad_y.astype(float))
    prewitt = np.uint8(np.clip(prewitt, 0, 255))
    return prewitt

def canny_edge(img, low=100, high=200):
    """Canny edge detection."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, low, high)
    return edges

def region_growing(img, seed_point=(100, 100), threshold=5):
    """Simple placeholder for region growing segmentation."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = np.zeros_like(img)
    output[seed_point] = 255  # just mark seed for now
    return output


# ---------- Morphological Operations ----------
def morphological_ops(img, operation='dilate', kernel_size=3, iterations=1):
    """Perform dilation, erosion, opening, or closing."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'dilate':
        return cv2.dilate(img, kernel, iterations=iterations)
    elif operation == 'erode':
        return cv2.erode(img, kernel, iterations=iterations)
    elif operation == 'open':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError("Invalid operation type.")

# ---------- Contour Detection ----------
def find_contours(img):
    """Find and draw contours on the image."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    return contour_img

# ---------- Visualization ----------
def show_comparison(original, processed, title1="Original", title2="Processed"):
    """Display two images side-by-side."""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    if len(original.shape) == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if len(processed.shape) == 3:
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(processed, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    plt.show()
