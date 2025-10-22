# transform.py
import cv2
import numpy as np

def translate_image(img, tx, ty):
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def scale_image(img, sx, sy, interp=cv2.INTER_LINEAR):
    return cv2.resize(img, None, fx=sx, fy=sy, interpolation=interp)

def rotate_image(img, angle_deg, center=None, scale=1.0, interp=cv2.INTER_LINEAR):
    rows, cols = img.shape[:2]
    if center is None:
        center = (cols/2, rows/2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    return cv2.warpAffine(img, M, (cols, rows), flags=interp)

def affine_transform(img, src_pts, dst_pts, output_size=None, interp=cv2.INTER_LINEAR):
    # src_pts and dst_pts: list of three points each [[x1,y1],...]
    M = cv2.getAffineTransform(np.float32(src_pts), np.float32(dst_pts))
    if output_size is None:
        h,w = img.shape[:2]
        output_size = (w,h)
    return cv2.warpAffine(img, M, output_size, flags=interp)

def radial_distortion_correction(img, k1=0.0, k2=0.0):
    # Basic undistort using estimated camera matrix and distortion coef
    h,w = img.shape[:2]
    # assume fx=fy, cx=center
    fx = 0.9 * w
    camera_matrix = np.array([[fx, 0, w/2],
                              [0, fx, h/2],
                              [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
    new_cam, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_cam)
    return undistorted

# Simple implementations of interpolation wrappers are provided via OpenCV flags:
INTERP_MAP = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC
}
