import cv2
import os
import tkinter as tk
from tkinter import filedialog, Toplevel, messagebox
from PIL import Image, ImageTk
import numpy as np


# === Import your custom modules ===
from module1.enhancement import (
    histogram_equalization,
    mean_filter,
    median_filter,
    laplacian_sharpen,
    high_pass_filter,
)
from module2.segmentation import (
    sobel_edge,
    prewitt_edge,
    canny_edge,
    otsu_threshold,
    region_growing,
)
from module3.transform import (
    rotate,
    scale,
    translate,
    affine_transform,
    perspective_transform,
)

# === MAIN WINDOW ===

root = tk.Tk()
root.title("Handwritten Recognizer Toolbox by Team")
root.geometry("950x750")
root.configure(bg="#f3f3f3")

title_label = tk.Label(root, text="Handwritten Recognizer Toolbox by Team No Name",
                       font=("Arial", 16, "bold"), bg="#000000")
title_label.pack(pady=10)

# This is just a container label to preview images
PREVIEW_MAX_W, PREVIEW_MAX_H = 900, 600
image_label = tk.Label(root, bg="lightgray")
image_label.pack(pady=10)

loaded_image = None
display_image = None  # keep a reference!


def show_image(img):
    """
    Display image (gray or BGR) resized to fit the preview area.
    """
    global display_image

    if img is None:
        return

    # Accept both grayscale and color
    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    scale = min(PREVIEW_MAX_W / max(w, 1), PREVIEW_MAX_H / max(h, 1), 1.0)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    pil_img = Image.fromarray(rgb).resize(new_size, Image.LANCZOS)

    display_image = ImageTk.PhotoImage(pil_img)          # strong ref
    image_label.configure(image=display_image)
    image_label.image = display_image                     # extra safety


def _need_image_guard():
    if loaded_image is None:
        messagebox.showwarning("No image", "Please load an image first.")
        return True
    return False


def load_image():
    global loaded_image
    file_path = filedialog.askopenfilename(
        title="Select an image",
        initialdir=os.path.expanduser("~"),
        filetypes=[
            ("Image files", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")),
            ("All files", "*.*"),
        ],
    )
    if not file_path:
        return

    loaded_image = cv2.imread(file_path)
    if loaded_image is None:
        messagebox.showerror("Error", "Failed to read the image file.")
        return
    show_image(loaded_image)

    
# ==========================================================
#  MODULE 1: Image Enhancement
# ==========================================================
def open_module1():
    if _need_image_guard(): return
    win = Toplevel(root)
    win.title("Module 1 - Enhancement Tools")
    win.geometry("400x400")
    win.configure(bg="#f7f7f7")

    def apply_hist_eq():
        gray = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
        result = histogram_equalization(gray)
        show_image(result)

    def apply_mean():
        result = mean_filter(loaded_image, 5)
        show_image(result)

    def apply_median():
        result = median_filter(loaded_image, 5)
        show_image(result)

    def apply_laplacian():
        gray = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
        result = laplacian_sharpen(gray)
        show_image(result)

    def apply_highpass():
        gray = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
        result = high_pass_filter(gray)
        show_image(result)

    tk.Label(win, text="Enhancement Techniques", font=("Arial", 12, "bold"), bg="#f7f7f7").pack(pady=10)
    tk.Button(win, text="Histogram Equalization", width=25, command=apply_hist_eq).pack(pady=5)
    tk.Button(win, text="Mean Filter", width=25, command=apply_mean).pack(pady=5)
    tk.Button(win, text="Median Filter", width=25, command=apply_median).pack(pady=5)
    tk.Button(win, text="Laplacian Sharpen", width=25, command=apply_laplacian).pack(pady=5)
    tk.Button(win, text="High-pass Filter", width=25, command=apply_highpass).pack(pady=5)

# ==========================================================
#  MODULE 2: Segmentation & Edge Detection
# ==========================================================
def open_module2():
    if _need_image_guard(): return
    win = Toplevel(root)
    win.title("Module 2 - Segmentation & Edge Detection")
    win.geometry("400x400")
    win.configure(bg="#f7f7f7")

    def sobel():
        result = sobel_edge(loaded_image)     # grayscale output
        show_image(result)

    def prewitt():
        result = prewitt_edge(loaded_image)   # grayscale output
        show_image(result)

    def canny():
        result = canny_edge(loaded_image)     # grayscale output
        show_image(result)

    def otsu():
        result = otsu_threshold(loaded_image) # grayscale output
        show_image(result)

    tk.Label(win, text="Edge Detection", font=("Arial", 12, "bold"), bg="#f7f7f7").pack(pady=10)
    tk.Button(win, text="Sobel Edge", width=25, command=sobel).pack(pady=5)
    tk.Button(win, text="Prewitt Edge", width=25, command=prewitt).pack(pady=5)
    tk.Button(win, text="Canny Edge", width=25, command=canny).pack(pady=5)

    tk.Label(win, text="Segmentation", font=("Arial", 12, "bold"), bg="#f7f7f7").pack(pady=10)
    tk.Button(win, text="Otsu Threshold", width=25, command=otsu).pack(pady=5)

# ==========================================================
#  MODULE 3: Transformation
# ==========================================================
def open_module3():
    if _need_image_guard(): return
    win = Toplevel(root)
    win.title("Module 3 - Transformation Tools")
    win.geometry("400x400")
    win.configure(bg="#f7f7f7")

    def rotate_action():
        result = rotate(loaded_image, 45)
        show_image(result)

    def scale_action():
        result = scale(loaded_image, 1.5, 1.5)
        show_image(result)

    def translate_action():
        result = translate(loaded_image, 50, 50)
        show_image(result)

    def affine_action():
        result = affine_transform(loaded_image)
        show_image(result)

    def perspective_action():
        result = perspective_transform(loaded_image)
        show_image(result)

    tk.Label(win, text="Geometric Transformations", font=("Arial", 12, "bold"), bg="#f7f7f7").pack(pady=10)
    tk.Button(win, text="Rotate 45°", width=25, command=rotate_action).pack(pady=5)
    tk.Button(win, text="Scale 1.5x", width=25, command=scale_action).pack(pady=5)
    tk.Button(win, text="Translate (50,50)", width=25, command=translate_action).pack(pady=5)
    tk.Button(win, text="Affine Transform", width=25, command=affine_action).pack(pady=5)
    tk.Button(win, text="Perspective Transform", width=25, command=perspective_action).pack(pady=5)

# ==========================================================
# MAIN BUTTONS
# ==========================================================
btn_frame = tk.Frame(root, bg="#f3f3f3")
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Load Image", width=15, command=load_image).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Enhance (Module1)", width=18, command=open_module1).grid(row=0, column=1, padx=10)
tk.Button(btn_frame, text="Segment/Edge (Module2)", width=20, command=open_module2).grid(row=0, column=2, padx=10)
tk.Button(btn_frame, text="Transform (Module3)", width=20, command=open_module3).grid(row=0, column=3, padx=10)

root.mainloop()