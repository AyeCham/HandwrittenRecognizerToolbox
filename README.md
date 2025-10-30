# Handwritten Recognizer Toolbox (Tkinter GUI)

A small desktop app for basic image processing:
- **Enhancement:** histogram equalization, mean/median, Laplacian sharpen, high-pass  
- **Segmentation & Edges:** Sobel, Prewitt, Canny, Otsu  
- **Transforms:** rotate, scale, translate, affine, perspective

---

## 1) Prerequisites
- **Python 3.12+**
- macOS/Windows/Linux supported
- Tkinter is included with most Python installs  
  - Linux may need: `sudo apt-get install -y python3-tk`

---

## 2) Clone the repository
```bash
git clone <https://github.com/AyeCham/HandwrittenRecognizerToolbox>.git
cd <YOUR-CLONED-REPO-FOLDER>
```
---


## 3) Create and activate a virtual environment
- macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate 
```

- Windows (PowerShell)
```bash
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```
- Your prompt should show (venv) when activated.

## 4) Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5) Run the app
```bash
python gui_main.py
```

Then click Load Image and choose a local file (.png .jpg .jpeg .bmp .tif .tiff).