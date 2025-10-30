# app.py
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from module1.enhancement import hist_equalize, mean_filter, median_filter, adjust_brightness_contrast, gamma_correction, frequency_filter, laplacian_sharpen
from module2.segmentation import otsu_threshold, canny_edge, color_kmeans_segmentation, morphological_ops
from module3.transform import rotate_image, scale_image, translate_image
from module5.classifier import predict

st.set_page_config(page_title="Handwritten Recognizer Toolbox", layout="wide")

st.title("Handwritten Recognizer Toolbox (Python)")

col1, col2 = st.columns([1,2])
with col1:
    uploaded = st.file_uploader("Upload Image", type=['png','jpg','jpeg','bmp'])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded", use_column_width=True)
    else:
        st.markdown("Upload an image to start.")

with col2:
    st.subheader("Choose Module")
    module = st.selectbox("Module", ["Enhancement","Segmentation","Transform","Recognition"])
    if uploaded:
        if module == "Enhancement":
            op = st.selectbox("Operation", ["Histogram Equalization", "Mean Filter", "Median Filter", "Laplacian Sharpen", "Gamma Correction", "Frequency Lowpass"])
            if st.button("Apply Enhancement"):
                if op == "Histogram Equalization":
                    out = hist_equalize(img)
                elif op == "Mean Filter":
                    out = mean_filter(img, k=5)
                elif op == "Median Filter":
                    out = median_filter(img, k=5)
                elif op == "Laplacian Sharpen":
                    out = laplacian_sharpen(img)
                elif op == "Gamma Correction":
                    out = gamma_correction(img, gamma=1.2)
                else:
                    out = frequency_filter(img, kind='low', cutoff=30)
                st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption=f"Result: {op}", use_column_width=True)
        elif module == "Segmentation":
            op = st.selectbox("Operation", ["Otsu Threshold", "Canny Edge", "Color KMeans", "Morphological Open"])
            if st.button("Apply Segmentation"):
                if op == "Otsu Threshold":
                    out = otsu_threshold(img)
                    st.image(out, caption=op, use_column_width=True)
                elif op == "Canny Edge":
                    out = canny_edge(img, 100, 200)
                    st.image(out, caption=op, use_column_width=True)
                elif op == "Color KMeans":
                    out = color_kmeans_segmentation(img, k=3)
                    st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption=op, use_column_width=True)
                else:
                    bw = otsu_threshold(img)
                    out = morphological_ops(bw, op='open', kernel_size=3)
                    st.image(out, caption=op, use_column_width=True)
        elif module == "Transform":
            op = st.selectbox("Operation", ["Rotate 30°", "Scale x1.5", "Translate (30,30)"])
            if st.button("Apply Transform"):
                if op.startswith("Rotate"):
                    out = rotate_image(img, 30)
                elif op.startswith("Scale"):
                    out = scale_image(img, sx=1.5, sy=1.5)
                else:
                    out = translate_image(img, 30, 30)
                st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption=op, use_column_width=True)
        elif module == "Recognition":
            st.write("Recognition uses pre-trained models in `models/` (run train_classifiers.py first).")
            model = st.selectbox("Model", ["mlp", "knn"])
            if st.button("Predict Digit"):
                try:
                    label = predict(img, model=model)
                    st.success(f"Predicted label: {label}")
                except Exception as e:
                    st.error(f"Error predicting: {e}. Make sure models exist (run training).")
