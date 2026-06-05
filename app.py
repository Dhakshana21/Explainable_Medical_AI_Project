import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from explain import get_gradcam

model = tf.keras.models.load_model("model/medical_model.h5")

dummy = np.zeros((1,224,224,3))
model(dummy)

st.title("Explainable Medical AI Analysis")

st.write("Upload MRI / X-Ray image")

file = st.file_uploader("Upload Image")

if file is not None:

    image = Image.open(file)
    st.image(image, caption="Uploaded Image")

    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array,axis=0)

    # ✅ FIRST: get prediction
    prediction = model.predict(img_array)[0][0]

    # ✅ THEN: use it
    if prediction > 0.5:
        result = "Tumor Detected"
    else:
        result = "Normal"

    st.subheader("Prediction: " + result)

    # 📋 AI Analysis Report
    st.write("## 📋 AI Analysis Report")

    if result == "Tumor Detected":
        st.write("""
        - Tumor detected in the brain region.
        - The highlighted regions indicate abnormal tissue patterns.
        - The model identified irregular intensity compared to normal scans.
        - These patterns may correspond to possible tumor-affected areas.
        """)
    else:
        st.write("""
        - No tumor detected in the scan.
        - The brain structure appears normal based on model analysis.
        """)

    st.warning("This is an AI-based analysis and not a medical analysis.")

    # ✅ GradCAM AFTER prediction
    heatmap = get_gradcam(model, img_array, layer_name="conv2d_2")

    heatmap = cv2.resize(heatmap,(224,224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = heatmap * 0.4 + np.array(img)

    st.image(superimposed.astype('uint8'),caption="Explainable Heatmap")