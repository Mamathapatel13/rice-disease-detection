import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st

# MUST be the first Streamlit command in the whole script
st.set_page_config(page_title="Rice Disease Detection", page_icon="🌾", layout="wide")

import numpy as np
import tf_keras as keras
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import sys
sys.path.insert(0, 'utils')
from shap_explain import get_shap_explanation

# --- PATCH: fix InputLayer deserialization mismatches across Keras versions ---
_InputLayer = keras.layers.InputLayer
_orig_from_config = _InputLayer.from_config.__func__

def _patched_from_config(cls, config):
    config = dict(config)
    if 'batch_shape' in config:
        config['batch_input_shape'] = config.pop('batch_shape')
    for bad_key in ('sparse', 'ragged', 'optional'):
        config.pop(bad_key, None)
    return _orig_from_config(cls, config)

_InputLayer.from_config = classmethod(_patched_from_config)
# --- END PATCH ---

@st.cache_resource
def load_model():
    return keras.models.load_model('model/efficientnetb0_rice.h5')

model = load_model()

CLASS_NAMES = [
    'Bacterial Leaf Blight',
    'Brown Spot',
    'Healthy Rice Leaf',
    'Leaf Blast',
    'Leaf Scald',
    'Sheath Blight'
]

DISEASE_INFO = {
    'Bacterial Leaf Blight': 'Caused by bacteria Xanthomonas oryzae. Leaves turn yellow and dry.',
    'Brown Spot': 'Caused by fungus Cochliobolus miyabeanus. Brown oval spots appear on leaves.',
    'Healthy Rice Leaf': 'Your rice plant is healthy! No disease detected.',
    'Leaf Blast': 'Caused by fungus Magnaporthe oryzae. Diamond shaped lesions on leaves.',
    'Leaf Scald': 'Caused by fungus Microdochium oryzae. Scalded appearance on leaf tips.',
    'Sheath Blight': 'Caused by fungus Rhizoctonia solani. Oval lesions on leaf sheath.'
}

TREATMENT = {
    'Bacterial Leaf Blight': 'Apply copper-based bactericides. Drain fields and reduce nitrogen.',
    'Brown Spot': 'Apply fungicides like Mancozeb or Propiconazole. Improve soil nutrition.',
    'Healthy Rice Leaf': 'No treatment needed. Continue regular monitoring.',
    'Leaf Blast': 'Apply Tricyclazole or Isoprothiolane fungicide immediately.',
    'Leaf Scald': 'Apply fungicides. Avoid excess nitrogen. Use resistant varieties.',
    'Sheath Blight': 'Apply fungicides like Hexaconazole. Reduce plant density.'
}


def apply_gradcam(img_input):
    """
    Fixed order: resize the tiny (7x7) heatmap to full size FIRST while it's
    still smooth, THEN blur, THEN threshold. Thresholding while still tiny is
    what caused the blocky rainbow artifacts you saw.
    """
    try:
        grad_model = tf.keras.models.Model(
            model.inputs,
            [model.get_layer('top_activation').output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_input)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0, 1)

        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)
        heatmap_resized = np.clip(heatmap_resized, 0, 1)

        heatmap_resized[heatmap_resized < 0.4] = 0

        img_display = img_input[0].copy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
        img_display = np.uint8(255 * img_display)

        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        mask = (heatmap_resized > 0).astype(np.uint8)[..., None]
        overlay = cv2.addWeighted(img_display, 0.5, heatmap_colored, 0.5, 0)
        blended = np.where(mask, overlay, img_display)
        return blended
    except Exception:
        return None


def get_chart(probs, predicted_class):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['red' if i == predicted_class else 'steelblue' for i in range(len(CLASS_NAMES))]
    y_pos = range(len(CLASS_NAMES))
    ax.barh(y_pos, probs * 100, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Disease Probability Analysis', fontweight='bold')
    ax.set_xlim(0, 110)
    for i, prob in enumerate(probs):
        ax.text(prob * 100 + 1, i, f'{prob*100:.1f}%', va='center', fontsize=9)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return Image.open(buf)


st.title("🌾 Rice Crop Health Monitoring System")
st.subheader("AI-Powered Disease Detection with Explainable AI (Grad-CAM + SHAP)")
st.markdown("---")

st.write("### 📷 Upload Rice Leaf Image")
uploaded_file = st.file_uploader(
    "Choose a rice leaf photo...",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    top_col1, top_col2 = st.columns([1, 1])

    with top_col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_preprocessed = preprocess_input(img_array.astype('float32'))
    img_input = np.expand_dims(img_preprocessed, axis=0)

    with st.spinner("🔍 Analyzing rice leaf..."):
        predictions = model.predict(img_input)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class]) * 100

    with top_col2:
        st.write("### 🔍 Detection Result")
        if CLASS_NAMES[predicted_class] == 'Healthy Rice Leaf':
            st.success(f"✅ {CLASS_NAMES[predicted_class]}")
        else:
            st.error(f"⚠️ {CLASS_NAMES[predicted_class]}")
        st.metric("Confidence", f"{confidence:.2f}%")
        st.info(f"ℹ️ {DISEASE_INFO[CLASS_NAMES[predicted_class]]}")
        st.warning(f"💊 Treatment: {TREATMENT[CLASS_NAMES[predicted_class]]}")

    st.markdown("---")

    st.write("### 🔥 Grad-CAM — Disease Location")
    gradcam_result = apply_gradcam(img_input)
    if gradcam_result is not None:
        gc1, gc2, gc3 = st.columns([1, 2, 1])
        with gc2:
            st.image(gradcam_result,
                      caption="🔴 Red = Disease Area | 🔵 Blue = Healthy Area",
                      use_container_width=True)
    else:
        st.warning("Grad-CAM not available")

    st.markdown("---")

    st.write("### 🔬 SHAP — Feature Importance")
    with st.spinner("Calculating SHAP values... (1-2 minutes)"):
        try:
            shap_img = get_shap_explanation(model, img_input, CLASS_NAMES)
            st.image(shap_img,
                      caption="SHAP: Brighter = More important features for prediction",
                      use_container_width=True)
        except Exception as e:
            st.error(f"SHAP error: {e}")

    st.markdown("---")

    st.write("### 📊 All Disease Probabilities")
    chart = get_chart(predictions[0], predicted_class)
    ch1, ch2, ch3 = st.columns([1, 2, 1])
    with ch2:
        st.image(chart, use_container_width=True)