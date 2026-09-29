import os

import streamlit as st

st.set_page_config(page_title="Rice Disease Detection", page_icon="🌾", layout="wide")

import re
import json
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import shap
from tensorflow.keras.applications.efficientnet import preprocess_input

# Model files to try, in this order (first one that loads is used)
CANDIDATE_MODELS = [
    'model/efficientnetb0_rice_fixed.keras',
    'model/efficientnetb0_rice_v2.h5',
    'model/efficientnetb0_rice.h5',
]

_KNOWN_BAD_KEYS = {'sparse', 'ragged', 'optional', 'quantization_config', 'synchronized'}


def _deep_fix_config(obj, bad_keys):
    if isinstance(obj, dict):
        if isinstance(obj.get('dtype'), dict):
            obj['dtype'] = obj['dtype'].get('config', {}).get('name', 'float32')
        if 'batch_shape' in obj:
            obj['batch_input_shape'] = obj.pop('batch_shape')
        for bad_key in bad_keys:
            obj.pop(bad_key, None)
        for k in list(obj.keys()):
            obj[k] = _deep_fix_config(obj[k], bad_keys)
        return obj
    elif isinstance(obj, list):
        return [_deep_fix_config(x, bad_keys) for x in obj]
    return obj


def _load_with_keras3(path):
    import keras as k3
    m = k3.models.load_model(path, compile=False, safe_mode=False)
    return m, k3


def _load_with_tf_keras_h5(path):
    import h5py
    import tf_keras as tfk
    with h5py.File(path, 'r') as f:
        cfg = f.attrs.get('model_config')
        if isinstance(cfg, bytes):
            cfg = cfg.decode('utf-8')
    raw = json.loads(cfg)
    bad_keys = set(_KNOWN_BAD_KEYS)
    for _ in range(15):
        try:
            c = _deep_fix_config(json.loads(json.dumps(raw)), bad_keys)
            m = tfk.models.model_from_json(json.dumps(c))
            m.load_weights(path)
            return m, tfk
        except TypeError as e:
            match = re.search(r"Keyword argument not understood:', '([^']+)'", str(e))
            if match and match.group(1) not in bad_keys:
                bad_keys.add(match.group(1))
                continue
            raise
    raise RuntimeError("tf_keras fallback exhausted")


@st.cache_resource
def load_model():
    errors = []
    for path in CANDIDATE_MODELS:
        if not os.path.exists(path):
            errors.append(f"{path}: file not found")
            continue
        try:
            m, kmod = _load_with_keras3(path)
            return m, kmod, f"{path} (Keras 3)"
        except Exception as e:
            errors.append(f"{path} [Keras 3]: {str(e)[:200]}")
        if path.endswith('.h5'):
            try:
                m, kmod = _load_with_tf_keras_h5(path)
                return m, kmod, f"{path} (tf_keras)"
            except Exception as e:
                errors.append(f"{path} [tf_keras]: {str(e)[:200]}")
    raise RuntimeError("Could not load any model:\n" + "\n".join(errors))


model, KMOD, LOADED_FROM = load_model()

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


def _first(x):
    """Keras 3 sometimes wraps a single tensor in a list. Unwrap it."""
    while isinstance(x, (list, tuple)):
        x = x[0]
    return x


def apply_gradcam(img_input):
    try:
        conv_layer_output = _first(model.get_layer('top_activation').output)
        model_output = _first(model.outputs)

        grad_model = KMOD.models.Model(
            inputs=model.inputs,
            outputs=[conv_layer_output, model_output]
        )

        img_tensor = tf.convert_to_tensor(img_input)
        call_arg = [img_tensor] if len(model.inputs) == 1 and isinstance(model.input, (list, tuple)) else img_tensor

        with tf.GradientTape() as tape:
            outs = grad_model(call_arg)
            conv_outputs = _first(outs[0])
            predictions = _first(outs[1])
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        img_display = img_input[0].copy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
        img_display = np.uint8(255 * img_display)

        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        return cv2.addWeighted(img_display, 0.6, heatmap_colored, 0.4, 0)
    except Exception as e:
        st.warning(f"Grad-CAM error: {e}")
        return None


def get_shap_explanation(img_input, predictions, predicted_class, nsamples=100):
    background = np.zeros((1, 224, 224, 3), dtype='float32')
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(img_input, nsamples=nsamples)

    if isinstance(shap_values, list):
        idx = predicted_class if len(shap_values) > predicted_class else 0
        shap_val = np.array(shap_values[idx])[0]
    else:
        shap_arr = np.array(shap_values)
        if shap_arr.ndim == 5:
            shap_val = shap_arr[0, :, :, :, predicted_class]
        elif shap_arr.ndim == 4:
            shap_val = shap_arr[0]
        else:
            shap_val = shap_arr

    shap_sum = np.sum(shap_val, axis=-1)
    shap_abs = np.abs(shap_sum)
    shap_norm = shap_abs / shap_abs.max() if shap_abs.max() > 0 else shap_abs

    img_display = img_input[0].copy()
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_display)
    axes[0].set_title('Original Image', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    im = axes[1].imshow(shap_norm, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('SHAP Feature Importance', fontsize=11, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_display)
    axes[2].imshow(shap_norm, cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title('SHAP Overlay on Leaf', fontsize=11, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle(
        f'SHAP — Predicted: {CLASS_NAMES[predicted_class]} ({predictions[0][predicted_class]*100:.1f}%)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return Image.open(buf)


def get_chart(probs, predicted_class):
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['red' if i == predicted_class else 'steelblue' for i in range(len(CLASS_NAMES))]
    y_pos = range(len(CLASS_NAMES))
    ax.barh(y_pos, probs * 100, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel('Confidence (%)', fontsize=9)
    ax.set_title('Disease Probability Analysis', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 110)
    for i, prob in enumerate(probs):
        ax.text(prob * 100 + 1, i, f'{prob*100:.1f}%', va='center', fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return Image.open(buf)


st.title("🌾 Rice Crop Health Monitoring System")
st.subheader("AI-Powered Disease Detection with Explainable AI (Grad-CAM + SHAP)")
st.caption(f"Model loaded from: {LOADED_FROM}")
st.markdown("---")

st.write("### 📷 Upload Rice Leaf Image")
uploaded_file = st.file_uploader("Choose a rice leaf photo...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    top_col1, top_col2 = st.columns([1, 1])

    with top_col1:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)

    img_preprocessed = preprocess_input(img_array.astype('float32'))
    img_input = np.expand_dims(img_preprocessed, axis=0)

    with st.spinner("🔍 Analyzing rice leaf..."):
        predictions = np.array(model.predict(img_input, verbose=0))
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
            st.image(
                gradcam_result,
                caption="🔴 Red = strongest focus | 🔵 Blue = least focus (a gradient, not an exact outline)",
                use_container_width=True
            )
    else:
        st.warning("Grad-CAM not available")

    st.markdown("---")
    st.write("### 🔬 SHAP — Feature Importance")
    with st.spinner("Calculating SHAP values... (1-2 minutes)"):
        try:
            shap_img = get_shap_explanation(img_input, predictions, predicted_class)
            sh1, sh2, sh3 = st.columns([1, 3, 1])
            with sh2:
                st.image(shap_img, use_container_width=True)
        except Exception as e:
            st.error(f"SHAP error: {e}")

    st.markdown("---")
    st.write("### 📊 All Disease Probabilities")
    chart = get_chart(predictions[0], predicted_class)
    ch1, ch2, ch3 = st.columns([1, 2, 1])
    with ch2:
        st.image(chart, use_container_width=True)