import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import shap

def get_shap_explanation(model, img_input, class_names):
    predictions = model.predict(img_input)
    predicted_class = int(np.argmax(predictions[0]))

    background = np.zeros((1, 224, 224, 3), dtype='float32')
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(img_input, nsamples=20)
    if isinstance(shap_values, list):
        shap_val = np.array(shap_values[predicted_class])[0]
    else:
        shap_arr = np.array(shap_values)
        if shap_arr.ndim == 5:  # shape: (batch, H, W, C, num_classes)
            shap_val = shap_arr[0, :, :, :, predicted_class]
        elif shap_arr.ndim == 4:  # shape: (batch, H, W, C)
            shap_val = shap_arr[0]
        else:
            shap_val = shap_arr

    shap_sum = np.sum(shap_val, axis=-1)
    shap_abs = np.abs(shap_sum)
    if shap_abs.max() > 0:
        shap_norm = shap_abs / shap_abs.max()
    else:
        shap_norm = shap_abs

    img_display = img_input[0].copy()
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_display)
    axes[0].set_title('Original Image', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    im = axes[1].imshow(shap_norm, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('SHAP Feature Importance', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_display)
    axes[2].imshow(shap_norm, cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title('SHAP Overlay on Leaf', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle(
        f'SHAP — Predicted: {class_names[predicted_class]} ({predictions[0][predicted_class]*100:.1f}%)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return Image.open(buf)