import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import os
import functools
import io
from PIL import Image

# --- External Imports ---
from ensem_4_mod_4_no_mod import create_model
import create_dataset as cd
from create_dataset import preprocess_slice

# --- 1. CONFIGURATION & CLINICAL DICTIONARY ---
TRAINED_MODEL_PATH = '/content/drive/MyDrive/fetal-brain-segmentation-v1.5/checkpoints/Model.keras'
PATH_INPUT_VOLUMES = '/content/drive/MyDrive/feta_2.1/nii_files_input'
PATH_LABEL_VOLUMES = '/content/drive/MyDrive/feta_2.1/nii_files_output'
NUM_CLASSES = 8
MC_SAMPLES = 25 

CLASS_NAMES = {
    0: "Background", 1: "Cerebellum (CB)", 2: "White Matter (WM)", 
    3: "Grey Matter (GM)", 4: "Lateral Ventricles (LV)", 
    5: "External Cerebrospinal Fluid (eCSF)", 6: "Deep Grey Matter (DGM)", 7: "Brainstem (BS)"
}

CLASS_COLORS = {
    1: [255, 0, 0], 2: [0, 0, 255], 3: [0, 255, 0], 4: [255, 255, 0],
    5: [0, 255, 255], 6: [255, 0, 255], 7: [128, 128, 128]
}

XAI_INFO = {
    "Gradient-weighted Class Activation Mapping": {
        "simple": "A 'heat map' showing the general area the AI prioritized.",
        "technical": "Grad-CAM: Gradients of the class score w.r.t. the final conv layer."
    },
    "Integrated Gradients": {
        "simple": "An 'evidence map' highlighting sharp edges and textures used as proof.",
        "technical": "Path Integral: Average gradient along a path from baseline to input."
    },
    "Intrinsic Attention Map Analysis": {
        "simple": "The AI's 'internal focus' from the model's own Attention Gates.",
        "technical": "Skip-connection attention coefficients showing spatial dependencies."
    },
    "Filter Activation Visualization": {
        "simple": "Shows how the AI breaks the MRI down into basic shapes and lines.",
        "technical": "Displays the first Conv2D layer output (8 filters)."
    },
    "Uncertainty Estimation via Monte Carlo Dropout": {
        "simple": "The 'Doubt Map.' Brighter areas mean the AI is unsure.",
        "technical": "Predictive Entropy: Variance across stochastic forward passes."
    }
}

# --- 2. CORE ENGINE FUNCTIONS ---

@functools.lru_cache(maxsize=1)
def load_models():
    m_attn = create_model(num_classes=NUM_CLASSES, return_attention_map=True)
    m_attn.load_weights(TRAINED_MODEL_PATH)
    m_grad = create_model(num_classes=NUM_CLASSES, return_attention_map=False)
    m_grad.load_weights(TRAINED_MODEL_PATH)
    return m_attn, m_grad

def overlay_heatmap(background_rgb, heatmap, cmap_name='hot', alpha=0.4):
    """Safely normalizes, colorizes, and resizes heatmaps for blending."""
    # 1. Clean data and normalize
    heatmap = np.nan_to_num(heatmap.squeeze())
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
    
    # 2. Resize heatmap to match background dimensions (H, W)
    heatmap_resized = cv2.resize(heatmap, (background_rgb.shape[1], background_rgb.shape[0]))
    
    # 3. Apply colormap
    colored_hm = plt.get_cmap(cmap_name)(heatmap_resized)[:, :, :3]
    colored_hm = (colored_hm * 255).astype(np.uint8)
    
    # 4. Blend
    return cv2.addWeighted(background_rgb, 1 - alpha, colored_hm, alpha, 0)

def get_grad_cam(model, img_batch, class_idx):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs], 
        outputs=[model.get_layer('gradcam_target_conv').output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        # Handle if model returns list
        if isinstance(predictions, list): predictions = predictions[0]
        loss = predictions[:, :, :, class_idx]
        
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = (conv_outputs[0] @ pooled_grads[..., tf.newaxis]).numpy()
    return np.maximum(heatmap, 0)

def get_integrated_gradients(model, img_batch, class_idx, m_steps=20):
    image = tf.cast(img_batch[0], tf.float32)
    baseline = tf.zeros_like(image)
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)
    interpolated = baseline + alphas[:, tf.newaxis, tf.newaxis, tf.newaxis] * (image - baseline)
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        logits = model(interpolated)
        if isinstance(logits, list): logits = logits[0]
        loss = logits[..., class_idx]
    
    grads = tape.gradient(loss, interpolated)
    avg_grads = tf.reduce_mean((grads[:-1] + grads[1:]) / 2.0, axis=0)
    ig = (image - baseline) * avg_grads
    return np.abs(ig).sum(axis=-1)

def get_filter_viz(model, img_batch):
    first_conv = next(l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D))
    vis_model = tf.keras.models.Model(inputs=model.inputs, outputs=first_conv.output)
    act = vis_model.predict(img_batch, verbose=0)[0]
    num_viz = min(8, act.shape[-1])
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < num_viz:
            ax.imshow(act[:, :, i], cmap='magma')
        ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return Image.open(buf)

# --- 3. MAIN INTERFACE WRAPPER ---

def run_clinical_suite(volume_name, slice_idx, target_class, xai_method, alpha_val):
    model_attn, model_grad = load_models()
    paths = volume_names_map[volume_name]
    
    img_vol = nib.load(paths[0]).get_fdata()
    lbl_vol = nib.load(paths[1]).get_fdata() if paths[1] else None
    orig_slice = img_vol[slice_idx, :, :]
    orig_lbl = lbl_vol[slice_idx, :, :].astype(np.int32) if lbl_vol is not None else np.zeros_like(orig_slice)
    
    processed_img, processed_gt = preprocess_slice(orig_slice, orig_lbl)
    img_batch = tf.expand_dims(tf.constant(processed_img, dtype=tf.float32), axis=0)
    class_idx = [k for k, v in CLASS_NAMES.items() if v == target_class][0]

    # Inference with MC Dropout
    preds, attns = [], []
    for _ in range(MC_SAMPLES):
        out = model_attn(img_batch, training=True)
        preds.append(out[0].numpy()) 
        attns.append(out[2].numpy())
    
    mean_pred = np.mean(preds, axis=0).squeeze()
    pred_mask = np.argmax(mean_pred, axis=-1)
    mri_display = (processed_img.squeeze() * 255).astype(np.uint8)
    mri_rgb = cv2.cvtColor(mri_display, cv2.COLOR_GRAY2RGB)
    
    def apply_overlay(bg, mask, color, alpha):
        overlay = bg.copy()
        if mask.sum() > 0:
            color_arr = np.tile(np.array(color, dtype=np.uint8), (mask.sum(), 1))
            overlay[mask] = cv2.addWeighted(overlay[mask], 1 - alpha, color_arr, alpha, 0).squeeze()
        return overlay

    gt_overlay = apply_overlay(mri_rgb, (processed_gt == class_idx), [255, 165, 0], alpha_val)
    pred_overlay = apply_overlay(mri_rgb, (pred_mask == class_idx), CLASS_COLORS.get(class_idx, [0, 255, 0]), alpha_val)

    # XAI Logic
    if xai_method == "Filter Activation Visualization":
        xai_res = get_filter_viz(model_grad, img_batch)
    elif xai_method == "Gradient-weighted Class Activation Mapping":
        hm = get_grad_cam(model_grad, img_batch, class_idx)
        xai_res = overlay_heatmap(mri_rgb, hm, 'hot')
    elif xai_method == "Integrated Gradients":
        hm = get_integrated_gradients(model_grad, img_batch, class_idx)
        xai_res = overlay_heatmap(mri_rgb, hm, 'hot')
    elif xai_method == "Uncertainty Estimation via Monte Carlo Dropout":
        unc = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=-1)
        xai_res = overlay_heatmap(mri_rgb, unc, 'magma', alpha=0.5)
    else: # Intrinsic Attention
        attn_map = np.mean(attns, axis=0).squeeze()
        if attn_map.ndim > 2: attn_map = np.mean(attn_map, axis=-1)
        xai_res = overlay_heatmap(mri_rgb, attn_map, 'viridis')

    dice = (2. * np.sum((pred_mask==class_idx)*(processed_gt==class_idx))) / (np.sum(pred_mask==class_idx)+np.sum(processed_gt==class_idx)+1e-10)
    status = f"### Structure Dice: {dice:.2f}\n" + ("✅ Validated" if dice > 0.65 else "🚨 Manual Review Required")
    
    info = XAI_INFO[xai_method]
    report = f"### 💡 AI Interpretation\n\n**Simple:** {info['simple']}\n\n**Technical:** {info['technical']}"
    
    return mri_display, gt_overlay, pred_overlay, xai_res, status, report

# --- 4. DATA SETUP & UI ---
all_files = cd.create_dataset(PATH_INPUT_VOLUMES, PATH_LABEL_VOLUMES, -1, 0)[0]
volume_names_map = {os.path.basename(p[0]): p for p in all_files}

with gr.Blocks(title="Fetal Brain Clinical Station") as demo:
    gr.Markdown("# 🧠 Fetal Brain Clinical Station V2.9")
    with gr.Sidebar():
        vol_in = gr.Dropdown(list(volume_names_map.keys()), label="Patient Volume")
        slc_in = gr.Slider(0, 100, step=1, label="Slice Index")
        cls_in = gr.Dropdown(list(CLASS_NAMES.values())[1:], label="Structure", value="Cerebellum (CB)")
        xai_in = gr.Radio(list(XAI_INFO.keys()), label="XAI Suite", value="Gradient-weighted Class Activation Mapping")
        alp_in = gr.Slider(0.1, 1.0, value=0.6, label="Overlay Opacity")
        btn = gr.Button("Analyze Slice", variant="primary")

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("🏥 Radiology View"):
                    with gr.Row():
                        mri_out = gr.Image(label="1. Original MRI")
                        gt_out = gr.Image(label="2. Expert GT (Orange)")
                        pred_out = gr.Image(label="3. AI Prediction")
                with gr.TabItem("🔍 Explainability (XAI)"):
                    xai_out = gr.Image(label="XAI Visualization")
        with gr.Column(scale=1):
            status_out = gr.Markdown(); report_out = gr.Markdown()

    btn.click(run_clinical_suite, [vol_in, slc_in, cls_in, xai_in, alp_in], [mri_out, gt_out, pred_out, xai_out, status_out, report_out])

demo.launch(theme=gr.themes.Soft(), share=True)
