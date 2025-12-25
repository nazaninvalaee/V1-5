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
# --- 1. CONFIGURATION ---
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
# Anatomical Colors for AI overlay
CLASS_COLORS = {
    1: [255, 0, 0], 2: [0, 0, 255], 3: [0, 255, 0], 4: [255, 255, 0],
    5: [0, 255, 255], 6: [255, 0, 255], 7: [128, 128, 128]
}
XAI_INFO = {
    "Gradient-weighted Class Activation Mapping": "Heatmap of high-level feature importance.",
    "Integrated Gradients": "Pixel-level attribution showing evidence for the class.",
    "Intrinsic Attention Map Analysis": "U-Net skip-connection attention gate activations.",
    "Filter Activation Visualization": "Low-level features (edges/textures) from the first layer.",
    "Uncertainty Estimation (MC Dropout)": "Predictive entropy showing where the model is 'unsure'."
}
# --- 2. CORE FUNCTIONS ---
@functools.lru_cache(maxsize=1)
def load_models():
    m_attn = create_model(num_classes=NUM_CLASSES, return_attention_map=True)
    m_attn.load_weights(TRAINED_MODEL_PATH)
    m_grad = create_model(num_classes=NUM_CLASSES, return_attention_map=False)
    m_grad.load_weights(TRAINED_MODEL_PATH)
    return m_attn, m_grad
def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, 0)
    image_x = tf.expand_dims(image, 0)
    delta = image_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images
def integral_approximation(gradients):
    grads = (gradients[:-1] + gradients[1:]) / 2.0
    integrated_gradients = tf.reduce_mean(grads, axis=0)
    return integrated_gradients
def get_integrated_gradients(model, img_batch, class_idx, m_steps=20, batch_size=10):
    image = img_batch[0]
    baseline = tf.zeros_like(image)
    # Use the logits layer for consistency with Grad-CAM
    output_layer = model.get_layer('logits_output').output
    grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=output_layer)
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)
    gradient_batches = []
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]
        interpolated = interpolate_images(baseline, image, alpha_batch)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            logits = grad_model(interpolated)
            loss = tf.reduce_sum(logits[..., class_idx], axis=[1, 2])
        grads = tape.gradient(loss, interpolated)
        gradient_batches.append(grads)
    total_gradients = tf.concat(gradient_batches, axis=0)
    avg_gradients = integral_approximation(total_gradients)
    ig = (image - baseline) * avg_gradients
    return ig
def get_grad_cam(model, img_batch, class_idx):
    grad_model = tf.keras.models.Model(inputs=[model.inputs], outputs=[model.get_layer('gradcam_target_conv').output, model.get_layer('logits_output').output])
    with tf.GradientTape() as tape:
        conv_outputs, logits = grad_model(img_batch)
        loss = tf.reduce_sum(logits[:, :, :, class_idx])
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = (conv_outputs[0] @ pooled_grads[..., tf.newaxis]).numpy()
    return cv2.resize(np.maximum(heatmap, 0), (256, 256))
def get_filter_viz(model, img_batch):
    act_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.get_layer('conv2d').output)
    activations = act_model.predict(img_batch)[0]
    c = activations.shape[-1]
    num_viz = min(8, c)
    if num_viz == 0:
        return None  # or some placeholder
    num_rows = (num_viz + 3) // 4
    fig, axes = plt.subplots(num_rows, 4, figsize=(10, 2.5 * num_rows))
    axes_flat = axes.flatten() if num_rows > 1 else axes
    for i in range(num_viz):
        axes_flat[i].imshow(activations[:, :, i], cmap='magma')
        axes_flat[i].axis('off')
    for i in range(num_viz, num_rows * 4):
        axes_flat[i].axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
# --- 3. MAIN INTERFACE WRAPPER ---
def run_clinical_suite(volume_name, slice_idx, target_class, xai_method, alpha_val):
    model_attn, model_grad = load_models()
    paths = volume_names_map[volume_name]
   
    # Load MRI and Label
    img_vol = nib.load(paths[0]).get_fdata()
    lbl_vol = nib.load(paths[1]).get_fdata() if paths[1] else None
    orig_slice = img_vol[slice_idx, :, :]
    orig_lbl = lbl_vol[slice_idx, :, :].astype(np.int32) if lbl_vol is not None else np.zeros_like(orig_slice)
   
    # Preprocess (This is the logic you confirmed works)
    processed_img, processed_gt = preprocess_slice(orig_slice, orig_lbl)
    img_batch = tf.expand_dims(tf.constant(processed_img, dtype=tf.float32), axis=0)
    class_idx = [k for k, v in CLASS_NAMES.items() if v == target_class][0]
    # MC Dropout Inference
    preds, attns = [], []
    for _ in range(MC_SAMPLES):
        p, _, a = model_attn(img_batch, training=True)
        preds.append(p.numpy()); attns.append(a.numpy())
   
    mean_pred = np.mean(preds, axis=0).squeeze()
    pred_mask = np.argmax(mean_pred, axis=-1)
   
    # Create the 3-Column Radiology View images
    mri_display = (processed_img.squeeze() * 255).astype(np.uint8)
    mri_rgb = cv2.cvtColor(mri_display, cv2.COLOR_GRAY2RGB)
   
    # GT Overlay (Orange)
    gt_overlay = mri_rgb.copy()
    gt_mask = (processed_gt == class_idx)
    color = np.array([255, 165, 0], dtype=np.uint8)
    num_pixels = gt_mask.sum()
    if num_pixels > 0:
        color_array = np.tile(color, (num_pixels, 1))
        blended = cv2.addWeighted(gt_overlay[gt_mask], 1 - alpha_val, color_array, alpha_val, 0)
        gt_overlay[gt_mask] = blended
   
    # AI Overlay (Specific Structure Color)
    pred_overlay = mri_rgb.copy()
    ai_mask = (pred_mask == class_idx)
    struct_color = np.array(CLASS_COLORS.get(class_idx, [0, 255, 0]), dtype=np.uint8)
    num_pixels = ai_mask.sum()
    if num_pixels > 0:
        color_array = np.tile(struct_color, (num_pixels, 1))
        blended = cv2.addWeighted(pred_overlay[ai_mask], 1 - alpha_val, color_array, alpha_val, 0)
        pred_overlay[ai_mask] = blended
    # XAI Suite Logic
    if xai_method == "Gradient-weighted Class Activation Mapping":
        hm = get_grad_cam(model_grad, img_batch, class_idx)
        xai_res = cv2.addWeighted(mri_rgb, 0.6, (plt.get_cmap('hot')(hm/hm.max())[:,:,:3]*255).astype(np.uint8), 0.4, 0)
    elif xai_method == "Integrated Gradients":
        ig = get_integrated_gradients(model_grad, img_batch, class_idx)
        hm = tf.reduce_sum(tf.abs(ig), axis=-1).numpy()
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-10)
        xai_res = cv2.addWeighted(mri_rgb, 0.6, (plt.get_cmap('hot')(hm)[:,:,:3]*255).astype(np.uint8), 0.4, 0)
    elif xai_method == "Intrinsic Attention Map Analysis":
        attn_mean = np.mean(attns, axis=0).squeeze()
        if attn_mean.ndim > 2:
            attn_mean = np.mean(attn_mean, axis=-1)
        hm = cv2.resize(attn_mean, (256, 256))
        xai_res = cv2.addWeighted(mri_rgb, 0.6, (plt.get_cmap('viridis')(hm/hm.max())[:,:,:3]*255).astype(np.uint8), 0.4, 0)
    elif xai_method == "Filter Activation Visualization":
        xai_res = get_filter_viz(model_grad, img_batch)
    elif xai_method == "Uncertainty Estimation (MC Dropout)":
        unc = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=-1)
        hm = (unc - unc.min()) / (unc.max() - unc.min() + 1e-10)
        xai_res = cv2.addWeighted(mri_rgb, 0.5, (plt.get_cmap('magma')(hm)[:,:,:3]*255).astype(np.uint8), 0.5, 0)
    # Dice Metric
    dice = (2. * np.sum(ai_mask * gt_mask)) / (np.sum(ai_mask) + np.sum(gt_mask) + 1e-10)
    status_msg = f"### Structure Dice: {dice:.2f}\n" + ("✅ Validated" if dice > 0.65 else "🚨 Manual Review Required")
    report = f"### 💡 AI Interpretation\n**Method:** {xai_method}\n\n**Note:** {XAI_INFO[xai_method]}"
    return mri_display, gt_overlay, pred_overlay, xai_res, status_msg, report
# --- 4. DATA SETUP & UI ---
all_files = cd.create_dataset(PATH_INPUT_VOLUMES, PATH_LABEL_VOLUMES, n=-1, s=0)[0]
volume_names_map = {os.path.basename(p[0]): p for p in all_files}
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Fetal Brain Clinical Station V2.2")
    with gr.Sidebar():
        vol_in = gr.Dropdown(list(volume_names_map.keys()), label="Patient Volume")
        slc_in = gr.Slider(0, 100, step=1, label="Slice Index")
        cls_in = gr.Dropdown(list(CLASS_NAMES.values())[1:], label="Target Structure", value="Cerebellum (CB)")
        xai_in = gr.Radio(list(XAI_INFO.keys()), label="XAI Suite", value="Gradient-weighted Class Activation Mapping")
        alp_in = gr.Slider(0.1, 1.0, value=0.6, label="Overlay Transparency")
        btn = gr.Button("Analyze Slice", variant="primary")
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("🏥 Radiology View"):
                    with gr.Row():
                        mri_out = gr.Image(label="1. Original MRI")
                        gt_out = gr.Image(label="2. Expert GT (Orange)")
                        pred_out = gr.Image(label="3. AI Prediction")
                with gr.TabItem("🔍 AI Focus (XAI)"):
                    xai_out = gr.Image(label="Interpretability Map")
        with gr.Column(scale=1):
            status_out = gr.Markdown()
            report_out = gr.Markdown()
    btn.click(run_clinical_suite,
              [vol_in, slc_in, cls_in, xai_in, alp_in],
              [mri_out, gt_out, pred_out, xai_out, status_out, report_out])
demo.launch(theme=gr.themes.Soft(), share=True)
