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
# Ensure these files are in your /content/V1-5/ directory
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

CLASS_COLORS = {
    1: [255, 0, 0], 2: [0, 0, 255], 3: [0, 255, 0], 4: [255, 255, 0],
    5: [0, 255, 255], 6: [255, 0, 255], 7: [128, 128, 128]
}

# --- 2. THE DUAL-EXPLANATION DICTIONARY ---
XAI_INFO = {
    "Gradient-weighted Class Activation Mapping": {
        "simple": "A 'heat map' showing where the AI prioritized its attention. Bright spots indicate high importance.",
        "technical": "Grad-CAM: Computes the gradients of the class score with respect to the feature maps of the final convolutional layer."
    },
    "Integrated Gradients": {
        "simple": "An 'evidence map' highlighting the exact textures and edges that influenced the AI's final decision.",
        "technical": "Attribution via Path Integral: Computes the average gradient of the output along a linear path from a baseline to the input."
    },
    "Intrinsic Attention Map Analysis": {
        "simple": "The AI's 'internal focus,' showing which global areas the model deemed relevant before looking at details.",
        "technical": "Extracts self-attention coefficients from the U-Net skip-connections, showing global spatial dependencies."
    },
    "Filter Activation Visualization": {
        "simple": "A look at the AI's 'neurons'—it shows how the model breaks the MRI down into basic shapes like lines and curves.",
        "technical": "Intermediate Feature Visualization: Displays the output of the first Conv2D layer (8 filters) to visualize feature extraction."
    },
    "Uncertainty Estimation via Monte Carlo Dropout": {
        "simple": "A 'Doubt Map.' Brighter areas mean the AI is unsure; these spots should be manually reviewed by a doctor.",
        "technical": "Predictive Entropy ($H$): Quantifies the variance across 25 forward passes to estimate model uncertainty."
    }
}

# --- 3. XAI CORE FUNCTIONS ---
@functools.lru_cache(maxsize=1)
def load_models():
    m_attn = create_model(num_classes=NUM_CLASSES, return_attention_map=True)
    m_attn.load_weights(TRAINED_MODEL_PATH)
    m_grad = create_model(num_classes=NUM_CLASSES, return_attention_map=False)
    m_grad.load_weights(TRAINED_MODEL_PATH)
    return m_attn, m_grad

def get_grad_cam(model, img_batch, class_idx):
    # 1. Define the Grad-CAM Model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer('gradcam_target_conv').output, model.get_layer('logits_output').output]
    )
    
    # 2. Record gradients
    with tf.GradientTape() as tape:
        conv_outputs, logits = grad_model(img_batch)
        loss = tf.reduce_sum(logits[:, :, :, class_idx])
    
    # 3. Compute importance weights
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 4. Generate the weighted heatmap
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(tf.maximum(heatmap, 0)) # ReLU logic
    
    # 5. FIX: Upsample and Normalize
    heatmap = cv2.resize(heatmap.numpy(), (256, 256))
    heatmap /= (np.max(heatmap) + 1e-10)
    
    # 6. FIX: Masking (The "Clinical Clean-up")
    # We use the model's prediction to zero out heatmap noise in the background
    pred_probs = model.predict(img_batch)[0]
    pred_mask = (np.argmax(pred_probs, axis=-1) == class_idx).astype(np.float32)
    
    # Clean the heatmap by multiplying it with the prediction mask
    cleaned_heatmap = heatmap * pred_mask
    
    return cleaned_heatmap
    
def get_integrated_gradients(model, img_batch, class_idx, steps=50):
    logits_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.get_layer('logits_output').output)
    baseline = tf.zeros_like(img_batch)
    alphas = tf.linspace(0.0, 1.0, steps)[:, tf.newaxis, tf.newaxis, tf.newaxis]
    interpolated = baseline + alphas * (img_batch - baseline)
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        logits = logits_model(interpolated)[:, :, :, class_idx]
    grads = tape.gradient(logits, interpolated)
    avg_grads = tf.reduce_mean(grads, axis=0)
    ig = (img_batch[0] - baseline[0]) * avg_grads
    ig = tf.reduce_sum(tf.abs(ig), axis=-1).numpy()
    return (ig - ig.min()) / (ig.max() - ig.min() + 1e-10)

def get_filter_viz(model, img_batch):
    act_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.get_layer('conv2d').output)
    activations = act_model.predict(img_batch)[0]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(8):
        ax = axes[i//4, i%4]
        ax.imshow(activations[:, :, i], cmap='magma')
        ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close()
    return Image.open(buf)

# --- 4. MAIN INTERFACE WRAPPER ---
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

    preds, attns = [], []
    for _ in range(MC_SAMPLES):
        p, _, a = model_attn(img_batch, training=True)
        preds.append(p.numpy())
        attns.append(a.numpy())
    
    mean_pred = np.mean(preds, axis=0).squeeze()
    pred_mask = np.argmax(mean_pred, axis=-1)
    
    mri_display = (processed_img.squeeze() * 255).astype(np.uint8)
    mri_rgb = cv2.cvtColor(mri_display, cv2.COLOR_GRAY2RGB)
    
    color = CLASS_COLORS.get(class_idx, [0, 255, 0])
    colored_mask = np.zeros_like(mri_rgb)
    colored_mask[pred_mask == class_idx] = color
    pred_overlay = cv2.addWeighted(mri_rgb, 1.0, colored_mask, alpha_val, 0)
    
    if xai_method == "Gradient-weighted Class Activation Mapping":
        hm = get_grad_cam(model_grad, img_batch, class_idx)
        xai_res = cv2.addWeighted(mri_rgb, 0.6, (plt.get_cmap('hot')(hm)[:,:,:3]*255).astype(np.uint8), 0.4, 0)
    elif xai_method == "Integrated Gradients":
        hm = get_integrated_gradients(model_grad, img_batch, class_idx)
        xai_res = cv2.addWeighted(mri_rgb, 0.6, (plt.get_cmap('plasma')(hm)[:,:,:3]*255).astype(np.uint8), 0.4, 0)
    elif xai_method == "Uncertainty Estimation via Monte Carlo Dropout":
        unc = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=-1)
        hm = (unc - unc.min()) / (unc.max() - unc.min() + 1e-10)
        xai_res = cv2.addWeighted(mri_rgb, 0.5, (plt.get_cmap('magma')(hm)[:,:,:3]*255).astype(np.uint8), 0.5, 0)
    elif xai_method == "Filter Activation Visualization":
        xai_res = get_filter_viz(model_grad, img_batch)
    else: 
        attn_avg = np.mean(attns, axis=0).squeeze()
        hm = (attn_avg - attn_avg.min()) / (attn_avg.max() - attn_avg.min() + 1e-10)
        xai_res = cv2.addWeighted(mri_rgb, 0.6, (plt.get_cmap('viridis')(hm)[:,:,:3]*255).astype(np.uint8), 0.4, 0)

    dice = (2. * np.sum((pred_mask==class_idx)*(processed_gt==class_idx))) / (np.sum(pred_mask==class_idx)+np.sum(processed_gt==class_idx)+1e-10)
    status_msg = f"### Structure Quality (Dice): {dice:.2f}\n" + ("✅ Validated" if dice > 0.7 else "🚨 Review Needed")
    report = f"### 💡 Interpretation\n\n**Patient-Friendly:** {XAI_INFO[xai_method]['simple']}\n\n**Expert Reference:** {XAI_INFO[xai_method]['technical']}"

    return mri_display, pred_overlay, xai_res, status_msg, report

# --- 5. DATA SETUP & UI ---
all_files, _ = cd.create_dataset(PATH_INPUT_VOLUMES, PATH_LABEL_VOLUMES, n=-1, s=0)
volume_names_map = {os.path.basename(p[0]): p for p in all_files}

# CORRECTED: Theme inside gr.Blocks, variables defined before click
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Fetal Brain Clinical Intelligence Portal")
    with gr.Sidebar():
        gr.Markdown("### ⚙️ View Settings")
        vol_in = gr.Dropdown(list(volume_names_map.keys()), label="Patient Volume")
        slc_in = gr.Slider(0, 100, step=1, label="Slice Number")
        cls_in = gr.Dropdown(list(CLASS_NAMES.values())[1:], label="Brain Structure", value="Cerebellum (CB)")
        xai_in = gr.Radio(list(XAI_INFO.keys()), label="Interpretability Mode", value="Gradient-weighted Class Activation Mapping")
        alp_in = gr.Slider(0.1, 0.9, value=0.5, label="Overlay Transparency")
        btn = gr.Button("Analyze Slice", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("🏥 Clinical Review"):
                    mri_out = gr.Image(label="Raw MRI")
                    pred_out = gr.Image(label="AI Segmentation")
                with gr.TabItem("🔍 AI Focus (XAI)"):
                    xai_out = gr.Image(label="Explainability Map")
        with gr.Column(scale=1):
            gr.Markdown("### 📊 AI Quality & Interpretation")
            status_out = gr.Markdown()
            report_out = gr.Markdown()

    btn.click(
        run_clinical_suite, 
        inputs=[vol_in, slc_in, cls_in, xai_in, alp_in], 
        outputs=[mri_out, pred_out, xai_out, status_out, report_out]
    )

# CORRECTED: No theme in launch
demo.launch(share=True, debug=True)
