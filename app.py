import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import nibabel as nib
import os
import functools
import warnings

# Suppress minor Matplotlib/Gradio warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure your 'ensem_4_mod_4_no_mod' file is correctly set up.
from ensem_4_mod_4_no_mod import create_model
import create_dataset as cd
from create_dataset import preprocess_slice

# --- Configuration ---
TRAINED_MODEL_PATH = '/content/drive/MyDrive/fetal-brain-segmentation-v1.5/checkpoints/Model.keras'
PATH_INPUT_VOLUMES = '/content/drive/MyDrive/feta_2.1/nii_files_input'
PATH_LABEL_VOLUMES = '/content/drive/MyDrive/feta_2.1/nii_files_output'
NUM_CLASSES = 8

CLASSES_TO_ANALYZE = [1, 2, 3, 4, 5, 6, 7]
MIN_PIXELS_FOR_ANALYSIS = 1000
POOR_PERFORMANCE_JACCARD_THRESHOLD = 0.5
MC_SAMPLES = 25 # Reduced samples for faster Gradio feedback

# Class names mapping for the output explanation
CLASS_NAMES = {
    0: "Background",
    1: "Cerebellum",
    2: "Cerebral White Matter",
    3: "Cerebral Cortex",
    4: "Lateral Ventricles",
    5: "Extra-Axial CSF",
    6: "Corpus Callosum",
    7: "Brainstem"
}
CLASSES_TO_ANALYZE_MAP = {k: CLASS_NAMES[k] for k in CLASSES_TO_ANALYZE}
CLINICAL_XAI_METHODS = {
    "Attention Map": "Regions of Focus (Attention)",
    "Grad-CAM": "Regions of Influence",
    "Integrated Gradients": "Feature Sensitivity"
}


# --- Core Prediction Logic (Adapted from previous step) ---
def predict_with_mc_dropout_and_attention(model_path, input_image_batch, num_samples=MC_SAMPLES, num_classes=NUM_CLASSES):
    """
    Performs Monte Carlo Dropout predictions and extracts attention maps.
    Returns mean prediction, uncertainty map, and averaged attention map.
    """
    if not isinstance(input_image_batch, tf.Tensor):
        input_image_batch = tf.constant(input_image_batch, dtype=tf.float32)

    if len(input_image_batch.shape) == 3:
        input_image_batch = tf.expand_dims(input_image_batch, axis=0)

    # Load model instance with attention map output
    xai_model = create_model(num_classes=num_classes, return_attention_map=True)
    xai_model.load_weights(model_path)

    all_predictions = []
    all_attention_maps = []

    for _ in range(num_samples):
        # FIX: Capture all 3 outputs, ignoring the auxiliary logits_output
        pred_mask_batch, _, attention_map_batch = xai_model(input_image_batch, training=True) 

        all_predictions.append(pred_mask_batch.numpy())
        all_attention_maps.append(attention_map_batch.numpy())

    all_predictions = np.array(all_predictions).squeeze(axis=1) # Squeeze batch axis if size 1
    all_attention_maps = np.array(all_attention_maps).squeeze(axis=1) # Squeeze batch axis if size 1

    # --- Calculate Mean Prediction ---
    mean_prediction = np.mean(all_predictions, axis=0)

    # --- Calculate Uncertainty Map (Entropy) ---
    epsilon = 1e-10
    uncertainty_map = -np.sum(mean_prediction * np.log(mean_prediction + epsilon), axis=-1)

    # --- Process Attention Map ---
    # Squeeze the last channel axis if it exists and then average over samples
    averaged_attention_map = np.mean(all_attention_maps, axis=0) # Average over samples only
    if averaged_attention_map.ndim == 3 and averaged_attention_map.shape[-1] == 1:
        averaged_attention_map = averaged_attention_map.squeeze(axis=-1)


    return mean_prediction, uncertainty_map, averaged_attention_map


# --- Helper Functions (Metrics, Grad-CAM, Overlay, etc.) ---
def jaccard_index(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

# Grad-CAM function (same as provided)
def generate_grad_cam_for_class(model, input_image_batch, target_class_idx, layer_name='gradcam_target_conv', logits_layer_name='logits_output'):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.get_layer(logits_layer_name).output]
    )
    with tf.GradientTape() as tape:
        tape.watch(input_image_batch)
        last_conv_layer_output, logits = grad_model(input_image_batch)
        # Use logits output for Grad-CAM target
        target_class_logit_map = logits[:, :, :, target_class_idx]
        target_class_score = tf.reduce_sum(target_class_logit_map)
    grads = tape.gradient(target_class_score, last_conv_layer_output)
    if grads is None or not tf.reduce_any(tf.math.is_finite(grads)) or tf.reduce_all(tf.equal(grads, 0)):
        return np.zeros(input_image_batch.shape[1:3])
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.maximum(heatmap, 0)
    max_heatmap = tf.reduce_max(heatmap)
    if max_heatmap == 0:
        normalized_heatmap = heatmap
    else:
        normalized_heatmap = heatmap / max_heatmap
    normalized_heatmap = normalized_heatmap.numpy().squeeze()
    original_height, original_width = input_image_batch.shape[1:3]
    heatmap_resized = cv2.resize(normalized_heatmap, (original_height, original_width))
    return heatmap_resized

# Integrated Gradients function (same as provided)
def integrated_gradients(model, input_image_batch, target_class_idx, steps=50, logits_layer_name='logits_output'):
    logits_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.get_layer(logits_layer_name).output
    )
    baseline = tf.zeros_like(input_image_batch)
    alphas = tf.linspace(start=0.0, stop=1.0, num=steps)[:, tf.newaxis, tf.newaxis, tf.newaxis]
    interpolated_images = baseline + alphas * (input_image_batch - baseline)
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        interpolated_logits = logits_model(interpolated_images)[:, :, :, target_class_idx]
    path_gradients = tape.gradient(interpolated_logits, interpolated_images)
    if path_gradients is None:
        return np.zeros(input_image_batch.shape[1:3])
    avg_gradients = tf.reduce_mean(path_gradients, axis=0)
    integrated_gradients_map = (input_image_batch - baseline) * avg_gradients
    integrated_gradients_map = tf.reduce_sum(tf.abs(integrated_gradients_map), axis=-1)
    normalized_heatmap = integrated_gradients_map.numpy().squeeze()
    max_val = np.max(normalized_heatmap)
    min_val = np.min(normalized_heatmap)
    if (max_val - min_val) == 0:
        normalized_heatmap = np.zeros_like(normalized_heatmap)
    else:
        normalized_heatmap = (normalized_heatmap - min_val) / (max_val - min_val)
    original_height, original_width = input_image_batch.shape[1:3]
    heatmap_resized = cv2.resize(normalized_heatmap, (original_height, original_width))
    return heatmap_resized


def overlay_heatmap(original_image_2d, heatmap, cmap='hot', alpha=0.5):
    """Overlays a heatmap on a grayscale image."""
    original_image_2d = original_image_2d.astype(np.float32)
    original_image_2d = (original_image_2d - original_image_2d.min()) / (original_image_2d.max() - original_image_2d.min() + 1e-10)
    img_rgb = cv2.cvtColor(np.uint8(255 * original_image_2d), cv2.COLOR_GRAY2RGB)
    norm_heatmap = heatmap
    if np.max(heatmap) > 0:
        norm_heatmap = (heatmap - np.min(norm_heatmap)) / (np.max(norm_heatmap) - np.min(norm_heatmap) + 1e-10)
    cmap_obj = plt.get_cmap(cmap)
    cmap_img = (cmap_obj(norm_heatmap)[:,:,:3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, cmap_img, alpha, 0)
    return overlay


# --- Semantic Explanation Functions (New) ---
def generate_segmentation_explanation(predicted_mask, class_names_map, min_pixels):
    """Generates a descriptive text report for the predicted segmentation."""
    unique_classes = np.unique(predicted_mask)
    explanation = "### Anatomical Report\n\nBased on its analysis of this image slice, the AI model has identified the following structures:\n\n"
    found_parts = False
    
    for class_id in unique_classes:
        if class_id == 0:  # Skip background
            continue

        class_name = class_names_map.get(class_id, f"Class {class_id}")
        pixel_count = np.sum(predicted_mask == class_id)

        if pixel_count > min_pixels:
            explanation += f"- **{class_name}:** Identified with a total area of **{pixel_count} pixels**.\n"
            found_parts = True

    if not found_parts:
        explanation += "The model could not confidently identify any significant anatomical structures in this slice."

    return explanation

def generate_confidence_explanation(uncertainty_map):
    """
    Interprets the uncertainty map to provide a summary of prediction risk.
    """
    if uncertainty_map is None:
        return "Confidence assessment is unavailable."

    # Normalize uncertainty map (0 to 1)
    min_val = np.min(uncertainty_map)
    max_val = np.max(uncertainty_map)
    uncertainty_map_normalized = (uncertainty_map - min_val) / (max_val - min_val + 1e-10)
    
    # Analyze max uncertainty in the relevant brain area
    relevant_uncertainty = uncertainty_map_normalized[uncertainty_map_normalized > 0.05]
    
    if relevant_uncertainty.size == 0:
        return "### AI Confidence Assessment\n\n**Assessment:** High Confidence. The AI is highly certain about all predicted boundaries. Minimal review is necessary."
        
    mean_risk = np.mean(relevant_uncertainty)
    max_risk = np.max(relevant_uncertainty)
    
    if max_risk < 0.3:
        assessment = "High Confidence: The segmentation boundaries are predicted with very high certainty."
    elif max_risk < 0.5:
        assessment = f"Moderate Confidence: Overall segmentation is stable, but watch for small areas (Max Risk: {max_risk:.2f}) where the boundaries may be slightly fuzzy. Review those areas."
    else:
        assessment = f"⚠️ Low Confidence: The model struggled to define boundaries in significant areas (Max Risk: {max_risk:.2f}). **Manual review of the predicted mask is strongly recommended**."
        
    return f"### AI Confidence Assessment\n\n**Assessment:** {assessment}"

def generate_influence_explanation(xai_method, target_class_name):
    """Generates a written explanation for the XAI map overlay based on method."""
    if xai_method == "Regions of Focus (Attention)":
        return f"### AI Focus Map Interpretation\n\nThis map shows which general areas the AI prioritized when determining **all** structure boundaries. Highlighting (brighter colors) indicates regions that received the most weight in the global decision-making process."
    
    elif xai_method == "Regions of Influence": # Grad-CAM
        return f"### AI Influence Map Interpretation\n\nThis map highlights the specific pixels that had the strongest influence on the prediction of the **{target_class_name}**. The brighter the area (yellow/red), the more that region served as the 'evidence' for the AI."
        
    elif xai_method == "Feature Sensitivity": # Integrated Gradients
        return f"### AI Feature Sensitivity Interpretation\n\nThis map shows which image features (edges, textures) contributed most to the **{target_class_name}** prediction across the entire image. Brighter areas mean the AI is highly sensitive to the features in that region."
        
    return "Select an AI Focus Map method above to get a detailed explanation."


# --- Model Loading (Cached) ---
full_model_with_attention = None
model_for_gradcam = None

def get_all_conv_layers(model):
    """Dynamically finds all Conv2D layer names in the model."""
    layer_names = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer_names.append(layer.name)
        elif isinstance(layer, tf.keras.Model):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    layer_names.append(f"{layer.name}/{sub_layer.name}")
    return sorted(list(set(layer_names)))


def load_models_once():
    """Loads the trained Keras models only once."""
    global full_model_with_attention, model_for_gradcam
    if full_model_with_attention is None:
        print("Loading models for Gradio...")
        # Load the attention-enabled model for MC Dropout and Attention Map
        full_model_with_attention = create_model(num_classes=NUM_CLASSES, return_attention_map=True)
        # We need a dummy compile to ensure all layers are built before loading weights
        full_model_with_attention.compile(optimizer='adam', loss='mae') 
        full_model_with_attention.load_weights(TRAINED_MODEL_PATH)

        # Load the standard model for Grad-CAM/IG
        model_for_gradcam = create_model(num_classes=NUM_CLASSES, return_attention_map=False)
        model_for_gradcam.compile(optimizer='adam', loss='mae') 
        model_for_gradcam.load_weights(TRAINED_MODEL_PATH)
        print("Models loaded.")

# Load models once when the script starts
load_models_once()


# --- Data Loading (Cached) ---
@functools.lru_cache(maxsize=None)
def load_volume_data(img_path, label_path):
    """Loads NIfTI volume data."""
    img_volume = nib.load(img_path).get_fdata()
    label_volume = None
    if label_path and os.path.exists(label_path):
        label_volume = nib.load(label_path).get_fdata()
    return img_volume, label_volume

@functools.lru_cache(maxsize=None)
def get_all_filepaths_cached():
    # n=-1 loads all files
    return cd.create_dataset(PATH_INPUT_VOLUMES, PATH_LABEL_VOLUMES, n=-1, s=0.0)[0]

all_filepaths_raw = get_all_filepaths_cached()
volume_names_map = {os.path.basename(p[0]): p for p in all_filepaths_raw}
available_volumes = list(volume_names_map.keys())

def get_max_slices(volume_name):
    """Updates the slice slider max value based on the selected volume."""
    if volume_name:
        img_path, _ = volume_names_map[volume_name]
        img_volume, _ = load_volume_data(img_path, '')
        max_slices = img_volume.shape[0] - 1
        return gr.Slider(minimum=0, maximum=max_slices, step=1, value=max_slices // 2, interactive=True)
    return gr.Slider(minimum=0, maximum=0, step=1, value=0, interactive=False)


# --- Gradio Interface Function ---
def explain_segmentation_clinical(volume_name, slice_idx, target_class_display_name, selected_xai_method):
    
    # 1. Input Validation and Mapping
    if not volume_name or slice_idx is None or target_class_display_name is None:
        blank_image = np.zeros((256, 256), dtype=np.uint8)
        blank_rgb_image = np.zeros((256, 256, 3), dtype=np.uint8)
        return (blank_image, blank_image, blank_image, "Please select Volume, Slice, and Class.",
                blank_rgb_image, blank_rgb_image, "No segmentation explanation available.", 
                "N/A", "N/A")

    # Map target class display name back to ID
    reverse_map = {v: k for k, v in CLASSES_TO_ANALYZE_MAP.items()}
    target_class_idx = reverse_map.get(target_class_display_name, 1)

    # 2. Data Loading and Preprocessing
    img_path, label_path = volume_names_map[volume_name]
    img_volume, label_volume = load_volume_data(img_path, label_path)

    original_slice = img_volume[slice_idx, :, :]
    original_label = label_volume[slice_idx, :, :] if label_volume is not None else np.zeros_like(original_slice, dtype=np.int32)
    input_image_processed, ground_truth_label = preprocess_slice(original_slice, original_label)
    input_image_batch = tf.expand_dims(tf.constant(input_image_processed, dtype=tf.float32), axis=0)

    # 3. Model Prediction and XAI Data Generation (MC Dropout)
    mean_prediction, uncertainty_map, averaged_attention_map = predict_with_mc_dropout_and_attention(
        model_path=TRAINED_MODEL_PATH,
        input_image_batch=input_image_batch,
        num_samples=MC_SAMPLES,
        num_classes=NUM_CLASSES
    )
    
    predicted_mask_full = np.argmax(mean_prediction, axis=-1)

    # 4. Mask Preparation and Metrics
    gt_mask_for_class = (ground_truth_label == target_class_idx).astype(np.uint8) * 255
    predicted_mask_for_class = (predicted_mask_full == target_class_idx).astype(np.uint8) * 255

    jaccard = jaccard_index(predicted_mask_for_class, gt_mask_for_class)
    
    # 5. Status and Quality Check
    gt_pixel_count = np.sum(gt_mask_for_class)
    status = f"Quality Check Score (Jaccard Index): **{jaccard:.3f}**"
    if gt_pixel_count > MIN_PIXELS_FOR_ANALYSIS and jaccard < POOR_PERFORMANCE_JACCARD_THRESHOLD:
        status += " ⚠️ **Requires Radiologist Review (Score Below 0.5)**"
    elif gt_pixel_count > MIN_PIXELS_FOR_ANALYSIS:
        status += " ✅ **Segmentation Acceptable**"

    # 6. XAI Heatmaps Generation
    xai_heatmap = np.zeros_like(input_image_processed, dtype=np.float32)
    xai_cmap = 'hot'
    
    if selected_xai_method == "Regions of Influence":
        xai_heatmap = generate_grad_cam_for_class(model_for_gradcam, input_image_batch, target_class_idx, layer_name='conv2d_10') # Assuming a late conv layer
    elif selected_xai_method == "Feature Sensitivity":
        xai_heatmap = integrated_gradients(model_for_gradcam, input_image_batch, target_class_idx)
        xai_cmap = 'plasma'
    elif selected_xai_method == "Regions of Focus (Attention)":
        xai_heatmap = averaged_attention_map
        xai_cmap = 'viridis'

    # 7. Image Outputs
    original_image_display = np.uint8(255 * input_image_processed.squeeze())
    
    # XAI (Influence) Map Overlay
    xai_image_display = overlay_heatmap(input_image_processed, xai_heatmap, cmap=xai_cmap)
    
    # Confidence Map Overlay (Uncertainty)
    confidence_map_display = overlay_heatmap(input_image_processed, uncertainty_map, cmap='magma_r', alpha=0.7)

    # 8. Written Explanations
    segmentation_explanation = generate_segmentation_explanation(predicted_mask_full, CLASS_NAMES, MIN_PIXELS_FOR_ANALYSIS)
    confidence_explanation = generate_confidence_explanation(uncertainty_map)
    influence_explanation = generate_influence_explanation(selected_xai_method, target_class_display_name)

    # 9. Return Results
    return (
        original_image_display,
        gt_mask_for_class,
        predicted_mask_for_class,
        status,
        xai_image_display,
        confidence_map_display,
        segmentation_explanation,
        confidence_explanation,
        influence_explanation
    )


# --- Gradio Interface Layout using gr.Blocks ---
with gr.Blocks(title="Fetal Brain Segmentation Clinical Review 🧠") as clinical_demo:
    gr.Markdown("# 🤰 Fetal Brain Segmentation Review Tool")
    gr.Markdown("Tool for Radiologists/Clinicians to verify AI segmentations and assess confidence.")

    # State component to hold the current class ID
    target_class_id_state = gr.State(value=1)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Data Selection")
            volume_dropdown = gr.Dropdown(choices=available_volumes, label="Select Patient Scan (Volume)")
            slice_slider = gr.Slider(minimum=0, maximum=10, step=1, label="Select Slice View", interactive=False, value=0)
            
            # Use Class Names for clarity
            class_dropdown = gr.Dropdown(choices=list(CLASSES_TO_ANALYZE_MAP.values()), label="Target Structure for Analysis", value=CLASSES_TO_ANALYZE_MAP[CLASSES_TO_ANALYZE[0]])
            
            submit_btn = gr.Button("Analyze Segmentation", variant="primary")

            gr.Markdown("### 2. AI Influence and Explanation")
            # Simplified XAI radio buttons
            xai_method_radio = gr.Radio(choices=list(CLINICAL_XAI_METHODS.values()),
                                        label="View AI's Focus Map", value="Regions of Influence")
            
        with gr.Column(scale=3):
            # --- ROW 1: Segmentation Results ---
            gr.Markdown("### Segmentation Results and Quality Check")
            
            with gr.Row():
                original_image_display_ui = gr.Image(label="Original MRI Slice", type="numpy", show_label=True)
                gt_mask_display_ui = gr.Image(label="Expert Mask (Ground Truth)", type="numpy", show_label=True)
                predicted_mask_display_ui = gr.Image(label="AI Predicted Mask", type="numpy", show_label=True)

            status_textbox = gr.Textbox(label="AI Quality Check Score", interactive=False, lines=2)

            # --- ROW 2: Written Reports ---
            gr.Markdown("### Written AI Reports")

            with gr.Row():
                # Markdown should only use 'label' or 'value' (initial content).
                segmentation_explanation_textbox = gr.Markdown(label="Segmentation Summary Report") 
            
            with gr.Row():
                # Markdown should only use 'label' or 'value' (initial content).
                confidence_explanation_textbox = gr.Markdown(label="AI Confidence Assessment")
            
            # --- ROW 3: Visual Explanations (Trust Maps) ---
            gr.Markdown("### Visual Trust Maps")
            
            with gr.Row():
                # XAI (Influence) Map Overlay
                xai_image_display_ui = gr.Image(label="AI Influence Map (Where did the AI look?)", type="numpy", show_label=True)
                # Confidence Map Overlay (NEW)
                confidence_map_display_ui = gr.Image(label="AI Confidence Map (Risk Areas)", type="numpy", show_label=True)

            # XAI Influence Explanation
            influence_explanation_textbox = gr.Markdown(label="Influence Map Explanation", interactive=False, lines=5)


    # --- Event Listeners ---
    volume_dropdown.change(fn=get_max_slices, inputs=volume_dropdown, outputs=slice_slider, queue=False)
    
    submit_btn.click(
        fn=explain_segmentation_clinical,
        inputs=[
            volume_dropdown,
            slice_slider,
            class_dropdown, # This passes the display name (e.g., 'Cerebellum')
            xai_method_radio,
        ],
        outputs=[
            original_image_display_ui,
            gt_mask_display_ui,
            predicted_mask_display_ui,
            status_textbox,
            xai_image_display_ui,
            confidence_map_display_ui,
            segmentation_explanation_textbox,
            confidence_explanation_textbox,
            influence_explanation_textbox
        ]
    )

clinical_demo.launch(share=True)
