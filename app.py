import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import nibabel as nib
import os
import functools

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
# ⭐ EDITED: Renamed and set a threshold appropriate for the Dice Coefficient.
POOR_PERFORMANCE_DICE_THRESHOLD = 0.65 
NUM_FILTERS_TO_SHOW = 8

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

# --- Helper Functions (Metrics) ---
def jaccard_index(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def dice_coefficient(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    # The smooth term (1e-10) prevents division by zero if both masks are empty
    return (2. * intersection) / (np.sum(mask1) + np.sum(mask2) + 1e-10)

def precision_score(mask1, mask2):
    true_positives = np.sum(mask1 * mask2)
    predicted_positives = np.sum(mask1)
    return true_positives / (predicted_positives + 1e-10)

def recall_score(mask1, mask2):
    true_positives = np.sum(mask1 * mask2)
    actual_positives = np.sum(mask2)
    return true_positives / (actual_positives + 1e-10)

# --- Model Loading (Cached for performance) ---
full_model_with_attention = None
model_for_gradcam = None
ALL_CONV_LAYERS = []

def get_all_conv_layers(model):
    """Dynamically finds all Conv2D layer names in the model."""
    layer_names = []
    for layer in model.layers:
        # Check for Conv2D layers directly
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer_names.append(layer.name)
        # Handle functional sub-models
        elif isinstance(layer, tf.keras.Model):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    layer_names.append(f"{layer.name}/{sub_layer.name}")
    return sorted(list(set(layer_names)))


def load_models_once():
    """Loads the trained Keras models only once and finds all Conv2D layers."""
    global full_model_with_attention, model_for_gradcam, ALL_CONV_LAYERS
    if full_model_with_attention is None:
        print("Loading models for Gradio...")
        full_model_with_attention = create_model(num_classes=NUM_CLASSES, return_attention_map=True)
        full_model_with_attention.load_weights(TRAINED_MODEL_PATH)

        model_for_gradcam = create_model(num_classes=NUM_CLASSES, return_attention_map=False)
        model_for_gradcam.load_weights(TRAINED_MODEL_PATH)
        print("Models loaded.")

        # Find all convolutional layers after models are loaded
        ALL_CONV_LAYERS = get_all_conv_layers(full_model_with_attention)

# Load models once when the script starts
load_models_once()


# --- Data Loading (Cached for performance) ---
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
    return cd.create_dataset(PATH_INPUT_VOLUMES, PATH_LABEL_VOLUMES, n=-1, s=0.0)[0]

all_filepaths_raw = get_all_filepaths_cached()
volume_names_map = {os.path.basename(p[0]): p for p in all_filepaths_raw}
available_volumes = list(volume_names_map.keys())

def get_max_slices(volume_name):
    if volume_name:
        img_path, _ = volume_names_map[volume_name]
        img_volume, _ = load_volume_data(img_path, '')
        max_slices = img_volume.shape[0] - 1
        return gr.Slider(minimum=0, maximum=max_slices, step=1, value=0, interactive=True)
    return gr.Slider(minimum=0, maximum=0, step=1, value=0, interactive=False)


# --- XAI Implementations ---
def generate_grad_cam_for_class(model, input_image_batch, target_class_idx, layer_name='gradcam_target_conv', logits_layer_name='logits_output'):
    """Generates a Grad-CAM heatmap."""
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.get_layer(logits_layer_name).output]
    )
    with tf.GradientTape() as tape:
        tape.watch(input_image_batch)
        last_conv_layer_output, logits = grad_model(input_image_batch)
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

def integrated_gradients(model, input_image_batch, target_class_idx, steps=50, logits_layer_name='logits_output'):
    """Computes Integrated Gradients."""
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

def visualize_filter_activations_gradio(model, input_image_batch, layer_name, num_filters_to_show=NUM_FILTERS_TO_SHOW):
    """
    Visualizes filter activations for a specific layer.
    """
    images_to_display = []
    # Add the input image as the first image
    images_to_display.append(np.uint8(255 * input_image_batch.numpy().squeeze()))

    if not layer_name:
        print("No layer selected for activation visualization.")
        return images_to_display

    try:
        layer_output = model.get_layer(layer_name).output
        act_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_output)
        activations = act_model.predict(input_image_batch)

        if activations.ndim == 4:
            activations = activations.squeeze(axis=0)
            num_channels = activations.shape[-1]
            display_channels = min(num_channels, num_filters_to_show)

            for channel_idx in range(display_channels):
                act_map = activations[:, :, channel_idx]
                if np.max(act_map) > 0:
                    act_map = (act_map - np.min(act_map)) / (np.max(act_map) - np.min(act_map) + 1e-10)
                act_map_colored = (plt.cm.magma(act_map)[:,:,:3] * 255).astype(np.uint8)
                images_to_display.append(act_map_colored)
        else:
            print(f"Layer {layer_name} output has unexpected shape: {activations.shape}. Skipping.")
    except Exception as e:
        print(f"Error visualizing activations for layer {layer_name}: {e}")
        images_to_display.append(np.zeros((128,128,3), dtype=np.uint8))

    return images_to_display

def overlay_heatmap(original_image_2d, heatmap, cmap='hot', alpha=0.5):
    """
    Overlays a heatmap on a grayscale image.
    """
    if original_image_2d.dtype != np.float32:
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

def generate_segmentation_explanation(predicted_mask, class_names_map, min_pixels):
    """Generates a descriptive text for the predicted segmentation."""
    unique_classes = np.unique(predicted_mask)
    explanation = "### Segmentation Analysis\n\n"
    found_parts = False

    for class_id in unique_classes:
        if class_id == 0: # Skip background
            continue

        class_name = class_names_map.get(class_id, f"Class {class_id}")
        pixel_count = np.sum(predicted_mask == class_id)

        if pixel_count > min_pixels:
            explanation += f"- **{class_name}:** The model has identified this structure, with a total area of {pixel_count} pixels.\n"
            found_parts = True

    if not found_parts:
        explanation += "The model could not confidently identify any significant anatomical structures in this slice."

    return explanation


# --- Gradio Interface Function ---
def explain_segmentation(volume_name, slice_idx, target_class_idx, selected_xai_method, selected_metric, selected_layer_for_activations):
    """
    Main function for the Gradio interface that performs segmentation and XAI.
    """
    if not volume_name or slice_idx is None or target_class_idx is None:
        blank_image = np.zeros((128, 128), dtype=np.uint8)
        blank_rgb_image = np.zeros((128, 128, 3), dtype=np.uint8)
        return (blank_image, blank_image, blank_image, "Please select a Volume, Slice, and Class.",
                blank_rgb_image, [], "No segmentation explanation available.")

    # Load selected volume data
    img_path, label_path = volume_names_map[volume_name]
    img_volume, label_volume = load_volume_data(img_path, label_path)

    # Preprocess slice
    original_slice = img_volume[slice_idx, :, :]
    if label_volume is not None:
        original_label = label_volume[slice_idx, :, :]
    else:
        original_label = np.zeros_like(original_slice, dtype=np.int32)
    input_image_processed, ground_truth_label = preprocess_slice(original_slice, original_label)
    input_image_batch = tf.expand_dims(tf.constant(input_image_processed, dtype=tf.float32), axis=0)

    # Get model predictions
    segmentation_output_softmax, _, averaged_attention_map_raw = full_model_with_attention(input_image_batch)
    predicted_mask_full = np.argmax(segmentation_output_softmax.numpy().squeeze(), axis=-1)

    # Convert masks to 0-255 uint8 for display
    gt_mask_for_class = (ground_truth_label == target_class_idx).astype(np.uint8) * 255
    predicted_mask_for_class = (predicted_mask_full == target_class_idx).astype(np.uint8) * 255

    # Calculate all metrics
    dice_value = dice_coefficient(predicted_mask_for_class, gt_mask_for_class)
    metrics = {
        "Jaccard Index": jaccard_index(predicted_mask_for_class, gt_mask_for_class),
        "Dice Coefficient": dice_value,
        "Precision": precision_score(predicted_mask_for_class, gt_mask_for_class),
        "Recall": recall_score(predicted_mask_for_class, gt_mask_for_class)
    }

    # Format status text based on selected metric
    # ⭐ CHANGE 1: Default to "Dice Coefficient" if the selected metric is not found.
    selected_metric_value = metrics.get(selected_metric, metrics["Dice Coefficient"])
    status = f"Segmentation Metric ({selected_metric}): {selected_metric_value:.3f}"
    gt_pixel_count = np.sum(gt_mask_for_class)
    
    # ⭐ CHANGE 2: Use Dice Coefficient and the new Dice threshold for performance check.
    if gt_pixel_count > MIN_PIXELS_FOR_ANALYSIS and dice_value < POOR_PERFORMANCE_DICE_THRESHOLD:
        status += " 🚨 Problematic Performance"
    elif gt_pixel_count > MIN_PIXELS_FOR_ANALYSIS:
        status += " ✅ Good Performance"

    # Prepare base images and XAI output
    original_image_display = np.uint8(255 * input_image_batch.numpy().squeeze())
    gt_mask_display = gt_mask_for_class
    predicted_mask_display = predicted_mask_for_class
    xai_image_display = np.zeros((original_image_display.shape[0], original_image_display.shape[1], 3), dtype=np.uint8)
    filter_activation_gallery_output = []
    
    if selected_xai_method == "Grad-CAM":
        heatmap = generate_grad_cam_for_class(model_for_gradcam, input_image_batch, target_class_idx)
        xai_image_display = overlay_heatmap(input_image_batch.numpy().squeeze(), heatmap, cmap='hot')
    elif selected_xai_method == "Integrated Gradients":
        heatmap = integrated_gradients(model_for_gradcam, input_image_batch, target_class_idx)
        xai_image_display = overlay_heatmap(input_image_batch.numpy().squeeze(), heatmap, cmap='plasma')
    elif selected_xai_method == "Attention Map":
        attn_map = tf.reduce_mean(averaged_attention_map_raw[0], axis=-1).numpy()
        xai_image_display = overlay_heatmap(input_image_batch.numpy().squeeze(), attn_map, cmap='viridis')
    elif selected_xai_method == "Filter Activations":
        filter_activation_gallery_output = visualize_filter_activations_gradio(full_model_with_attention, input_image_batch, selected_layer_for_activations)
        # When showing filter activations, the main XAI image slot will be blank
        xai_image_display = np.zeros_like(xai_image_display)

    # Generate the semantic explanation for the segmentation
    segmentation_explanation = generate_segmentation_explanation(predicted_mask_full, CLASS_NAMES, MIN_PIXELS_FOR_ANALYSIS)

    return original_image_display, gt_mask_display, predicted_mask_display, status, xai_image_display, filter_activation_gallery_output, segmentation_explanation


# --- Gradio Interface Layout using gr.Blocks ---
with gr.Blocks(title="Fetal Brain Segmentation XAI Dashboard 🧠") as demo:
    gr.Markdown("# 🧠 Fetal Brain Segmentation Explainability Dashboard")
    gr.Markdown("Select a fetal MRI volume, slice, and target anatomical class to view the model's segmentation prediction and various XAI explanations.")

    with gr.Row():
        with gr.Column(scale=1):
            volume_dropdown = gr.Dropdown(choices=available_volumes, label="Select Volume")
            slice_slider = gr.Slider(minimum=0, maximum=10, step=1, label="Select Slice Index", interactive=False, value=0)
            class_dropdown = gr.Dropdown(choices=CLASSES_TO_ANALYZE, label="Select Target Class", value=CLASSES_TO_ANALYZE[0])
            
            xai_method_radio = gr.Radio(choices=["Grad-CAM", "Integrated Gradients", "Attention Map", "Filter Activations"],
                                        label="Choose XAI Method", value="Grad-CAM")
            
            # New dropdown for metrics
            metrics_dropdown = gr.Dropdown(
                choices=["Jaccard Index", "Dice Coefficient", "Precision", "Recall"],
                label="Select Performance Metric",
                # ⭐ CHANGE 3: Set the default value to Dice Coefficient
                value="Dice Coefficient" 
            )
            
            # New dropdown for layer selection
            layer_dropdown = gr.Dropdown(
                choices=ALL_CONV_LAYERS,
                label="Select Layer for Activations",
                value=ALL_CONV_LAYERS[0] if ALL_CONV_LAYERS else None,
                visible=False # Initially hidden
            )

            submit_btn = gr.Button("Analyze!")

        with gr.Column(scale=2):
            # Core image outputs
            with gr.Row():
                original_image_display_ui = gr.Image(label="Original Image", type="numpy", show_label=True)
                gt_mask_display_ui = gr.Image(label="Ground Truth Mask", type="numpy", show_label=True)
                predicted_mask_display_ui = gr.Image(label="Predicted Mask", type="numpy", show_label=True)

            with gr.Row():
                # Textbox for metrics
                status_textbox = gr.Textbox(label="Segmentation Metrics", interactive=False)
                # Textbox for semantic explanation
                segmentation_explanation_textbox = gr.Textbox(label="XAI Segmentation Explanation", interactive=False, lines=4)

            # XAI heatmap output
            xai_image_display_ui = gr.Image(label="XAI Heatmap Overlay", type="numpy", visible=True, show_label=True)

            # Grad-CAM explanation text
            gradcam_explanation_textbox = gr.Textbox(
                label="Grad-CAM Explanation",
                value="Grad-CAM highlights the most important regions for a specific class prediction. Red/yellow indicates high importance, while blue/purple indicates low importance.",
                interactive=False,
                visible=True
            )

            # Gallery for filter activations
            filter_activation_gallery_ui = gr.Gallery(
                label="Filter Activations (Input + Filters)",
                elem_id="gallery",
                object_fit="contain",
                height="auto",
                visible=False
            )


    # --- Event Listeners ---
    volume_dropdown.change(fn=get_max_slices, inputs=volume_dropdown, outputs=slice_slider, queue=False)

    # Function to toggle visibility of XAI-specific UI elements
    def toggle_xai_visibility(xai_method):
        if xai_method == 'Filter Activations':
            return gr.Image(visible=False), gr.Gallery(visible=True), gr.Textbox(visible=False), gr.Dropdown(visible=True)
        elif xai_method == 'Grad-CAM':
            return gr.Image(visible=True), gr.Gallery(visible=False), gr.Textbox(visible=True), gr.Dropdown(visible=False)
        else:
            return gr.Image(visible=True), gr.Gallery(visible=False), gr.Textbox(visible=False), gr.Dropdown(visible=False)

    xai_method_radio.change(
        fn=toggle_xai_visibility,
        inputs=[xai_method_radio],
        outputs=[xai_image_display_ui, filter_activation_gallery_ui, gradcam_explanation_textbox, layer_dropdown],
        queue=False
    )

    submit_btn.click(
        fn=explain_segmentation,
        inputs=[
            volume_dropdown,
            slice_slider,
            class_dropdown,
            xai_method_radio,
            metrics_dropdown,
            layer_dropdown
        ],
        outputs=[
            original_image_display_ui,
            gt_mask_display_ui,
            predicted_mask_display_ui,
            status_textbox,
            xai_image_display_ui,
            filter_activation_gallery_ui,
            segmentation_explanation_textbox
        ]
    )

demo.launch(share=True)
