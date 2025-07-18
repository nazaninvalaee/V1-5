import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import nibabel as nib
import os
import io
import functools # <--- ADD THIS LINE

# Ensure your 'ensem_4_mod_4_no_mod' file is correctly set up.
from ensem_4_mod_4_no_mod import create_model
import create_dataset as cd
from create_dataset import preprocess_slice

# --- Configuration ---
# IMPORTANT: Adjust these paths to your specific environment
TRAINED_MODEL_PATH = '/content/drive/MyDrive/fetal-brain-segmentation-v1.5/checkpoints/Model.keras'
PATH_INPUT_VOLUMES = '/content/drive/MyDrive/feta_2.1/nii_files_input'
PATH_LABEL_VOLUMES = '/content/drive/MyDrive/feta_2.1/nii_files_output'
NUM_CLASSES = 8

CLASSES_TO_ANALYZE = [1, 2, 3, 4, 5, 6, 7]
MIN_PIXELS_FOR_ANALYSIS = 1000
POOR_PERFORMANCE_JACCARD_THRESHOLD = 0.5
NUM_FILTERS_TO_SHOW = 8

# --- Helper Functions (Jaccard) ---
def jaccard_index(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

# --- Model Loading (Cached for performance) ---
full_model_with_attention = None
model_for_gradcam = None

def load_models_once():
    """Loads the trained Keras models only once."""
    global full_model_with_attention, model_for_gradcam
    if full_model_with_attention is None:
        print("Loading models for Gradio...")
        full_model_with_attention = create_model(num_classes=NUM_CLASSES, return_attention_map=True)
        full_model_with_attention.load_weights(TRAINED_MODEL_PATH)

        model_for_gradcam = create_model(num_classes=NUM_CLASSES, return_attention_map=False)
        model_for_gradcam.load_weights(TRAINED_MODEL_PATH)
        print("Models loaded.")

# Load models once when the script starts
load_models_once()


# --- Data Loading (Cached for performance) ---
# Use functools.lru_cache for reliable caching of data loading functions
@functools.lru_cache(maxsize=None) # <--- CHANGE THIS LINE
def load_volume_data(img_path, label_path):
    """Loads NIfTI volume data."""
    img_volume = nib.load(img_path).get_fdata()
    label_volume = nib.load(label_path).get_fdata()
    return img_volume, label_volume

@functools.lru_cache(maxsize=None) # <--- CHANGE THIS LINE
def get_all_filepaths_cached():
    return cd.create_dataset(PATH_INPUT_VOLUMES, PATH_LABEL_VOLUMES, n=-1, s=0.0)[0]

all_filepaths_raw = get_all_filepaths_cached()
volume_names_map = {os.path.basename(p[0]): p for p in all_filepaths_raw}
available_volumes = list(volume_names_map.keys())

# New function to get max slices for a selected volume
def get_max_slices(volume_name):
    if volume_name:
        img_path, _ = volume_names_map[volume_name]
        img_volume, _ = load_volume_data(img_path, '') # Only need img_volume for shape
        # Return an update object for the slider component
        return gr.Slider(minimum=0, maximum=img_volume.shape[0] - 1, step=1, value=0, interactive=True)
    # Return an inactive slider if no volume is selected
    return gr.Slider(minimum=0, maximum=0, step=1, value=0, interactive=False)


# --- XAI Implementations (adapted for Gradio to return image objects) ---

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
    heatmap_resized = cv2.resize(normalized_heatmap, (original_width, original_height))

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
    heatmap_resized = cv2.resize(normalized_heatmap, (original_width, original_height))

    return heatmap_resized

def visualize_filter_activations_gradio(model, input_image_batch, num_filters_to_show=NUM_FILTERS_TO_SHOW):
    """
    Visualizes filter activations and returns a list of PIL Images (or similar).
    Adapted for Gradio to return image outputs directly.
    """
    detected_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            detected_layers.append(layer.name)
        elif isinstance(layer, tf.keras.Model) and 'functional' in layer.name:
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    detected_layers.append(f"{layer.name}/{sub_layer.name}")

    detected_layers = sorted(list(set(detected_layers)))
    layer_names_to_visualize = detected_layers[:6] # Display up to first 6 detected layers

    if not layer_names_to_visualize:
        print("No suitable Conv2D layers found for activation visualization.")
        return [np.zeros((128,128,3), dtype=np.uint8)] # Return a blank image

    images_to_display = []
    # Add the input image as the first image in the gallery
    images_to_display.append(np.uint8(255 * input_image_batch.numpy().squeeze()))

    for name in layer_names_to_visualize:
        try:
            layer_output = model.get_layer(name).output
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
                    # Convert activation map to a displayable RGB image for Gradio gallery
                    act_map_colored = (plt.cm.magma(act_map)[:,:,:3] * 255).astype(np.uint8)
                    images_to_display.append(act_map_colored)
            else:
                print(f"Layer {name} output has unexpected shape: {activations.shape}. Skipping.")
        except Exception as e:
            print(f"Error visualizing activations for layer {name}: {e}")
            images_to_display.append(np.zeros((128,128,3), dtype=np.uint8)) # Add a blank image on error

    return images_to_display # Return the list of images for the gallery


def overlay_heatmap(original_image_2d, heatmap, cmap='hot', alpha=0.5):
    """
    Overlays a heatmap on a grayscale image.
    Returns an RGB image suitable for Gradio display.
    """
    if original_image_2d.dtype != np.float32:
        original_image_2d = original_image_2d.astype(np.float32)
        original_image_2d = (original_image_2d - original_image_2d.min()) / (original_image_2d.max() - original_image_2d.min() + 1e-10)

    img_rgb = cv2.cvtColor(np.uint8(255 * original_image_2d), cv2.COLOR_GRAY2RGB)

    norm_heatmap = heatmap
    if np.max(heatmap) > 0:
        norm_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)

    cmap_obj = plt.get_cmap(cmap)
    cmap_img = (cmap_obj(norm_heatmap)[:,:,:3] * 255).astype(np.uint8)

    overlay = cv2.addWeighted(img_rgb, 1 - alpha, cmap_img, alpha, 0)
    return overlay

# --- Gradio Interface Function ---
def explain_segmentation(volume_name, slice_idx, target_class_idx, xai_method):
    """
    Main function for the Gradio interface that performs segmentation and XAI.
    """
    if not volume_name or slice_idx is None or target_class_idx is None:
        # Return blank outputs or initial state if inputs are not fully selected
        blank_image_128_128 = np.zeros((128, 128), dtype=np.uint8)
        blank_rgb_image_128_128_3 = np.zeros((128, 128, 3), dtype=np.uint8)
        # Ensure correct number of outputs for the main function
        return (blank_image_128_128, blank_image_128_128, blank_image_128_128,
                "Please select a Volume, Slice, and Class.",
                blank_rgb_image_128_128_3, [])


    # Load selected volume data
    img_path, label_path = volume_names_map[volume_name]
    img_volume, label_volume = load_volume_data(img_path, label_path)

    # Preprocess slice
    original_slice = img_volume[slice_idx, :, :]
    original_label = label_volume[slice_idx, :, :]
    input_image_processed, ground_truth_label = preprocess_slice(original_slice, original_label)
    input_image_batch = tf.expand_dims(tf.constant(input_image_processed, dtype=tf.float32), axis=0)

    # Get model predictions
    segmentation_output_softmax, _, averaged_attention_map_raw = full_model_with_attention(input_image_batch)
    predicted_mask_full = np.argmax(segmentation_output_softmax.numpy().squeeze(), axis=-1)

    # Convert masks to 0-255 uint8 for display
    gt_mask_for_class = (ground_truth_label == target_class_idx).astype(np.uint8) * 255
    predicted_mask_for_class = (predicted_mask_full == target_class_idx).astype(np.uint8) * 255

    # Calculate metrics
    seg_jaccard = jaccard_index(predicted_mask_for_class, gt_mask_for_class)
    gt_pixel_count = np.sum(gt_mask_for_class)
    status = f"Segmentation Jaccard: {seg_jaccard:.3f} | GT Pixels: {int(gt_pixel_count)} | Status: {'🚨 Problematic' if seg_jaccard < POOR_PERFORMANCE_JACCARD_THRESHOLD else '✅ Good Performance'}"

    # Prepare base images for display
    original_image_display = np.uint8(255 * input_image_batch.numpy().squeeze())
    gt_mask_display = gt_mask_for_class
    predicted_mask_display = predicted_mask_for_class

    xai_image_display = None
    filter_activation_gallery_output = [] # This will be the list of images for the gallery

    if xai_method == "Grad-CAM":
        heatmap = generate_grad_cam_for_class(model_for_gradcam, input_image_batch, target_class_idx)
        xai_image_display = overlay_heatmap(input_image_batch.numpy().squeeze(), heatmap, cmap='hot')
    elif xai_method == "Integrated Gradients":
        heatmap = integrated_gradients(model_for_gradcam, input_image_batch, target_class_idx)
        xai_image_display = overlay_heatmap(input_image_batch.numpy().squeeze(), heatmap, cmap='plasma')
    elif xai_method == "Attention Map":
        attn_map = tf.reduce_mean(averaged_attention_map_raw[0], axis=-1).numpy()
        xai_image_display = overlay_heatmap(input_image_batch.numpy().squeeze(), attn_map, cmap='viridis')
    elif xai_method == "Filter Activations":
        filter_activation_gallery_output = visualize_filter_activations_gradio(full_model_with_attention, input_image_batch)
        # When showing filter activations, the main XAI image slot will be blank
        xai_image_display = np.zeros((original_image_display.shape[0], original_image_display.shape[1], 3), dtype=np.uint8)


    return original_image_display, gt_mask_display, predicted_mask_display, status, xai_image_display, filter_activation_gallery_output


# --- Gradio Interface Layout using gr.Blocks ---
class_names_map_for_gradio = {i: f"Class {i}" for i in CLASSES_TO_ANALYZE}

with gr.Blocks(title="Fetal Brain Segmentation XAI Dashboard 🧠") as demo:
    gr.Markdown("# 🧠 Fetal Brain Segmentation Explainability Dashboard")
    gr.Markdown("Select a fetal MRI volume, slice, and target anatomical class to view the model's segmentation prediction and various XAI explanations (Grad-CAM, Integrated Gradients, Attention Maps, Filter Activations).")

    with gr.Row():
        with gr.Column(scale=1):
            volume_dropdown = gr.Dropdown(choices=available_volumes, label="Select Volume")
            # Initialize slider with a default range, it will be updated dynamically
            slice_slider = gr.Slider(minimum=0, maximum=10, step=1, label="Select Slice Index", interactive=False, value=0)
            class_dropdown = gr.Dropdown(choices=CLASSES_TO_ANALYZE, label="Select Target Class", value=CLASSES_TO_ANALYZE[0], format_func=lambda x: class_names_map_for_gradio[x])
            xai_method_radio = gr.Radio(choices=["Original/Prediction", "Grad-CAM", "Integrated Gradients", "Attention Map", "Filter Activations"],
                                         label="Choose XAI Method", value="Original/Prediction")

            # Metrics output
            status_textbox = gr.Textbox(label="Segmentation Metrics & Status", interactive=False)

            submit_btn = gr.Button("Analyze!")

        with gr.Column(scale=2):
            # Core image outputs
            with gr.Row():
                original_image_display_ui = gr.Image(label="Original Image", type="numpy", show_label=True)
                gt_mask_display_ui = gr.Image(label="Ground Truth Mask", type="numpy", show_label=True)
                predicted_mask_display_ui = gr.Image(label="Predicted Mask", type="numpy", show_label=True)

            # XAI heatmap output (will show one of Grad-CAM, IG, Attention Map)
            xai_image_display_ui = gr.Image(label="XAI Heatmap Overlay", type="numpy", visible=True, show_label=True)

            # Gallery for filter activations (will be visible only when "Filter Activations" is chosen)
            filter_activation_gallery_ui = gr.Gallery(label="Filter Activations (Input + Filters)", show_label=True, elem_id="gallery", object_fit="contain", height="auto", visible=False)


    # --- Event Listeners ---

    # When volume_dropdown changes, update slice_slider
    volume_dropdown.change(
        fn=get_max_slices, # This function returns an updated gr.Slider component
        inputs=volume_dropdown,
        outputs=slice_slider,
        queue=False # This update should happen quickly and not block the queue
    )

    # When submit_btn is clicked, run the main explanation function
    submit_btn.click(
        fn=explain_segmentation,
        inputs=[
            volume_dropdown,
            slice_slider,
            class_dropdown,
            xai_method_radio
        ],
        outputs=[
            original_image_display_ui,
            gt_mask_display_ui,
            predicted_mask_display_ui,
            status_textbox,
            xai_image_display_ui,
            filter_activation_gallery_ui
        ]
    )

    # Update visibility when xai_method_radio changes
    xai_method_radio.change(
        fn=None, # No Python function needed, just client-side JS
        inputs=[xai_method_radio],
        outputs=[xai_image_display_ui, filter_activation_gallery_ui],
        _js="""
        (xai_method) => {
            if (xai_method === 'Filter Activations') {
                return [gradio_image_update(visible=false), gradio_image_update(visible=true)];
            } else {
                return [gradio_image_update(visible=true), gradio_image_update(visible=false)];
            }
        }
        """
    )


demo.launch(share=True) # Set share=True to get a public link
