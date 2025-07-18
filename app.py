import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import nibabel as nib
import os
import io

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
# Use a global variable to store models, as Gradio functions should be pure
# and not re-load models on every call if possible.
# Gradio's caching decorators can help manage this if functions are structured correctly.
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

# Call this once at the start of the script
load_models_once()

# --- Data Loading ---
all_filepaths_raw = cd.create_dataset(PATH_INPUT_VOLUMES, PATH_LABEL_VOLUMES, n=-1, s=0.0)[0]
volume_names_map = {os.path.basename(p[0]): p for p in all_filepaths_raw}
available_volumes = list(volume_names_map.keys())

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
    layer_names_to_visualize = detected_layers[:6]

    if not layer_names_to_visualize:
        print("No suitable Conv2D layers found for activation visualization.")
        return [np.zeros((128,128,3), dtype=np.uint8)] # Return a blank image

    images_to_display = []
    for name in layer_names_to_visualize:
        try:
            layer_output = model.get_layer(name).output
            act_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_output)
            activations = act_model.predict(input_image_batch)
            
            if activations.ndim == 4:
                activations = activations.squeeze(axis=0)

                num_channels = activations.shape[-1]
                display_channels = min(num_channels, num_filters_to_show)

                cols = display_channels + 1 if display_channels > 0 else 1
                rows = 1
                if cols > 6:
                    cols = 6
                    rows = (display_channels + 1 + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
                axes = axes.flatten()

                # Input image
                axes[0].imshow(input_image_batch.numpy().squeeze(), cmap='gray')
                axes[0].set_title('Input')
                axes[0].axis('off')

                for channel_idx in range(display_channels):
                    ax = axes[channel_idx + 1]
                    act_map = activations[:, :, channel_idx]
                    if np.max(act_map) > 0:
                        act_map = (act_map - np.min(act_map)) / (np.max(act_map) - np.min(act_map) + 1e-10)
                    ax.imshow(act_map, cmap='magma')
                    ax.set_title(f'Filter {channel_idx+1}')
                    ax.axis('off')
                
                for j in range(display_channels + 1, len(axes)):
                    axes[j].axis('off')

                plt.suptitle(f'Activations for Layer: {name}', fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                # Convert matplotlib figure to an image that Gradio can display
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                plt.close(fig) # Close the figure to free memory
                images_to_display.append(buf.getvalue()) # Gradio can handle bytes for images

            else:
                print(f"Layer {name} output has unexpected shape: {activations.shape}. Skipping.")
        except Exception as e:
            print(f"Error visualizing activations for layer {name}: {e}")
            images_to_display.append(np.zeros((128,128,3), dtype=np.uint8)) # Add a blank image on error
    return images_to_display


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
        return (None, None, None, "Please select all inputs.") + (None,) * (NUM_FILTERS_TO_SHOW + 1) # Return placeholders

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
    
    predicted_mask_for_class = (predicted_mask_full == target_class_idx).astype(np.float32)
    gt_mask_for_class = (ground_truth_label == target_class_idx).astype(np.float32)

    # Calculate metrics
    seg_jaccard = jaccard_index(predicted_mask_for_class, gt_mask_for_class)
    gt_pixel_count = np.sum(gt_mask_for_class)
    status = f"Segmentation Jaccard: {seg_jaccard:.3f} | GT Pixels: {int(gt_pixel_count)} | Status: {'🚨 Problematic' if seg_jaccard < POOR_PERFORMANCE_JACCARD_THRESHOLD else '✅ Good Performance'}"

    # Prepare base images for display
    original_image_display = np.uint8(255 * input_image_batch.numpy().squeeze()) # Convert to 0-255 for Gradio
    gt_mask_display = np.uint8(255 * gt_mask_for_class)
    predicted_mask_display = np.uint8(255 * predicted_mask_for_class)

    xai_image_display = None
    filter_activation_images = []

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
        filter_activation_images = visualize_filter_activations_gradio(full_model_with_attention, input_image_batch)
        # For filter activations, the primary XAI output will be a gallery, not a single image.
        # We can set xai_image_display to a blank or original image if desired, or handle it as a separate output.
        xai_image_display = np.zeros_like(original_image_display) # Blank placeholder for primary XAI image

    # Pad filter_activation_images list to match the number of outputs in the interface
    # This is crucial for Gradio's multi-output component matching
    padded_filter_images = filter_activation_images + [None] * (NUM_FILTERS_TO_SHOW * 6) # Pad sufficiently

    return original_image_display, gt_mask_display, predicted_mask_display, status, xai_image_display, *padded_filter_images


# --- Gradio Interface Layout ---
class_names_map_for_gradio = {i: f"Class {i}" for i in CLASSES_TO_ANALYZE}

# Dynamically create outputs for filter activations
filter_activation_outputs = [gr.Image(label=f"Filter Activation {i+1}", visible=False) for i in range(NUM_FILTERS_TO_SHOW * 6)] # Enough for multiple layers * filters


iface = gr.Interface(
    fn=explain_segmentation,
    inputs=[
        gr.Dropdown(choices=available_volumes, label="Select Volume"),
        gr.Slider(minimum=0, maximum=img_volume.shape[0]-1 if 'img_volume' in locals() else 10, step=1, label="Select Slice Index"),
        gr.Dropdown(choices=CLASSES_TO_ANALYZE, label="Select Target Class", value=CLASSES_TO_ANALYZE[0]),
        gr.Radio(choices=["Original/Prediction", "Grad-CAM", "Integrated Gradients", "Attention Map", "Filter Activations"], 
                 label="Choose XAI Method", value="Original/Prediction")
    ],
    outputs=[
        gr.Image(label="Original Image", type="numpy"),
        gr.Image(label="Ground Truth Mask", type="numpy"),
        gr.Image(label="Predicted Mask", type="numpy"),
        gr.Textbox(label="Segmentation Metrics & Status"),
        gr.Image(label="XAI Heatmap Overlay", type="numpy"),
        *filter_activation_outputs # Unpack the list of filter activation outputs
    ],
    title="🧠 Fetal Brain Segmentation Explainability Dashboard",
    description="Select a fetal MRI volume, slice, and target anatomical class to view the model's segmentation prediction and various XAI explanations (Grad-CAM, Integrated Gradients, Attention Maps, Filter Activations).",
    live=False, # Set to True if you want updates on every slider move, but can be slow
    allow_flagging='never', # Disable flagging for this demo
    css="footer {visibility: hidden}" # Hide Gradio footer for cleaner look
)

# JavaScript to dynamically show/hide filter activation outputs
js_code = """
function(volume_name, slice_idx, target_class_idx, xai_method, original_image_display, gt_mask_display, predicted_mask_display, status, xai_image_display, ...filter_activation_images) {
    let outputs = [original_image_display, gt_mask_display, predicted_mask_display, status, xai_image_display];
    let visibility = xai_method === 'Filter Activations';

    for (let i = 0; i < filter_activation_images.length; i++) {
        outputs.push({
            __type__: "update",
            value: filter_activation_images[i],
            visible: visibility
        });
    }
    return outputs;
}
"""

# Link JavaScript to the interface
iface.js = js_code

if __name__ == "__main__":
    iface.launch(share=True) # Set share=True to get a public link (expires after 72 hours)
