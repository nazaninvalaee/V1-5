import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import tensorflow as tf
import nibabel as nib
import os
import io # For capturing Matplotlib figures

# Ensure your 'ensem_4_mod_4_no_mod' file is correctly set up as per previous instructions.
# This implies 'create_model' function is in ensem_4_mod_4_no_mod.py
from ensem_4_mod_4_no_mod import create_model
# This implies 'preprocess_slice' is in create_dataset.py
import create_dataset as cd
from create_dataset import preprocess_slice

# --- Configuration ---
# IMPORTANT: Adjust these paths to your specific environment
TRAINED_MODEL_PATH = '/content/drive/MyDrive/fetal-brain-segmentation-v1.5/checkpoints/Model.keras'
PATH_INPUT_VOLUMES = '/content/drive/MyDrive/feta_2.1/nii_files_input'
PATH_LABEL_VOLUMES = '/content/drive/MyDrive/feta_2.1/nii_files_output'
NUM_CLASSES = 8

# Define classes to analyze (e.g., exclude background if Class 0 is background)
CLASSES_TO_ANALYZE = [1, 2, 3, 4, 5, 6, 7] # Example: Analyzing classes 1 through 7

# Minimum pixel count for a class in a slice to be considered for analysis
MIN_PIXELS_FOR_ANALYSIS = 1000

# Define a threshold for "poor segmentation performance" to identify problematic samples
POOR_PERFORMANCE_JACCARD_THRESHOLD = 0.5

# Limit the number of filters to show per layer in activation visualization
NUM_FILTERS_TO_SHOW = 8

# --- Helper Functions (Jaccard and Dice) ---
def jaccard_index(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

# --- Model Loading (Cached for performance) ---
@st.cache_resource(show_spinner="Loading AI Model...")
def load_models():
    """Loads the trained Keras models."""
    full_model = create_model(num_classes=NUM_CLASSES, return_attention_map=True)
    full_model.load_weights(TRAINED_MODEL_PATH)
    
    # Grad-CAM model needs logits output for gradients
    model_for_gradcam = create_model(num_classes=NUM_CLASSES, return_attention_map=False)
    model_for_gradcam.load_weights(TRAINED_MODEL_PATH)
    
    return full_model, model_for_gradcam

full_model_with_attention, model_for_gradcam = load_models()

# --- Data Loading (Cached for performance) ---
@st.cache_data(show_spinner="Loading Volume Data...")
def load_volume_data(img_path, label_path):
    """Loads NIfTI volume data."""
    img_volume = nib.load(img_path).get_fdata()
    label_volume = nib.load(label_path).get_fdata()
    return img_volume, label_volume

@st.cache_data(show_spinner="Preparing Dataset Paths...")
def get_all_filepaths():
    """Fetches all file paths using create_dataset."""
    # Use n=-1 to load all data, or a smaller number for testing
    return cd.create_dataset(PATH_INPUT_VOLUMES, PATH_LABEL_VOLUMES, n=-1, s=0.0)[0]

all_filepaths = get_all_filepaths()

# --- XAI Implementations (adapted for Streamlit) ---

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

def visualize_filter_activations(model, input_image_batch, layer_names_to_visualize=None, num_filters_to_show=NUM_FILTERS_TO_SHOW):
    """
    Visualizes filter activations and returns a list of figures or images.
    Adapted for Streamlit to return figures directly for st.pyplot.
    """
    if layer_names_to_visualize is None:
        # Auto-detect relevant Conv2D layers.
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
            st.warning("No suitable Conv2D layers found for activation visualization by default. Please check `model.summary()` and specify `layer_names_to_visualize` manually.")
            return []

    st.write(f"Attempting to visualize activations for layers: {layer_names_to_visualize}")

    activation_models = []
    
    for name in layer_names_to_visualize:
        try:
            layer_output = model.get_layer(name).output
            activation_models.append(tf.keras.models.Model(inputs=model.inputs, outputs=layer_output))
        except ValueError:
            st.warning(f"Layer '{name}' not found. Skipping.")
            continue

    if not activation_models:
        st.error(f"Could not create activation models for any of the specified layers: {layer_names_to_visualize}.")
        return []

    figures = []
    for i, act_model in enumerate(activation_models):
        layer_name_for_title = layer_names_to_visualize[i]
        try:
            activations = act_model.predict(input_image_batch)
            if activations.ndim == 4:
                activations = activations.squeeze(axis=0)

                num_channels = activations.shape[-1]
                display_channels = min(num_channels, num_filters_to_show)

                st.write(f"Displaying {display_channels} activations from layer: **{layer_name_for_title}** (Total channels: {num_channels})")

                cols = display_channels + 1 if display_channels > 0 else 1
                rows = 1
                if cols > 6:
                    cols = 6
                    rows = (display_channels + 1 + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
                axes = axes.flatten()

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

                plt.suptitle(f'Activations for Layer: {layer_name_for_title}', fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                figures.append(fig)
            else:
                st.warning(f"Layer {layer_name_for_title} output has unexpected shape: {activations.shape}. Skipping.")
        except Exception as e:
            st.error(f"Error visualizing activations for layer {layer_name_for_title}: {e}")
    return figures

def overlay_heatmap(original_image_2d, heatmap, cmap='hot', alpha=0.5):
    """
    Overlays a heatmap on a grayscale image.
    original_image_2d: 2D numpy array (grayscale)
    heatmap: 2D numpy array (0-1 normalized)
    """
    # Ensure image is 0-1 for consistent scaling before converting to uint8
    if original_image_2d.dtype != np.float32:
        original_image_2d = original_image_2d.astype(np.float32)
        original_image_2d = (original_image_2d - original_image_2d.min()) / (original_image_2d.max() - original_image_2d.min() + 1e-10)

    # Convert grayscale image to RGB for color overlay
    img_rgb = cv2.cvtColor(np.uint8(255 * original_image_2d), cv2.COLOR_GRAY2RGB)
    
    # Apply colormap to heatmap
    norm_heatmap = heatmap
    if np.max(heatmap) > 0: # Normalize again to ensure 0-1 range for colormapping
        norm_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)
    
    cmap_obj = plt.get_cmap(cmap)
    cmap_img = (cmap_obj(norm_heatmap)[:,:,:3] * 255).astype(np.uint8) # Get RGB from RGBA and scale to 0-255

    # Blending
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, cmap_img, alpha, 0)
    return overlay

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Fetal Brain Segmentation XAI Dashboard 🧠")

st.title("🧠 Fetal Brain Segmentation Explainability Dashboard")
st.markdown("Explore model predictions and generate eXplainable AI (XAI) insights for fetal MRI scans.")

# --- Sidebar for Controls ---
st.sidebar.header("Data & Analysis Options")

if not all_filepaths:
    st.sidebar.error("No input volumes found! Please check `PATH_INPUT_VOLUMES` configuration.")
    st.stop()

# Create a mapping from display name to actual path for dropdown
volume_names_map = {os.path.basename(p[0]): p for p in all_filepaths}
selected_volume_key = st.sidebar.selectbox("Select Volume:", list(volume_names_map.keys()))

if selected_volume_key:
    selected_img_path, selected_label_path = volume_names_map[selected_volume_key]
    img_volume, label_volume = load_volume_data(selected_img_path, selected_label_path)

    available_slices = list(range(img_volume.shape[0]))
    selected_slice_idx = st.sidebar.selectbox("Select Slice Index:", available_slices)

    class_names = { # Example: You might want to map these to actual anatomical names
        1: "Class 1", 2: "Class 2", 3: "Class 3", 4: "Class 4",
        5: "Class 5", 6: "Class 6", 7: "Class 7"
    }
    selected_class_idx = st.sidebar.selectbox("Select Target Class:", CLASSES_TO_ANALYZE, format_func=lambda x: class_names.get(x, f"Class {x}"))

    # Preprocess the selected slice
    original_slice = img_volume[selected_slice_idx, :, :]
    original_label = label_volume[selected_slice_idx, :, :]
    input_image_processed, ground_truth_label = preprocess_slice(original_slice, original_label)
    input_image_batch = tf.expand_dims(tf.constant(input_image_processed, dtype=tf.float32), axis=0)

    # --- Run Model and Get Basic Predictions ---
    segmentation_output_softmax, _, averaged_attention_map_raw = full_model_with_attention(input_image_batch)
    predicted_mask_full = np.argmax(segmentation_output_softmax.numpy().squeeze(), axis=-1)
    
    predicted_mask_for_class = (predicted_mask_full == selected_class_idx).astype(np.float32)
    gt_mask_for_class = (ground_truth_label == selected_class_idx).astype(np.float32)

    seg_jaccard = jaccard_index(predicted_mask_for_class, gt_mask_for_class)
    gt_pixel_count = np.sum(gt_mask_for_class)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Segmentation Metrics")
    st.sidebar.write(f"**Segmentation Jaccard:** `{seg_jaccard:.3f}`")
    st.sidebar.write(f"**GT Pixel Count:** `{int(gt_pixel_count)}`")
    status_emoji = "🚨" if seg_jaccard < POOR_PERFORMANCE_JACCARD_THRESHOLD else "✅"
    st.sidebar.write(f"**Status:** {status_emoji} `{'Problematic' if seg_jaccard < POOR_PERFORMANCE_JACCARD_THRESHOLD else 'Good Performance'}`")

    st.sidebar.markdown("---")
    st.sidebar.subheader("XAI Method")
    xai_method = st.sidebar.radio("Choose Explanation Type:",
                                 ("Original/Prediction", "Grad-CAM", "Integrated Gradients", "Attention Map", "Filter Activations"))

    # --- Main Content Area ---
    st.header("Model Prediction & Explanation")

    if xai_method == "Original/Prediction":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(input_image_batch.numpy().squeeze(), cmap='gray', caption="Original Image", use_column_width=True)
        with col2:
            st.image(gt_mask_for_class, cmap='gray', caption=f"Ground Truth (Class {selected_class_idx})", use_column_width=True)
        with col3:
            st.image(predicted_mask_for_class, cmap='gray', caption=f"Predicted Mask (Class {selected_class_idx})", use_column_width=True)

    else:
        # Display Original, GT, Prediction in small columns, then the XAI viz
        col_img, col_gt, col_pred = st.columns(3)
        with col_img:
            st.image(input_image_batch.numpy().squeeze(), cmap='gray', caption="Original Image", use_column_width=True)
        with col_gt:
            st.image(gt_mask_for_class, cmap='gray', caption=f"Ground Truth (Class {selected_class_idx})", use_column_width=True)
        with col_pred:
            st.image(predicted_mask_for_class, cmap='gray', caption=f"Predicted Mask (Class {selected_class_idx})", use_column_width=True)

        st.markdown("---")
        st.subheader(f"XAI Visualization: {xai_method}")

        if xai_method == "Grad-CAM":
            grad_cam_heatmap = generate_grad_cam_for_class(model_for_gradcam, input_image_batch, selected_class_idx)
            st.image(overlay_heatmap(input_image_batch.numpy().squeeze(), grad_cam_heatmap, cmap='hot'),
                     caption="Grad-CAM (shows regions contributing to target class prediction)", use_column_width=True)
        
        elif xai_method == "Integrated Gradients":
            integrated_grads_heatmap = integrated_gradients(model_for_gradcam, input_image_batch, selected_class_idx)
            st.image(overlay_heatmap(input_image_batch.numpy().squeeze(), integrated_grads_heatmap, cmap='plasma'),
                     caption="Integrated Gradients (pixel importance for target class prediction)", use_column_width=True)
        
        elif xai_method == "Attention Map":
            attn_map_problem = tf.reduce_mean(averaged_attention_map_raw[0], axis=-1).numpy()
            if np.max(attn_map_problem) > 0:
                attn_map_problem_normalized = (attn_map_problem - np.min(attn_map_problem)) / (np.max(attn_map_problem) - np.min(attn_map_problem) + 1e-10)
            else:
                attn_map_problem_normalized = np.zeros_like(attn_map_problem)
            st.image(overlay_heatmap(input_image_batch.numpy().squeeze(), attn_map_problem_normalized, cmap='viridis'),
                     caption="Model's Internal Channel Attention Map", use_column_width=True)
        
        elif xai_method == "Filter Activations":
            st.write("Below are the activations of selected filters from different layers. Each subplot shows how a specific filter 'sees' the input image.")
            # Call the function, it will directly display plots using st.pyplot
            filter_figs = visualize_filter_activations(full_model_with_attention, input_image_batch)
            for fig in filter_figs:
                st.pyplot(fig) # Display each captured figure
                plt.close(fig) # Close the figure to free up memory

else:
    st.info("Please ensure your dataset paths are correctly configured and contain NIfTI files.")
