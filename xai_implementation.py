import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os # Ensure this import is present
from sklearn.model_selection import train_test_split
import ensem_4_mod_4_no_mod
import create_dataset as cd
from tensorflow.keras import backend as K
import gc

# --- Initializing Configuration ---
print("--- Initializing Configuration ---")

path1 = '/content/drive/MyDrive/feta_2.1/nii_files_input'
path2 = '/content/drive/MyDrive/feta_2.1/nii_files_output'
# Assuming model_path is '/content/drive/MyDrive/fetal-brain-attencertain/checkpoints/Model.keras'
# based on the training script's output
model_path = '/content/drive/MyDrive/fetal-brain-segmentation-v1.5/checkpoints/Model.keras'
NUM_SLICES_PER_VOLUME = 50

brain_parts_names = [
    'Background',
    'Intracranial space',
    'Gray matter',
    'White matter',
    'Ventricles',
    'Cerebellum',
    'Deep gray matter',
    'Brainstem and spinal cord'
]

# --- 1. Load the Model ---
print("\n--- Loading Model ---")
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully from Model.keras.")
except Exception as e:
    print(f"Error loading model directly from {model_path}: {e}")
    print("Attempting to load architecture from ensem_4_mod_4_no_mod and weights...")

    try:
        from ensem_4_mod_4_no_mod import create_model
        model = create_model(dropout_rate=0.2)

        weights_path = '/content/drive/MyDrive/fetal-brain-segmentation-v1.5/checkpoints/trained_ensemble_weights_with_dropout.weights.h5'
        model.load_weights(weights_path)
        print("Model loaded using architecture definition and weights.")

    except Exception as e:
        print(f"Failed to load model architecture and weights: {e}")
        exit()

# Automatically find a target layer (as defined in the original script)
target_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D) and 'conv' in layer.name:
        if layer.output.shape[-1] == len(brain_parts_names):
            continue
        target_layer_name = layer.name
        break

if target_layer_name:
    print(f"Using '{target_layer_name}' as the Grad-CAM target layer.")
else:
    target_layer_name = 'gradcam_target_conv' # fallback
    try:
        _ = model.get_layer(target_layer_name)
        print(f"Using default target_layer_name: '{target_layer_name}'")
    except ValueError:
        print(f"Error: Layer '{target_layer_name}' not found.")
        exit()


# --- Grad-CAM Generation Function (same as original) ---
def generate_grad_cam(input_image_batch, model, target_layer_name, target_class_idx):
    """
    Generates a Grad-CAM heatmap for a given input batch and target class index.
    """
    input_image_batch = tf.cast(input_image_batch, tf.float32)

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(input_image_batch)
        # Note: We index the predictions for the specific sample (batch size 1)
        target_output = predictions[0, :, :, target_class_idx]
        target_output_sum = tf.reduce_sum(target_output)

    # Compute gradients of the target class output with respect to the conv layer output
    grads = tape.gradient(target_output_sum, conv_output)

    # Global average pooling of gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply the pooled gradients by the feature map activations
    heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to the heatmap (only positive activations contribute)
    heatmap = tf.maximum(heatmap, 0)

    # Normalize the heatmap to 0-1 range
    heatmap_norm = heatmap / tf.reduce_max(heatmap + 1e-6)
    heatmap_np = heatmap_norm.numpy()

    # Resize heatmap to match original image dimensions (256x256)
    original_h, original_w = input_image_batch.shape[1:3]
    if heatmap_np.shape != (original_h, original_w):
        heatmap_resized = cv2.resize(heatmap_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    else:
        heatmap_resized = heatmap_np

    return heatmap_resized

# --- Dataset Preparation (assuming create_dataset.py is accessible) ---
# We use the updated create_dataset.py with augmentation
all_filepaths, _ = cd.create_dataset(path1, path2, n=40, s=0)
_, test_filepaths = train_test_split(all_filepaths, test_size=0.1, random_state=38)

test_dataset = cd.create_tf_dataset(
    test_filepaths,
    batch_size=1,
    is_training=False,
    slices_per_volume=NUM_SLICES_PER_VOLUME
)

# --- Quantitative XAI Metrics ---

def calculate_jaccard_similarity(mask1, mask2, smooth=1e-6):
    """
    Calculates the Jaccard Index (IoU) between two binary masks.

    Args:
        mask1 (np.ndarray): Binary mask 1.
        mask2 (np.ndarray): Binary mask 2.

    Returns:
        float: Jaccard Index.
    """
    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    return (intersection + smooth) / (union + smooth)

# --- Visualization Function (Modified to include metrics) ---
# Added sample_id parameter to generate unique filenames
def plot_grad_cam_results(original_image, true_mask, predicted_mask, heatmap, brain_part_name, jaccard_score, sample_id):

    # Create the figure and axes
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original Image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Ground Truth (overlay with transparency)
    axes[1].imshow(original_image, cmap='gray')
    axes[1].imshow(true_mask, cmap='jet', alpha=0.5 * (true_mask != 0))
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    # Predicted Segmentation (overlay with transparency)
    axes[2].imshow(original_image, cmap='gray')
    axes[2].imshow(predicted_mask, cmap='jet', alpha=0.5 * (predicted_mask != 0))
    axes[2].set_title('Predicted Segmentation')
    axes[2].axis('off')

    # Grad-CAM Heatmap (overlay with transparency)
    im3 = axes[3].imshow(original_image, cmap='gray')
    im4 = axes[3].imshow(heatmap, cmap='jet', alpha=0.6)

    # Add title with the Jaccard score for the heatmap alignment
    axes[3].set_title(f'Grad-CAM for {brain_part_name}\nHeatmap-GT Jaccard: {jaccard_score:.4f}')
    axes[3].axis('off')

    plt.tight_layout()

    # --- ADDED: Code to save the figure ---
    output_dir = 'grad_cam_outputs' # Directory to save images
    os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist

    # Create a unique filename
    # Sanitize brain_part_name for filename use
    safe_brain_part_name = brain_part_name.replace(' ', '_').replace('/', '_')
    file_name = f"grad_cam_sample_{sample_id}_class_{safe_brain_part_name}.png"
    save_path = os.path.join(output_dir, file_name)

    plt.savefig(save_path)
    print(f"Saved Grad-CAM plot to: {save_path}")

    # --- END ADDED CODE ---

    # Close the figure to free up memory, important when generating many plots
    plt.close(fig)

# --- Main Visualization Loop (Modified to calculate and pass Jaccard) ---
print("\n--- Generating Grad-CAM Visualizations and Quantitative Analysis ---")

NUM_SAMPLES_TO_VISUALIZE = 5
MAX_ITERATIONS = 300
found_samples = 0
iteration_count = 0
test_dataset_iterator = iter(test_dataset)

while found_samples < NUM_SAMPLES_TO_VISUALIZE and iteration_count < MAX_ITERATIONS:
    try:
        input_batch, true_mask_batch = next(test_dataset_iterator)
        true_mask = true_mask_batch[0].numpy()

        if not np.any(true_mask != 0):
            iteration_count += 1
            continue  # Skip empty masks

        input_image = input_batch[0, :, :, 0].numpy()
        prediction_probs = model.predict(input_batch, verbose=0)
        predicted_mask = np.argmax(prediction_probs, axis=-1).squeeze()

        # Determine class for Grad-CAM
        target_class_idx = None

        # Prefer the first non-zero class in the true mask for relevance
        unique_true_classes = np.unique(true_mask)
        for cls in unique_true_classes:
            if cls != 0:
                target_class_idx = int(cls)
                break

        # If no target class found in ground truth (shouldn't happen if we passed the check above, but as a fallback)
        if target_class_idx is None:
            unique_predicted_classes = np.unique(predicted_mask)
            for cls in unique_predicted_classes:
                if cls != 0:
                    target_class_idx = int(cls)
                    break

        if target_class_idx is None:
            target_class_idx = 2  # fallback to Gray Matter

        heatmap = generate_grad_cam(
            input_image_batch=input_batch,
            model=model,
            target_layer_name=target_layer_name,
            target_class_idx=target_class_idx
        )

        # --- Quantitative XAI Analysis ---
        # 1. Create binary mask for the target class from the true mask
        target_true_mask = (true_mask == target_class_idx).astype(np.float32)

        # 2. Threshold the heatmap to create a binary mask of "activated regions"
        # We can use a simple threshold (e.g., 0.5) or a percentile threshold.
        # A threshold of 0.5 is a common starting point for binary interpretation of Grad-CAM.
        heatmap_mask = (heatmap > 0.5).astype(np.float32)

        # 3. Calculate Jaccard similarity between the heatmap mask and the target true mask
        heatmap_jaccard = calculate_jaccard_similarity(heatmap_mask, target_true_mask)
        # ---------------------------------

        print(f"Sample {found_samples + 1}: Grad-CAM for {brain_parts_names[target_class_idx]}")

        plot_grad_cam_results(
            input_image,
            true_mask,
            predicted_mask,
            heatmap,
            brain_parts_names[target_class_idx],
            heatmap_jaccard,
            found_samples + 1 # Pass the current sample ID for unique filenames
        )

        found_samples += 1
        iteration_count += 1

    except StopIteration:
        print("Reached end of dataset.")
        break
    except Exception as e:
        print(f"An error occurred during visualization iteration {iteration_count}: {e}")
        iteration_count += 1
        continue

if found_samples < NUM_SAMPLES_TO_VISUALIZE:
    print(f"Only found {found_samples} meaningful samples for visualization.")
