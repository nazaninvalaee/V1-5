import os
import gc
import time
import numpy as np
import nibabel as nib
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import random
from preprocessing import reduce_2d, flip, blur  

# --- Preprocess a Single 2D Slice ---
def preprocess_slice(img_slice_2d, label_slice_2d):
    """
    Resizes, normalizes, and reshapes a 2D slice and its label.
    """
    img_resized = resize(img_slice_2d, (256, 256), preserve_range=True, anti_aliasing=True)
    label_resized = resize(label_slice_2d, (256, 256), order=0, anti_aliasing=False, preserve_range=True)

    max_val = np.max(img_resized)
    img_normalized = img_resized.astype(np.float32) / max_val if max_val > 0 else img_resized.astype(np.float32)

    img_final = np.expand_dims(img_normalized, axis=-1)
    label_final = label_resized.astype(np.uint8)

    return img_final, label_final

# --- Data Augmentation Wrapper (New Function) ---
def apply_augmentations(img, label):
    """
    Applies random augmentation to a 2D slice and its label.
    """
    
    # 1. Random Flipping (Horizontal or Vertical)
    if random.random() < 0.5:
        # Randomly choose flip code: 0 (vertical), 1 (horizontal), or -1 (both)
        flip_code = random.choice([0, 1]) 
        img, label = flip(img, label, flip_code)
        
    # 2. Random Gaussian Blur
    # We apply blur with a 50% probability
    if random.random() < 0.5:
        img = blur(img, apply_blur=True)
    
    return img, label

# --- Prepare List of Input/Output Filepaths ---
def prepare_filepaths(path1, path2, n):
    # ... (function body remains the same) ...
    # [Rest of the prepare_filepaths function code]
    input_files = sorted(os.listdir(path1))
    label_files = sorted(os.listdir(path2))

    filepaths = []
    
    # Iterate through input files and match with corresponding label files
    for input_file in input_files[:n]:
        # Assuming label files have a _dseg suffix before the extension
        base_name = input_file.replace('.nii.gz', '')
        label_file = base_name + '_dseg.nii.gz'

        if label_file in label_files:
            input_path = os.path.join(path1, input_file)
            label_path = os.path.join(path2, label_file)
            filepaths.append((input_path, label_path))
        else:
            print(f"Warning: Label file {label_file} not found for {input_file}")

    return filepaths


# --- TensorFlow Data Generator ---
def tf_data_generator(filepaths_list, is_training=True, slices_per_volume=50):
    """
    Generator that loads volumes, extracts slices, preprocesses, and augments (if training).
    """
    while True:
        # Shuffle filepaths for training to ensure diverse batches
        if is_training:
            random.shuffle(filepaths_list)

        for img_path, label_path in filepaths_list:
            try:
                # Load volumes
                img_volume = nib.load(img_path).get_fdata().astype(np.float32)
                label_volume = nib.load(label_path).get_fdata().astype(np.uint8)

                # Reduce black slices (optional, based on your implementation of reduce_2d)
                # Ensure reduce_2d is compatible with the volumes loaded.
                
                # We will extract slices from all three planes (X, Y, Z)

                slices_extracted = 0

                # Iterate through axes (0, 1, 2)
                for axis in [0, 1, 2]:
                    total_axis_slices = img_volume.shape[axis]

                    # Select a subset of slices if slices_per_volume is specified, 
                    # ensuring we capture key slices and variability.
                    if slices_per_volume > 0:
                        # Select indices that are likely to contain brain tissue
                        start_idx = total_axis_slices // 4
                        end_idx = 3 * total_axis_slices // 4
                        indices = np.linspace(start_idx, end_idx - 1, slices_per_volume, dtype=int)
                    else:
                        indices = range(total_axis_slices)
                    
                    for i in indices:
                        if axis == 0:
                            img_slice = img_volume[i, :, :]
                            label_slice = label_volume[i, :, :]
                        elif axis == 1:
                            img_slice = img_volume[:, i, :]
                            label_slice = label_volume[:, i, :]
                        else:
                            img_slice = img_volume[:, :, i]
                            label_slice = label_volume[:, :, i]

                        # Preprocess the slice (resize, normalize)
                        img, label = preprocess_slice(img_slice, label_slice)
                        
                        # Apply augmentation only during training
                        if is_training:
                            img, label = apply_augmentations(img, label)

                        # Check if the slice contains brain segmentation (non-zero label)
                        if np.sum(label) > 0 or not is_training:
                             yield img, label

                        slices_extracted += 1

                # Clean up memory
                del img_volume, label_volume
                gc.collect()

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue


def create_tf_dataset(filepaths_list, batch_size, is_training, slices_per_volume):
    """
    Creates a TensorFlow dataset from the generator.
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: tf_data_generator(filepaths_list, is_training, slices_per_volume),
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256), dtype=tf.uint8)
        )
    )

    if is_training:
        # Shuffle the buffer if it's training data. 
        # Shuffling is handled by the generator itself, but we can also shuffle the buffer.
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# --- Helper function for slice counting (from original file) ---
def count_slices_in_filepaths(filepaths_list, slices_per_volume):
    """
    Counts the total number of slices that will be processed.
    """
    total_slices = 0
    for img_path, _ in filepaths_list:
        try:
            # We assume 3 anatomical planes (3 * slices_per_volume) for each volume
            if slices_per_volume and slices_per_volume > 0:
                total_slices += slices_per_volume * 3
            else:
                volume = nib.load(img_path).get_fdata()
                total_slices += sum(volume.shape)
                del volume
                gc.collect()
        except Exception as e:
            continue
    return total_slices


# --- High-Level Dataset Preparer ---
def create_dataset(path1, path2, n=40, s=0.05):
    """
    Splits dataset into train/test filepaths after verifying pairs.
    """
    all_filepaths = prepare_filepaths(path1, path2, n)

    if s > 0:
        train_fp, test_fp = train_test_split(all_filepaths, test_size=s, random_state=38)
        print(f"Dataset prepared: {len(train_fp)} volumes for training, {len(test_fp)} for testing.")
        return train_fp, test_fp
    else:
        return all_filepaths, None
