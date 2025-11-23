import numpy as np
import cv2 as cv


def reduce_2d(data1, data2, axis):
    """
    Removes black (all-zero, or constant-value) slices from the beginning and end
    along a given axis (0, 1, or 2).

    Args:
        data1 (np.ndarray): First 3D volume (Image).
        data2 (np.ndarray): Second 3D volume (Label).
        axis (int): Axis along which to remove black slices.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cleaned 3D volumes.
    """
    # Sum the slice content along all axes EXCEPT the target axis
    non_empty_indices = np.sum(data1, axis=tuple(i for i in range(3) if i != axis))
    
    # Find the indices where the sum is greater than zero (i.e., not a black slice)
    non_zero_indices = np.where(non_empty_indices > 0)[0]
    
    if len(non_zero_indices) == 0:
        # If the whole volume is empty, return empty arrays or original data
        return data1, data2 
        
    start_trim = non_zero_indices[0]
    end_trim = non_zero_indices[-1] + 1

    # Prepare slicing tuple for np.take (a safer way to slice along an arbitrary axis)
    # The slicing tuple should be (slice(None), slice(None), slice(None))
    slicing_tuple = [slice(None)] * 3 
    
    # Insert the trimming indices for the target axis
    slicing_tuple[axis] = slice(start_trim, end_trim)
    slicing_tuple = tuple(slicing_tuple)
    
    # Apply the slice to both arrays
    data1_cleaned = data1[slicing_tuple]
    data2_cleaned = data2[slicing_tuple]

    return data1_cleaned, data2_cleaned


def flip(d1_img, d2_label, flip_code):
    """
    Flips a 2D image slice and its corresponding label.

    Args:
        d1_img (np.ndarray): Image of shape (H, W, 1).
        d2_label (np.ndarray): Label of shape (H, W).
        flip_code (int): 0 = vertical, 1 = horizontal, -1 = both.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Flipped image and label.
    """
    flipped_img = np.expand_dims(cv.flip(d1_img.squeeze(), flip_code), axis=-1)
    flipped_label = cv.flip(d2_label, flip_code)
    return flipped_img, flipped_label


def blur(x_img, apply_blur=True):
    """
    Applies random Gaussian blur to simulate motion blur.

    Args:
        x_img (np.ndarray): Image of shape (H, W, 1).
        apply_blur (bool): Whether to apply the blur.

    Returns:
        np.ndarray: Blurred image of shape (H, W, 1).
    """
    if not apply_blur:
        return x_img

    f = np.random.randint(3)
    img_2d = x_img.squeeze()

    if f == 0:
        blurred = cv.GaussianBlur(img_2d, (11, 11), 0)
    elif f == 1:
        blurred = cv.GaussianBlur(img_2d, (15, 1), 0)
    else:
        blurred = cv.GaussianBlur(img_2d, (1, 15), 0)

    return np.expand_dims(blurred, axis=-1)


def detect_edges(label):
    """
    Applies Canny edge detection to a label mask.

    Args:
        label (np.ndarray): Label of shape (H, W), uint8 type.

    Returns:
        np.ndarray: Edge map (binary mask).
    """
    return cv.Canny(label.astype(np.uint8), threshold1=100, threshold2=200)
