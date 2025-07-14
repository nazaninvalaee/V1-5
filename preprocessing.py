import numpy as np
import cv2 as cv


def reduce_2d(data1, data2, axis):
    """
    Removes black (all-zero) slices from the beginning and end along a given axis.

    Args:
        data1 (np.ndarray): First 3D volume.
        data2 (np.ndarray): Second 3D volume (e.g., label).
        axis (int): Axis along which to remove black slices (0, 1, or 2).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cleaned 3D volumes.
    """
    i = 0
    c = 0

    while True:
        if axis == 2:
            x = data1[:, :, i].ravel()
        elif axis == 1:
            x = data1[:, i, :].ravel()
        else:  # axis == 0
            x = data1[i, :, :].ravel()

        if x.max() == x.min():
            if c == 0:
                if axis == 2:
                    data1, data2 = data1[:, :, 1:], data2[:, :, 1:]
                elif axis == 1:
                    data1, data2 = data1[:, 1:, :], data2[:, 1:, :]
                else:
                    data1, data2 = data1[1:, :, :], data2[1:, :, :]
            else:
                if axis == 2:
                    data1, data2 = data1[:, :, :-1], data2[:, :, :-1]
                elif axis == 1:
                    data1, data2 = data1[:, :-1, :], data2[:, :-1, :]
                else:
                    data1, data2 = data1[:-1, :, :], data2[:-1, :, :]
        elif c == 0:
            i = -1
            c = 1
        else:
            break

    return data1, data2


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
