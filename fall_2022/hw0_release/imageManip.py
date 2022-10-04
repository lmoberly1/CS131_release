import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """   
    out = io.imread(image_path)
    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    end_row = start_row + num_rows
    end_col = start_col + num_cols
    out = image[start_row : end_row :, start_col : end_col :, ::]
    print('Checking Shape: ', out.shape)

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = 0.5 * np.square(image)
    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    row_scale_factor = input_rows / output_rows
    col_scale_factor = input_cols / output_cols
    for out_i in range(output_rows):
        for out_j in range(output_cols):
            input_i = int(out_i * row_scale_factor)
            input_j = int(out_j * col_scale_factor)
            output_image[out_i, out_j, :] = input_image[input_i, input_j, :]

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    rot_matrix = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    return np.dot(point, rot_matrix)
    

def is_valid_point(rows, cols, x, y):
    if (x >= 0 and x < cols and y >= 0 and y < rows):
        return True
    return False

def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)

    # 2. Calculate center of image for use as new origin
    center_i = int(input_rows / 2)
    center_j = int(input_cols / 2)
    print(center_i, center_j)
    
    # 3. Iterate over output image
    for out_i in range(input_rows):
        for out_j in range(input_cols):
            shifted_point = np.array([out_i - center_i, out_j - center_j])
            rot_point = rotate2d(shifted_point, theta)
            new_X = int(rot_point[0]) + center_i
            new_Y = int(rot_point[1]) + center_j
            # Check if valid point
            if is_valid_point(input_rows, input_cols, new_X, new_Y):
                output_image[out_i, out_j, :] = input_image[new_X, new_Y, :]            
    # 4. Return the output image
    return output_image
