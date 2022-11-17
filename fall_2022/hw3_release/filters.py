"""
CS131 - Computer Vision: Foundations and Applications
Assignment 3
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/18/2022
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    h = Hk // 2
    w = Wk // 2

    # Loop through each pixel in image
    for i in range(Hi):
        for j in range(Wi):
            # Loop through each pixel in kernel
            for k in range(Hk):
                for l in range(Wk):
                    # Check if kernel is in bounds of image
                    height = i - (k - h)
                    width = j - (l - w)
                    if height >= 0 and height < Hi and width >= 0 and width < Wi:
                        # Apply convolution filter
                        out[i, j] += image[height, width] * kernel[k, l] 
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    out = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'constant')
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    # Pad image and flip kernel
    padded_image = zero_pad(image, Hk // 2, Wk // 2)
    kernel_flip = np.flip(kernel)
    
    # Compute weighted sum of the neighborhood at each pixel
    for i in range(Hi):
        for j in range(Wi):
            out[i][j] = np.sum(padded_image[i:i + Hk, j:j + Wk] * kernel_flip)
    
    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = conv_fast(f, np.flip(g))
    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = cross_correlation(f, g - g.mean())
    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    
    
    Hi, Wi = f.shape
    Ht, Wt = g.shape
    out = np.zeros((Hi, Wi))
    
    f = zero_pad(f, Ht // 2, Wt // 2)
    
    # Normalize kernel
    normalized_kernel = (g - np.mean(g))/np.std(g)
    
    # Compute weighted sum of the neighborhood at each pixel
    for i in range(Hi):
        for j in range(Wi):
            # Normalize patch
            patch = f[i:i + Ht, j:j + Wt]
            normalized_patch = (patch - np.mean(patch))/np.std(patch)
            # Compute weighted sum
            out[i][j] = np.sum(normalized_patch * normalized_kernel)
    
    return out
