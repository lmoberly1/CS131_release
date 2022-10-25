"""
CS131 - Computer Vision: Foundations and Applications
Assignment 2
Author: John Nguyen (nguyenjd@stanford.edu)
Date created: 10/2022
Last modified: 10/12/2022
Python Version: 3.5+
"""

import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad, get_output_space, warp_image


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve,
        which is already imported above. If you use convolve(), remember to
        specify zero-padding to match our equations, for example:

            out_image = convolve(in_image, kernel, mode='constant', cval=0)

        You can also use for nested loops compute M and the subsequent Harris
        corner response for each output pixel, intead of using convolve().
        Your implementation of conv_fast or conv_nested in HW1 may be a
        useful reference!

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    # 1. Compute x and y derivatives (I_x, I_y) of an image
    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    ### YOUR CODE HERE
    # 2. Compute products of derivatives (I_x^2, I_y^2, I_xy) at each pixel
    Ix2 = dx ** 2
    Iy2 = dy ** 2
    Ixy = dy * dx

    # 3. Compute matrix M at each pixel
    Sx2 = convolve(Ix2, window, mode='constant', cval=0)
    Sy2 = convolve(Iy2, window, mode='constant', cval=0)
    Sxy = convolve(Ixy, window, mode='constant', cval=0)

    # 4. Compute corner response R=Det(M) - k*(Trace(M)^2) at each pixel
    det = (Sx2 * Sy2) - (Sxy ** 2)
    trace = Sx2 + Sy2
    response = det - k * (trace ** 2)
    ### END YOUR CODE

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.

    The normalization will make the descriptor more robust to change
    in lighting condition.

    Hint:
        In this case of normalization, if a denominator is zero, divide by 1 instead.

    Args:
        patch: grayscale image patch of shape (H, W)

    Returns:
        feature: 1D array of shape (H * W)
    """
    feature = []
    ### YOUR CODE HERE
    if patch.std() == 0:
        feature = (patch - patch.mean()).flatten()
    else:
        #feature = ((patch - patch.mean()) / np.maximum(patch.std(), 1)).flatten()
        feature = ((patch-patch.mean())/patch.std()).flatten()
    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint

    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be STRICTLY SMALLER
    than the threshold (NOT equal to). Return the matches as pairs of vector indices.

    Hint:
        The Numpy functions np.sort, np.argmin, np.asarray might be useful

        The Scipy function cdist calculates Euclidean distance between all
        pairs of inputs
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """
    matches = []

    M = desc1.shape[0]
    dists = cdist(desc1, desc2)
 
    for i in range(len(dists)):
        row = dists[i]
        A, B = np.argpartition(row, 1)[0:2]
        if row[A] < (row[B] * threshold):
            new = np.array([i, A])
            matches.append(new)
            
    out = np.array(matches)
    return out
    


def fit_affine_matrix(p1, p2):
    """
    Fit affine matrix such that p2 * H = p1. First, pad the descriptor vectors
    with a 1 using pad() to convert to homogeneous coordinates, then return
    the least squares fit affine matrix in homogeneous coordinates.

    Hint:
        You can use np.linalg.lstsq function to solve the problem.

        Explicitly specify np.linalg.lstsq's new default parameter rcond=None
        to suppress deprecation warnings, and match the autograder.

    Args:
        p1: an array of shape (M, P) holding descriptors of size P about M keypoints
        p2: an array of shape (M, P) holding descriptors of size P about M keypoints

    Return:
        H: a matrix of shape (P+1, P+1) that transforms p2 to p1 in homogeneous
        coordinates
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)
    
    H, resid, rank, s = np.linalg.lstsq(p2, p1, rcond=None)

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation:

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers via Euclidean distance
        4. Keep the largest set of inliers (use >, i.e. break ties by whichever set is seen first)
        5. Re-compute least-squares estimate on all of the inliers

    Update max_inliers as a boolean array where True represents the keypoint
    at this index is an inlier, while False represents that it is not an inlier.

    Hint:
        You can use np.linalg.lstsq function to solve the problem.

        Explicitly specify np.linalg.lstsq's new default parameter rcond=None
        to suppress deprecation warnings, and match the autograder.

        You can compute elementwise boolean operations between two numpy arrays,
        and use boolean arrays to select array elements by index:
        https://numpy.org/doc/stable/reference/arrays.indexing.html#boolean-array-indexing

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()

    N = matches.shape[0]
    n_samples = int(N * 0.2)
    
    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N, dtype=bool)
    n_inliers = 0
    
    # RANSAC iteration start

    # Note: while there're many ways to do random sampling, we use
    # `np.random.shuffle()` followed by slicing out the first `n_samples`
    # matches here in order to align with the auto-grader.
    # Sample with this code:
    
    for i in range(n_iters):
        # 1. Select random set of matches
        np.random.shuffle(matches)
        samples = matches[:n_samples]
        sample1 = keypoints1[samples[:,0]]
        sample2 = keypoints2[samples[:,1]]
            
        # 2. Calculate affine transformation matrix
        # H = fit_affine_matrix(sample1, sample2)
        p1 = pad(sample1)
        p2 = pad(sample2)
        H = np.linalg.lstsq(p2, p1, rcond=None)[0]

        # 3. Find inliers using given threshold
        inliers_mask = np.linalg.norm(np.dot(matched2, H) - matched1, axis=1) < threshold
        inliers_count = np.count_nonzero(inliers_mask)
        if inliers_count > n_inliers:
            max_inliers = inliers_mask
            n_inliers = inliers_count
        
    # Recompute matrix with all inliers
    H = np.linalg.lstsq(matched2[max_inliers], matched1[max_inliers], rcond=None)[0]
    # H[:,2] = np.array([0, 0, 1])
    return H, orig_matches[max_inliers]


def linear_blend(img1_warped, img2_warped):
    """
    Linearly blend img1_warped and img2_warped by following the steps:

    1. Define left and right margins (already done for you)
    2. Define a weight matrices for img1_warped and img2_warped
        np.linspace and np.tile functions will be useful
    3. Apply the weight matrices to their corresponding images
    4. Combine the images

    Args:
        img1_warped: Refernce image warped into output space
        img2_warped: Transformed image warped into output space

    Returns:
        merged: Merged image in output space
    """
    out_H, out_W = img1_warped.shape # Height and width of output space
    img1_mask = (img1_warped != 0)  # Mask == 1 inside the image
    img2_mask = (img2_warped != 0)  # Mask == 1 inside the image

    # Find column of middle row where warped image 1 ends
    # This is where to end weight mask for warped image 1
    right_margin = out_W - np.argmax(np.fliplr(img1_mask)[out_H//2, :].reshape(1, out_W), 1)[0]

    # Find column of middle row where warped image 2 starts
    # This is where to start weight mask for warped image 2
    left_margin = np.argmax(img2_mask[out_H//2, :].reshape(1, out_W), 1)[0]
    
    # Create weight matrices
    weights1 = np.ones((out_H, out_W))
    weights2 = np.ones((out_H, out_W))
    
    if left_margin < right_margin:
        margin_size = right_margin - left_margin
        # Create gradient for margin
        gradient = np.linspace(1, 0, num = margin_size)
        # Insert gradient in weight matrices
        weights1[:, left_margin:right_margin] = gradient
        weights2[:, left_margin:right_margin] = np.flip(gradient)
        # Zero out output space not relevant to image
        weights1[:, right_margin:out_W] = np.zeros((out_H, out_W - right_margin))
        weights2[:, 0:left_margin] = np.zeros((out_H, left_margin))

  
    # Apply weight matrices to images
    img1_new = img1_warped * weights1
    img2_new = img2_warped * weights2
    merged = img1_new + img2_new 
       
    return merged


def stitch_multiple_images(imgs, desc_func=simple_descriptor, patch_size=5):
    """
    Stitch an ordered chain of images together.

    Args:
        imgs: List of length m containing the ordered chain of m images
        desc_func: Function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: Size of square patch at each keypoint

    Returns:
        panorama: Final panorma image in coordinate frame of reference image
    """
    # Detect keypoints in each image
    keypoints = []  # keypoints[i] corresponds to imgs[i]
    for img in imgs:
        kypnts = corner_peaks(harris_corners(img, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
        keypoints.append(kypnts)
    # Describe keypoints
    descriptors = []  # descriptors[i] corresponds to keypoints[i]
    for i, kypnts in enumerate(keypoints):
        desc = describe_keypoints(imgs[i], kypnts,
                                  desc_func=desc_func,
                                  patch_size=patch_size)
        descriptors.append(desc)
    # Match keypoints in neighboring images
    matches = []  # matches[i] corresponds to matches between
                  # descriptors[i] and descriptors[i+1]
    for i in range(len(imgs)-1):
        mtchs = match_descriptors(descriptors[i], descriptors[i+1], 0.7)
        matches.append(mtchs)

    H_list = [np.eye(3,3)] 
    num_images = len(imgs)
     
    # Compute transformation matrices between images
    for i in range(num_images-1):
        H, new_matches = ransac(keypoints[i], keypoints[i+1], matches[i])
        H_list.append(H)
    
    # Get transformation matrices for each image
    for i in range(1, num_images):
        H_list[i] = np.dot(H_list[i], H_list[i - 1])
    
    # FROM NOTEBOOK: Get output space
    output_shape, offset = get_output_space(imgs[0], imgs[1:], H_list[1:])
    # FROM NOTEBOOK: Warp images into output sapce
    warped_imgs = []
    img_masks = []
    for i in range(num_images):
        img_warped = warp_image(imgs[i], H_list[i], output_shape, offset)
        img_mask = (img_warped != -1) # Mask == 1 inside the image
        img_masks.append(img_mask)
        img_warped[~img_mask] = 0     # Return background values to 0
        warped_imgs.append(img_warped)

    # Combine warped images
    merged = warped_imgs[0]
    overlap = img_masks[0] * 1.0
    for i in range(1, num_images):
        merged += warped_imgs[i]
        overlap += img_masks[i] * 1.0
        
    panorama = merged / np.maximum(overlap, 1.0)
    
    return panorama
        

    


