from typing import Tuple

import numpy as np


def camera_from_world_transform(d: float = 1.0) -> np.ndarray:
    """Define a transformation matrix in homogeneous coordinates that
    transforms coordinates from world space to camera space, according
    to the coordinate systems in Question 1.


    Args:
        d (float, optional): Total distance of displacement between world and camera
            origins. Will always be greater than or equal to zero. Defaults to 1.0.

    Returns:
        T (np.ndarray): Left-hand transformation matrix, such that c = Tw
            for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
    """
    theta = 135 * np.pi / 180
    
    T = np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), d],
        [0, 0, 0, 1],
        ])
    
    assert T.shape == (4, 4)
    return T


def apply_transform(T: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray]:
    """Apply a transformation matrix to a set of points.

    Hint: You'll want to first convert all of the points to homogeneous coordinates.
    Each point in the (3,N) shape edges is a length 3 vector for x, y, and z, so
    appending a 1 after z to each point will make this homogeneous coordinates.

    You shouldn't need any loops for this function.

    Args:
        T (np.ndarray):
            Left-hand transformation matrix, such that c = Tw
                for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
        points (np.ndarray):
            Shape = (3,N) where 3 means 3D and N is the number of points to transform.

    Returns:
        points_transformed (np.ndarray):
            Transformed points.
            Shape = (3,N) where 3 means 3D and N is the number of points.
    """
    N = points.shape[1]
    assert points.shape == (3, N)
    
    # convert all points to homogenous coordinates and apply transformation matrix
    points_homo = np.append(points, np.ones((1, N)), axis=0)
    points_t_homo = np.dot(T, points_homo)
       
    # convert back to hetergenous coordinates
    vec = points_t_homo[-1, :]
    points_transformed = points_t_homo / np.transpose(vec[:, None])
    points_transformed = points_t_homo[:-1, :]
    
    assert points_transformed.shape == (3, N)
    return points_transformed


def get_det(a, b):
    """Find the determinent of two points.
    
    Args:
        a: (np.ndarray): First point; shape `(2,)`.
        b: (np.ndarray): First point; shape `(2,)`.
    
    Returns: scalar determinent
    
    """
    return a[0] * b[1] - a[1] * b[0]
    

def intersection_from_lines(
    a_0: np.ndarray, a_1: np.ndarray, b_0: np.ndarray, b_1: np.ndarray
) -> np.ndarray:
    """Find the intersection of two lines (infinite length), each defined by a
    pair of points.

    Args:
        a_0 (np.ndarray): First point of first line; shape `(2,)`.
        a_1 (np.ndarray): Second point of first line; shape `(2,)`.
        b_0 (np.ndarray): First point of second line; shape `(2,)`.
        b_1 (np.ndarray): Second point of second line; shape `(2,)`.

    Returns:
        np.ndarray: the intersection of the two lines definied by (a0, a1)
                    and (b0, b1).
    """
    # Validate inputs
    assert a_0.shape == a_1.shape == b_0.shape == b_1.shape == (2,)
    assert a_0.dtype == a_1.dtype == b_0.dtype == b_1.dtype == np.float

    # Intersection point between lines
    out = np.zeros(2)
    
    # Calculate x and y distances between points
    x_diffs = [a_0[0] - a_1[0], b_0[0] - b_1[0]]
    y_diffs = [a_0[1] - a_1[1], b_0[1] - b_1[1]]

    # Get determinent and find point of intersection
    det = get_det(x_diffs, y_diffs)

    d = (get_det(a_0, a_1), get_det(b_0, b_1))
    x = get_det(d, x_diffs) / det
    y = get_det(d, y_diffs) / det
    
    out[0] = x
    out[1] = y
    
    assert out.shape == (2,)
    assert out.dtype == np.float

    return out


def optical_center_from_vanishing_points(
    A: np.ndarray, B: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """Compute the optical center of our camera intrinsics from three vanishing
    points corresponding to mutually orthogonal directions.

    Hints:
    - Your `intersection_from_lines()` implementation might be helpful here.
    - It might be worth reviewing vector projection with dot products.

    Args:
        a (np.ndarray): Vanishing point in image space; shape `(2,)`.
        b (np.ndarray): Vanishing point in image space; shape `(2,)`.
        c (np.ndarray): Vanishing point in image space; shape `(2,)`.

    Returns:
        np.ndarray: Optical center; shape `(2,)`.
    """
    assert A.shape == B.shape == C.shape == (2,), "Wrong shape!"
    
    optical_center = np.zeros(2)
    
    BC_norm = np.sqrt(sum((C-B)**2))
    proj_BA_BC = B + (np.dot((A-B), (C-B))/pow(BC_norm, 2)) * (C-B)
    
    AC_norm = np.sqrt(sum((C-A)**2))
    proj_AB_AC = A + (np.dot((B-A), (C-A))/pow(AC_norm, 2)) * (C-A)
    
    res = intersection_from_lines(A, proj_BA_BC, B, proj_AB_AC)

    optical_center = res
    assert optical_center.shape == (2,)
    return optical_center


def focal_length_from_two_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, optical_center: np.ndarray
) -> np.ndarray:
    """Compute focal length of camera, from two vanishing points and the
    calibrated optical center.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        optical_center (np.ndarray): Calibrated optical center; shape `(2,)`.

    Returns:
        float: Calibrated focal length.
    """
    assert v0.shape == v1.shape == optical_center.shape == (2,), "Wrong shape!"

    exp = np.dot((v0 - optical_center), (v1 - optical_center))
    return pow(-exp, 1/2)


def physical_focal_length_from_calibration(
    f: float, sensor_diagonal_mm: float, image_diagonal_pixels: float
) -> float:
    """Compute the physical focal length of our camera, in millimeters.

    Args:
        f (float): Calibrated focal length, using pixel units.
        sensor_diagonal_mm (float): Length across the diagonal of our camera
            sensor, in millimeters.
        image_diagonal_pixels (float): Length across the diagonal of the
            calibration image, in pixels.

    Returns:
        float: Calibrated focal length, in millimeters.
    """
    f_mm = f * sensor_diagonal_mm / image_diagonal_pixels


    return f_mm
