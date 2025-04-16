import numpy as np
from PIL import Image
from pdf2image import convert_from_path


def is_index_valid(matrix: np.ndarray, row: int, col: int) -> bool:
    """
    Checks if the position (row, col) exits in the array matrix.

    Args
    ----
    matrix: `np.ndarray`
        Array with shape (n, m).
    row: int
        Possible row index.
    col:
        Possible column index

    Output
    ------
    Boolean that determines the existence of the (row, col) element in the array.
    """
    # Check if row index is within the valid range
    if row < 0 or row >= matrix.shape[0]:
        return False
    # Check if column index is within the valid range
    if col < 0 or col >= matrix.shape[1]:
        return False
    return True


def solve_affine_transformation(
    bbox_y: np.ndarray, bbox_x: np.ndarray
) -> np.ndarray:
    """
    Finds the affine transformation between two array of bounding boxes.

    Args
    ----
    bbox_y: `np.ndarray`
        Inference bounding box.
    bbox_x:  `np.ndarray`
        Reference bounding box.

    Output
    ------
    A: `np.ndarray`
        Affine transformation.
    """
    X_prime = bbox_y
    M = np.hstack((bbox_x, np.ones((4, 1))))
    A = np.linalg.lstsq(M, X_prime, rcond=None)[0]
    return A