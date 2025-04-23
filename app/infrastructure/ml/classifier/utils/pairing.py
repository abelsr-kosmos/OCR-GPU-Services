import itertools
import numpy as np
from typing import Dict, Union
from scipy.optimize import linear_sum_assignment
from .filters import filter_non_outlier_costs, filter_shared_fields
from .linalg import is_index_valid, solve_affine_transformation
from scipy.linalg import lstsq
import torch


IDENTITY = np.array([[1, 0, 0], [0, 1, 0]])


def get_custom_cost(result: Dict, apply_log: bool = True) -> Union[float, None]:
    """
    Calculates the custom cost of pairing which will be used for the classification task.
    If the pairing was not succesuful the custom cost willl be None.

    Args
    ----
    result: `Dict`
        Is the output from the function `pair_fields`.

    Output
    -------
    mean_cost: float | None
        Represets the average cost of paring the bounding boxes of the inference document
        with the corresponding of the reference document. The value could be None if the
        pairing was not succesful.
    """
    cost = result["cost"]
    mean_cost = None
    if cost:
        mean_cost = cost / result["num_bboxes"]["used"]["min"] ** 2
        mean_cost *= result["num_bboxes"]["total"]["x"]
        if apply_log:
            mean_cost = np.log(mean_cost)
    return mean_cost


def pair_fields(
    doc_x: Dict,
    doc_y: Dict,
    max_cost: int = 1_000,
    remove_ol_method: str = "z-score",
    zscore_threshold: Union[int, float] = 3,
) -> Dict:
    """
    Solves the assigment problem for pairing the bounding boxes of the reference and inference document.
    This assumes that only words in both reference and inference documents could be paired.
    Notice that the pairing is not always possible, in this case the elements of the result will be left as
    None.

    Args
    ----
    docx_x: `Dict`
        Reference OCR document with keys pages, file, width, height.
    docx_y: `Dict`
        Inference OCR document with keys pages, file, width, height.
    max_cost: int
        Maximum cost assigned to bounding boxes whose words are not the same.
    remove_ol_method : str
        Method for removing pairs of every bounding box which has an outlier cost.
        - 'z-score'.
        - None
    zscore_threshold : int | float
        The number of std set for removing the outlier cost after calculating the z-score.

    Output
    ------
    result: `Dict`
        Contains the information of the solution of the assingment problem.
    """

    (bboxes_x, texts_x), (bboxes_y, texts_y) = filter_shared_fields(
        doc_x, doc_y
    )

    width_height = np.array([doc_x["width"], doc_x["height"]], dtype=np.float64)

    bboxes_x = np.array(bboxes_x, dtype=np.float64)
    bboxes_y = np.array(bboxes_y, dtype=np.float64)

    cost_matrix = np.empty((bboxes_x.shape[0], bboxes_y.shape[0]))
    affine_transforms = np.empty((bboxes_x.shape[0], bboxes_y.shape[0], 2, 3))
    # Normalize bboxes
    if texts_x:
        bboxes_x_ = bboxes_x / width_height
        bboxes_y_ = bboxes_y / width_height

        last_idx = max(bboxes_x.shape[0], bboxes_y.shape[0])
        # indices = list(itertools.product(list(range(last_idx)), repeat=2))
        indices = [
            (i, j)
            for i, j in itertools.product(range(last_idx), repeat=2)
            if j < bboxes_y.shape[0] and i < bboxes_x.shape[0]
        ]
        # for i, j in indices:
        #    if is_index_valid(cost_matrix, i, j):
        #        A = solve_affine_transformation(bboxes_x_[i], bboxes_y_[j])
        #        cost = np.linalg.norm(A.T - IDENTITY)
        #        cost_matrix[i, j] = (
        #            cost if texts_x[i] == texts_y[j] else max_cost)
        #        affine_transforms[i, j] = A.T

        # resolver todos los sistemas de ecuaciones
        a_matrices = [
            np.hstack((bboxes_x_[i], np.ones((4, 1)))) for i, j in indices
        ]
        b_vectors = [bboxes_y_[j] for i, j in indices]
        A = torch.from_numpy(np.array(a_matrices))
        b = torch.from_numpy(np.array(b_vectors))
        array_A = torch.linalg.lstsq(A, b).solution
        array_IDENTITY = np.stack([IDENTITY] * array_A.shape[0], axis=0)
        if array_A.shape != array_IDENTITY.shape:
            array_IDENTITY = array_IDENTITY.transpose(0, 2, 1)
        costos = np.linalg.norm(array_A - array_IDENTITY, axis=(1, 2))
        for c, (i, j) in zip(costos, indices):
            cost_matrix[i, j] = c if texts_x[i] == texts_y[j] else max_cost

        # Solve the assignment problem using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        num_x, num_y = cost_matrix.shape

        if not isinstance(remove_ol_method, str):
            # Order the bboxes and texts according to pairs
            bboxes_x = bboxes_x[row_ind]
            bboxes_y = bboxes_y[col_ind]

            texts_x = np.array(texts_x)[row_ind].tolist()
            texts_y = np.array(texts_y)[col_ind].tolist()

            # Remove the indices where the words doesn't concide
            inds = [i for i in range(len(texts_x)) if texts_x[i] == texts_y[i]]

            bboxes_x = bboxes_x[inds]
            bboxes_y = bboxes_y[inds]

            # Get the minimum total distance
            row_ind = row_ind[inds]
            col_ind = col_ind[inds]
            min_total_cost = cost_matrix[row_ind, col_ind].sum()

        elif remove_ol_method == "z-score":
            mask = filter_non_outlier_costs(
                cost_matrix[row_ind, col_ind],
                threshold=zscore_threshold,  # zscore_threshold tener en cuenta
            )
            row_ind = row_ind[mask]
            col_ind = col_ind[mask]

            bboxes_x = bboxes_x[row_ind]
            bboxes_y = bboxes_y[col_ind]

            texts_x = np.array(texts_x)[row_ind].tolist()
            texts_y = np.array(texts_y)[col_ind].tolist()
            min_total_cost = cost_matrix[row_ind, col_ind].sum()

        min_num_bboxes = len(row_ind)
    else:
        row_ind = None
        col_ind = None
        num_x = None
        num_y = None
        min_total_cost = None
        min_num_bboxes = None

    return dict(
        bboxes=dict(x=bboxes_x.tolist(), y=bboxes_y.tolist()),
        num_bboxes=dict(
            total=dict(x=len(doc_x["pages"][0]), y=len(doc_y["pages"][0])),
            used=dict(x=num_x, y=num_y, min=min_num_bboxes),
        ),
        cost=min_total_cost,
    )
