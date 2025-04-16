import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Any
from .utils import get_custom_cost
from scipy.optimize import linear_sum_assignment
from .utils.filters import filter_non_outlier_costs, filter_shared_fields
import logging

logger = logging.getLogger(__name__)

class OCRClassifier:
    """Document classifier based on OCR text and Hungarian algorithm for matching"""
    
    def __init__(self, zscore_threshold=2, cost_threshold=2, max_cost=1000, apply_log2cost=True, n_jobs=-1):
        """
        Initialize the OCR classifier
        
        Args:
            zscore_threshold: Threshold for outlier removal
            cost_threshold: Threshold for determining document matching
            max_cost: Maximum cost for non-matching text pairs
            apply_log2cost: Whether to apply log2 to costs
            n_jobs: Number of parallel jobs
        """
        self.zscore_threshold = zscore_threshold
        self.cost_threshold = cost_threshold
        self.max_cost = max_cost
        self.apply_log2cost = apply_log2cost
        self.n_jobs = n_jobs
        
        logger.info("OCR classifier initialized successfully")
    
    def _preprocess_text(self, text_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess text data for classification"""
        # In a real implementation, this would prepare the text for classification
        # For example, filtering, normalization, etc.
        return text_data
    
    def _calculate_similarity(self, sample: List[Dict[str, Any]], reference: List[Dict[str, Any]]) -> float:
        """Calculate similarity between sample and reference documents"""
        # In a real implementation, this would use Hungarian algorithm
        # For demo, use simple text matching
        
        sample_text = " ".join([item.get("text", "") for item in sample])
        reference_text = " ".join([item.get("text", "") for item in reference])
        
        # Simple similarity based on common words
        sample_words = set(sample_text.lower().split())
        reference_words = set(reference_text.lower().split())
        
        common_words = sample_words.intersection(reference_words)
        
        if not sample_words or not reference_words:
            return 0.0
            
        similarity = len(common_words) / max(len(sample_words), len(reference_words))
        return similarity
    
    def predict(self, sample: List[Dict[str, Any]], reference_samples: List[List[Dict[str, Any]]]) -> Tuple[int, float]:
        """
        Classify sample against reference samples
        
        Args:
            sample: List of dictionaries containing OCR results for the sample
            reference_samples: List of lists of dictionaries containing OCR results for references
            
        Returns:
            Tuple of (best_match_index, similarity_score)
        """
        try:
            if not sample or not reference_samples:
                logger.warning("Empty sample or reference samples provided")
                return -1, 0.0
                
            # Preprocess
            processed_sample = self._preprocess_text(sample)
            
            # Calculate similarity with each reference
            similarities = []
            for i, reference in enumerate(reference_samples):
                processed_reference = self._preprocess_text(reference)
                similarity = self._calculate_similarity(processed_sample, processed_reference)
                similarities.append((i, similarity))
            
            # Find best match
            if not similarities:
                return -1, 0.0
                
            best_match = max(similarities, key=lambda x: x[1])
            best_match_index, best_similarity = best_match
            
            logger.info(f"Classification completed. Best match: {best_match_index} with similarity: {best_similarity:.4f}")
            return best_match_index, best_similarity
            
        except Exception as e:
            logger.error(f"Error in document classification: {str(e)}", exc_info=True)
            return -1, 0.0

class OCRclassifier(object):
    """
    OCR classifier based on the Hungarian algorithm for solving the assignment problem.
    Uses PyTorch for GPU acceleration.
    """

    def __init__(
        self,
        zscore_threshold: Union[int, float] = 2,
        cost_threshold: Union[int, float] = 0.0,
        max_cost: Union[int, float] = 1_000,
        apply_log2cost: bool = True,
        device: str = None,
        n_jobs: int = 1,
    ) -> None:
        """
        Args
        ----
        zscore_threshold : int | float
            The number of std set for removing the outlier cost after calculating the z-score.
        cost_threshold; int | float
            Threshold setted in order to determine if a pair of documents are the same or not.
        max_cost: int | float
            Maximum cost assigned to non-matching text pairs.
        apply_log2cost: bool
            Whether to apply logarithm to the cost.
        device: str
            Device to use for PyTorch computations ('cuda', 'cpu'). If None, uses CUDA if available.
        """
        self._zscore_threshold = zscore_threshold
        self._cost_threshold = cost_threshold
        self._apply_log2cost = apply_log2cost
        self._max_cost = max_cost
        
        # Set up device for PyTorch
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
    
    def predict(
        self, sample: Dict, reference_samples: List[Dict]
    ):
        """
        Predicts the class of the inference document according to a list of reference documents.
        Uses PyTorch for GPU acceleration.

        Args
        ----------
        sample: `Dict`
            Processed inference OCR document.
        reference_samples: `List[Dict]`
            Processed reference OCR documents.

        Output
        ------
        index: int | None
            Represents the index of predicted class in reference_samples.
        cost: float | None
            Represents the cost of assignment problem of pairing the inference document
            with the predicted reference document.
        """
        kwargs = dict(
            max_cost=self._max_cost,
            zscore_threshold=self._zscore_threshold,
            device=self.device
        )
        
        # Process samples using GPU acceleration where possible
        results = []
        for x_sample in reference_samples:
            result = self._pair_fields_torch(x_sample, sample, **kwargs)
            results.append(result)
            
        # Get custom costs
        costs_enumerated = [
            (e, get_custom_cost(r, self._apply_log2cost))
            for e, r in enumerate(results)
            if r["cost"]
        ]
        
        # Filter costs based on threshold
        costs_filtered = [
            (e, c)
            for e, c in costs_enumerated
            if c and c <= self._cost_threshold
        ]

        index = None
        cost = None
        if costs_filtered:
            index, cost = min(costs_filtered, key=lambda x: x[1])
        return index, cost
        
    def _pair_fields_torch(
        self,
        doc_x: Dict,
        doc_y: Dict,
        max_cost: int = 1_000,
        remove_ol_method: str = "z-score",
        zscore_threshold: Union[int, float] = 2,
        device: torch.device = None,
    ) -> Dict:
        """
        GPU-accelerated version of pair_fields that uses PyTorch where possible.
        """
        
        
        # Set default device
        if device is None:
            device = self.device
            
        # Get shared bounding boxes and texts
        (bboxes_x, texts_x), (bboxes_y, texts_y) = filter_shared_fields(doc_x, doc_y)
        
        # Convert to arrays
        width_height = np.array([doc_x["width"], doc_x["height"]], dtype=np.float64)
        bboxes_x = np.array(bboxes_x, dtype=np.float64)
        bboxes_y = np.array(bboxes_y, dtype=np.float64)
        
        # Initialize cost matrix with max_cost
        cost_matrix = np.full((bboxes_x.shape[0], bboxes_y.shape[0]), 
                              fill_value=max_cost, dtype=np.float64)
        
        # If no texts, we can't pair
        if not texts_x:
            return dict(
                bboxes=dict(x=[], y=[]),
                num_bboxes=dict(
                    total=dict(x=len(doc_x["pages"][0]), y=len(doc_y["pages"][0])),
                    used=dict(x=None, y=None, min=None),
                ),
                cost=None,
            )
            
        # Normalize bounding boxes
        bboxes_x_ = bboxes_x / width_height
        bboxes_y_ = bboxes_y / width_height
        
        # Text matching (CPU operation as string comparison isn't optimal for GPU)
        texts_x_arr = np.array(texts_x)
        texts_y_arr = np.array(texts_y)
        match_mask = np.equal.outer(texts_x_arr, texts_y_arr)
        
        # Get matching pairs
        matching_indices = np.argwhere(match_mask)
        
        if len(matching_indices) > 0:
            # Prepare matrices for least squares
            a_matrices = []
            b_matrices = []
            for (i, j) in matching_indices:
                Aij = np.hstack((bboxes_x_[i], np.ones((4, 1))))
                Bij = bboxes_y_[j]
                a_matrices.append(Aij)
                b_matrices.append(Bij)
                
            # Convert to PyTorch tensors and move to device
            A_torch = torch.tensor(np.array(a_matrices, dtype=np.float64), device=device)
            B_torch = torch.tensor(np.array(b_matrices, dtype=np.float64), device=device)
            
            # Solve batch equation A * T = B using GPU-accelerated least squares
            array_A = torch.linalg.lstsq(A_torch, B_torch).solution
            
            # Move result back to CPU for further processing
            array_A_np = array_A.cpu().numpy()
            
            # Prepare identity matrices for comparison
            IDENTITY = np.array([[1, 0, 0], [0, 1, 0]])
            array_IDENTITY = np.tile(IDENTITY, (array_A_np.shape[0], 1, 1))
            
            # Adjust shape if needed
            if array_A_np.shape != array_IDENTITY.shape:
                array_A_np = np.transpose(array_A_np, (0, 2, 1))
                
            # Calculate costs
            costs = np.linalg.norm(array_A_np - array_IDENTITY, axis=(1, 2))
            
            # Update cost matrix
            for (cost_val, (i, j)) in zip(costs, matching_indices):
                cost_matrix[i, j] = cost_val
                
        # Hungarian algorithm (CPU operation)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        num_x, num_y = cost_matrix.shape
        
        # Filter outliers if requested
        if remove_ol_method == "z-score":
            cost_assigned = cost_matrix[row_ind, col_ind]
            mask_ol = filter_non_outlier_costs(cost_assigned, threshold=zscore_threshold)
            row_ind = row_ind[mask_ol]
            col_ind = col_ind[mask_ol]
            
        # Extract final results
        bboxes_x_final = bboxes_x[row_ind]
        bboxes_y_final = bboxes_y[col_ind]
        
        min_total_cost = cost_matrix[row_ind, col_ind].sum()
        min_num_bboxes = len(row_ind)
        
        return dict(
            bboxes=dict(x=bboxes_x_final.tolist(), y=bboxes_y_final.tolist()),
            num_bboxes=dict(
                total=dict(x=len(doc_x["pages"][0]), y=len(doc_y["pages"][0])),
                used=dict(x=num_x, y=num_y, min=min_num_bboxes),
            ),
            cost=min_total_cost,
        )