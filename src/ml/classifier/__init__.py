from .model import OCRClassifier

# Default configuration for the classifier
config = {
    "zscore_threshold": 2,
    "cost_threshold": 2,
    "max_cost": 1000,
    "apply_log2cost": True,
    "n_jobs": -1
}
