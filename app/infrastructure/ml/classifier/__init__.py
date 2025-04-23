from .model import OCRclassifier

# Configuration for the OCR classifier
config = dict(
    zscore_threshold=2,
    cost_threshold=2,
    max_cost=1000,
    n_jobs=-1,
    apply_log2cost=True,
)
