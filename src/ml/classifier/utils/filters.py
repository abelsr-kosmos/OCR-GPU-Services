import re
import nltk
import string
import numpy as np
from scipy.stats import zscore
from collections import Counter
from typing import Dict, Tuple, List, Union


STOP_WORDS = nltk.corpus.stopwords.words("spanish")


def _filter_text(text: str):
    """
    Removes punctuation, double spaces and stop words from the text.

    Args
    ----
    text: str
        Raw spanish text.

    Output
    ------
    text_no_stopwords: str
        filtered text.
    """
    # remove numbers
    text = str(text)
    text_nonum = re.sub(r"\d+", "", text)
    # remove urls that don't start with http
    # text_no_urls = " ".join([word for word in text_nonum.split() if '.com' not in word])
    # remove punctuations and convert characters to lower case
    text_nopunct = "".join(
        [char.lower() for char in text_nonum if char not in string.punctuation]
    )
    # remove double spaces
    text_no_doublespace = re.sub(r"\s+", " ", text_nopunct).strip()
    # remove urls
    text_no_urls = re.sub(r"http\S+", "", text_no_doublespace)
    # remove stopwords
    text_no_stopwords = " ".join(
        [word for word in text_no_urls.split() if word not in STOP_WORDS]
    )
    return text_no_stopwords


def filter_shared_fields(
    doc_x: Dict, doc_y: Dict
) -> Tuple[Tuple[List[List], List[str]], Tuple[List[List], List[str]]]:
    """
    Filters shared the bounding boxes of both reference and inference documents. This looks fo the shared words and removes
    the bounding boxes with the corresponding texts that are not shared in both documents.

    Args
    ----
    doc_x: `Dict`
        Represents the OCR of the reference document processed.
    doc_y: `Dict`
        Represents the OCR of the inference document processed.

    Output
    ------
    outpt : Tuple
        (bboxes_x, texts_x) : Tuple[List[List], List[str]]
            bboxes_x: The reference bounding boxes filtered.
            texts_x:  The reference texts corresponding to the filtered bboxes.

        (bboxes_y, texts_y) : Tuple[List[List], List[str]]
            bboxes_y: The inference bounding boxes filtered.
            texts_y:  The inference texts corresponding to the filtered bboxes.
    """
    ## TODO: Add levenshtein ratio for filtering the words that are similar but not the same.
    inference_list = [
        _filter_text(item["Text"])
        for item in doc_y["pages"][0]
        if _filter_text(item["Text"])
    ]
    reference_list = [
        _filter_text(item["Text"])
        for item in doc_x["pages"][0]
        if _filter_text(item["Text"])
    ]
    inference_counter = Counter(inference_list)
    reference_counter = Counter(reference_list)

    common_words = inference_counter & reference_counter

    bboxes_x = [
        item["Coordinates"]
        for item in doc_x["pages"][0]
        if _filter_text(item["Text"]) in common_words
    ]
    bboxes_y = [
        item["Coordinates"]
        for item in doc_y["pages"][0]
        if _filter_text(item["Text"]) in common_words
    ]

    texts_x = [
        _filter_text(item["Text"])
        for item in doc_x["pages"][0]
        if _filter_text(item["Text"]) in common_words
    ]
    texts_y = [
        _filter_text(item["Text"])
        for item in doc_y["pages"][0]
        if _filter_text(item["Text"]) in common_words
    ]

    return (bboxes_x, texts_x), (bboxes_y, texts_y)


def filter_non_outlier_costs(
    selected_costs: np.ndarray, threshold: Union[int, float] = 2
):
    """
    Removes the outlier selected scores according to their z-score.

    Args
    ----
    selected_costs: `np.ndarray`
        The array of costs.
    threshold : int | float
        The number of std that will be consider for removing outliers.

    Output
    ------
    non_outlier_mask: `np.ndarray`
        Is an array with the same shape of selected_costs. Represents which indices will be consider after
        removing the outliers. The element at an specific index smust be preserved if the value at the
        same index in this array is True.
    """
    # Compute Z-scores to identify outliers
    z_scores = zscore(selected_costs)

    # Define a threshold for Z-scores (e.g., remove elements with |z| > thr)
    non_outlier_mask = np.abs(z_scores) <= threshold

    return non_outlier_mask
