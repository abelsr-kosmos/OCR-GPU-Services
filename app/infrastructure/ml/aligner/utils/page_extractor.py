import cv2
import torch
import numpy as np
from PIL import Image

from app.infrastructure.ml.aligner.utils import evaluation

class PageExtractor(object):
    def __init__(self, cornerModel_path: str, documentModel_path: str):
        self.corners_extractor = evaluation.corner_extractor.GetCorners(
            documentModel_path
        )
        self.corner_refiner = evaluation.corner_refiner.corner_finder(
            cornerModel_path
        )

    def extract_corners(self, image: np.ndarray, retain_factor: float = 0.85):
        oImg = image.copy()
        self.img = image
        extracted_corners = self.corners_extractor.get(oImg, 0.286)

        # Create a more efficient refinement process
        corner_address = []
        
        # Check if we can use GPU
        use_gpu = torch.cuda.is_available()
        
        # Process corners in sequence but with optimized GPU utilization
        # Note: true parallelism would require more complex threading/multiprocessing
        for i, corner in enumerate(extracted_corners):
            corner_img = corner[0]
            # Maximum iterations reduced in corner_refiner for better performance
            refined_corner = np.array(
                self.corner_refiner.get_location(
                    corner_img, float(retain_factor)
                )
            )

            # Converting from local co-ordinate to global co-ordinates of the image
            refined_corner[0] += corner[3]
            refined_corner[1] += corner[1]

            # Final results
            corner_address.append(refined_corner)
            
            
        # Explicitly clear GPU cache after processing all corners
        if use_gpu:
            torch.cuda.empty_cache()
            
        return corner_address

    def highlight_bounding_box(
        self, image_path: str, retain_factor: float = 0.85
    ):

        corners = self.extract_corners(image_path, retain_factor)
        for a in range(0, len(corners)):
            cv2.line(
                self.img,
                tuple(corners[a % 4]),
                tuple(corners[(a + 1) % 4]),
                (255, 0, 0),
                4,
            )
        return corners, self.img

    def extract_document(self, image: np.ndarray, retain_factor: float) -> tuple:
        """
        Extract a document from an image by finding its corners and applying perspective transform.
        
        Args:
            image: np.ndarray - The input image as a numpy array
            retain_factor: float - Factor to control corner detection sensitivity
            
        Returns:
            tuple - (corners, warped_image) where corners are the detected corners and
                   warped_image is the perspective-corrected document
        """
        corners = self.extract_corners(image, retain_factor)

        (tl, tr, br, bl) = corners
        
        # Calculate width efficiently using numpy
        width_points = np.array([[br[0], br[1]], [bl[0], bl[1]], [tr[0], tr[1]], [tl[0], tl[1]]])
        widths = np.sqrt(np.sum(np.diff(width_points[[0, 1, 2, 3, 0]], axis=0)**2, axis=1))
        maxWidth = int(max(widths[0], widths[2]))  # br-bl and tr-tl
        
        # Calculate height efficiently
        height_points = np.array([[tr[0], tr[1]], [br[0], br[1]], [tl[0], tl[1]], [bl[0], bl[1]]])
        heights = np.sqrt(np.sum(np.diff(height_points[[0, 1, 2, 3, 0]], axis=0)**2, axis=1))
        maxHeight = int(max(heights[0], heights[2]))
        
        
        dst = np.array([
            [0, 0],
            [maxWidth, 0],
            [maxWidth, maxHeight],
            [0, maxHeight],
        ], dtype="float32")
        
        corners_array = np.array([tl, tr, br, bl], dtype="float32")
        
        # Perspective transform
        M = cv2.getPerspectiveTransform(corners_array, dst)
        warped = cv2.warpPerspective(self.img, M, (maxWidth, maxHeight))
        
        # Fast conversion from OpenCV BGR to PIL RGB
        if len(warped.shape) == 3 and warped.shape[2] == 3:
            # Use direct array manipulation for BGR->RGB conversion (faster than cv2.cvtColor)
            warped_rgb = warped[:, :, ::-1]
            warped_pil = Image.fromarray(warped_rgb)
        else:
            warped_pil = Image.fromarray(warped).convert("RGB")
            
        return corners, warped_pil
