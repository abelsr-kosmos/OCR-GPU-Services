import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from app.infrastructure.ml.aligner.utils import model


class GetCorners:
    def __init__(self, checkpoint_dir):
        self.model = model.ModelFactory.get_model("resnet", "document")
        
        # Use GPU if available for model loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model weights directly to the target device
        model_data_dict = torch.load(checkpoint_dir, map_location=device)
        model_state_dict = self.model.state_dict()
        missing_layers_keys = set([x for x in model_state_dict.keys()]) - set(
            [x for x in model_data_dict.keys()]
        )
        missing_layers = {x: model_state_dict[x] for x in missing_layers_keys}
        model_data_dict.update(missing_layers)
        self.model.load_state_dict(model_data_dict)
        
        # Move model to device
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        
        # Pre-cache transforms for efficiency
        self.test_transform = transforms.Compose(
            [transforms.Resize([32, 32]), transforms.ToTensor()]
        )

    def calculate_area(
        self, lower_bound_y, upper_bound_y, lower_bound_x, upper_bound_x
    ):
        if float(lower_bound_y) > float(upper_bound_y):
            return 0
        if float(lower_bound_x) > float(upper_bound_x):
            return 0
        return (float(upper_bound_y) - float(lower_bound_y)) * (
            float(upper_bound_x) - float(lower_bound_x)
        )

    def calculate_euclidian_distance(self, cordinate_1, cordinate_2):
        x1, y1 = cordinate_1
        x2, y2 = cordinate_2

        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get(self, pil_image, leeway):
        with torch.no_grad():
            # Check if the image is already a numpy array or a PIL Image
            if isinstance(pil_image, np.ndarray):
                image_array = pil_image
                pil_image = Image.fromarray(pil_image)
            else:
                image_array = np.array(pil_image)
                pil_image = Image.fromarray(image_array)
            
            # Check if GPU is available
            use_gpu = torch.cuda.is_available()
            
            img_temp = self.test_transform(pil_image)
            img_temp = img_temp.unsqueeze(0)

            # Run model prediction with GPU optimization
            if use_gpu:
                # Use non-blocking transfer and mixed precision
                img_temp = img_temp.cuda(non_blocking=True)
                with torch.cuda.amp.autocast():
                    model_prediction = self.model(img_temp)
                # Transfer back to CPU after computation is complete
                model_prediction = model_prediction.cpu().data.numpy()[0]
            else:
                model_prediction = self.model(img_temp).data.numpy()[0]
                
            # Process coordinates using optimized numpy operations
            # Convert to numpy array and reshape
            model_prediction = np.array(model_prediction)
            # Extract coordinates (tl tr br bl)
            x_cords = model_prediction[[0, 2, 4, 6]]
            y_cords = model_prediction[[1, 3, 5, 7]]

            # Efficient document dimension calculation using vectorized operations
            points = np.column_stack((x_cords, y_cords))
            # Calculate distances between consecutive points using vectorized operations
            shifts = np.roll(points, -1, axis=0) - points
            distances = np.sqrt(np.sum(shifts**2, axis=1))
            
            # Calculate document width and height from distances
            # Width is average of top and bottom edges
            doc_width = (distances[0] + distances[2]) / 2
            # Height is average of left and right edges
            doc_height = (distances[1] + distances[3]) / 2

            # Extract the four corners of the image
            # Use direct array operations for faster processing
            partitions_dictionary = self._calculate_partitions(
                x_cords, y_cords, doc_width, doc_height, image_array.shape, leeway
            )

            # Create region crops
            top_left, top_right, bottom_right, bottom_left = self._create_corner_regions(
                partitions_dictionary, image_array
            )
            
            # Clean up GPU memory if used
            if use_gpu:
                torch.cuda.empty_cache()
                
            return top_left, top_right, bottom_right, bottom_left
            
    def _calculate_partitions(self, x_cords, y_cords, doc_width, doc_height, image_shape, leeway):
        """
        Calculate corner partitions using vectorized operations for better performance.
        """
        # Precompute some values for efficiency
        img_height, img_width = image_shape[0], image_shape[1]
        
        # Calculate boundaries using numpy operations
        top_left_y_lower_bound = np.maximum(0, (2 * y_cords[0] - (y_cords[3] + y_cords[0]) / 2))
        top_left_y_upper_bound = (y_cords[3] + y_cords[0]) / 2
        top_left_x_lower_bound = np.maximum(0, (2 * x_cords[0] - (x_cords[1] + x_cords[0]) / 2))
        top_left_x_upper_bound = (x_cords[1] + x_cords[0]) / 2

        top_right_y_lower_bound = np.maximum(0, (2 * y_cords[1] - (y_cords[1] + y_cords[2]) / 2))
        top_right_y_upper_bound = (y_cords[1] + y_cords[2]) / 2
        top_right_x_lower_bound = (x_cords[1] + x_cords[0]) / 2
        top_right_x_upper_bound = np.minimum(1, (x_cords[1] + (x_cords[1] - x_cords[0]) / 2))

        bottom_right_y_lower_bound = (y_cords[1] + y_cords[2]) / 2
        bottom_right_y_upper_bound = np.minimum(1, (y_cords[2] + (y_cords[2] - y_cords[1]) / 2))
        bottom_right_x_lower_bound = (x_cords[2] + x_cords[3]) / 2
        bottom_right_x_upper_bound = np.minimum(1, (x_cords[2] + (x_cords[2] - x_cords[3]) / 2))

        bottom_left_y_lower_bound = (y_cords[0] + y_cords[3]) / 2
        bottom_left_y_upper_bound = np.minimum(1, (y_cords[3] + (y_cords[3] - y_cords[0]) / 2))
        bottom_left_x_lower_bound = np.maximum(0, (2 * x_cords[3] - (x_cords[2] + x_cords[3]) / 2))
        bottom_left_x_upper_bound = (x_cords[3] + x_cords[2]) / 2

        partitions_dictionary = {
            "top_left": [
                top_left_y_lower_bound,
                top_left_y_upper_bound,
                top_left_x_lower_bound,
                top_left_x_upper_bound,
                (x_cords[0], y_cords[0]),
            ],
            "top_right": [
                top_right_y_lower_bound,
                top_right_y_upper_bound,
                top_right_x_lower_bound,
                top_right_x_upper_bound,
                (x_cords[1], y_cords[1]),
            ],
            "bottom_right": [
                bottom_right_y_lower_bound,
                bottom_right_y_upper_bound,
                bottom_right_x_lower_bound,
                bottom_right_x_upper_bound,
                (x_cords[2], y_cords[2]),
            ],
            "bottom_left": [
                bottom_left_y_lower_bound,
                bottom_left_y_upper_bound,
                bottom_left_x_lower_bound,
                bottom_left_x_upper_bound,
                (x_cords[3], y_cords[3]),
            ],
        }

        # Adjust areas that are too small
        for key in partitions_dictionary.keys():
            current_bb = partitions_dictionary[key]
            if (
                self.calculate_area(
                    current_bb[0],
                    current_bb[1],
                    current_bb[2],
                    current_bb[3],
                )
                < 0.05
            ):
                y_lower_bound = current_bb[4][1] - leeway * doc_height
                y_upper_bound = current_bb[4][1] + leeway * doc_height
                x_lower_bound = current_bb[4][0] - leeway * doc_width
                x_upper_bound = current_bb[4][0] + leeway * doc_width
                partitions_dictionary[key] = [
                    y_lower_bound,
                    y_upper_bound,
                    x_lower_bound,
                    x_upper_bound,
                    (current_bb[4][0], current_bb[4][1]),
                ]

        return partitions_dictionary
        
    def _create_corner_regions(self, partitions_dictionary, image_array):
        """
        Create corner regions using the calculated partitions.
        """
        # Get the image dimensions
        img_height, img_width = image_array.shape[0], image_array.shape[1]
        
        # Create slices for more efficient array operations
        top_left = image_array[
            int(partitions_dictionary["top_left"][0] * img_height) : int(
                partitions_dictionary["top_left"][1] * img_height
            ),
            int(partitions_dictionary["top_left"][2] * img_width) : int(
                partitions_dictionary["top_left"][3] * img_width
            ),
        ]

        top_right = image_array[
            int(partitions_dictionary["top_right"][0] * img_height) : int(
                partitions_dictionary["top_right"][1] * img_height
            ),
            int(partitions_dictionary["top_right"][2] * img_width) : int(
                partitions_dictionary["top_right"][3] * img_width
            ),
        ]

        bottom_right = image_array[
            int(partitions_dictionary["bottom_right"][0] * img_height) : int(
                partitions_dictionary["bottom_right"][1] * img_height
            ),
            int(partitions_dictionary["bottom_right"][2] * img_width) : int(
                partitions_dictionary["bottom_right"][3] * img_width
            ),
        ]

        bottom_left = image_array[
            int(partitions_dictionary["bottom_left"][0] * img_height) : int(
                partitions_dictionary["bottom_left"][1] * img_height
            ),
            int(partitions_dictionary["bottom_left"][2] * img_width) : int(
                partitions_dictionary["bottom_left"][3] * img_width
            ),
        ]

        # Convert coordinates to absolute pixel values
        x_cords = x_cords = partitions_dictionary["top_left"][4][0] * img_width
        y_cords = partitions_dictionary["top_left"][4][1] * img_height
        x_cords_tr = partitions_dictionary["top_right"][4][0] * img_width
        y_cords_tr = partitions_dictionary["top_right"][4][1] * img_height
        x_cords_br = partitions_dictionary["bottom_right"][4][0] * img_width
        y_cords_br = partitions_dictionary["bottom_right"][4][1] * img_height
        x_cords_bl = partitions_dictionary["bottom_left"][4][0] * img_width
        y_cords_bl = partitions_dictionary["bottom_left"][4][1] * img_height

        # Create final tuples with all the required metadata
        top_left = (
            top_left,
            partitions_dictionary["top_left"][0] * img_height,
            partitions_dictionary["top_left"][1] * img_height,
            partitions_dictionary["top_left"][2] * img_width,
            partitions_dictionary["top_left"][3] * img_width,
            (x_cords, y_cords),
        )

        top_right = (
            top_right,
            partitions_dictionary["top_right"][0] * img_height,
            partitions_dictionary["top_right"][1] * img_height,
            partitions_dictionary["top_right"][2] * img_width,
            partitions_dictionary["top_right"][3] * img_width,
            (x_cords_tr, y_cords_tr),
        )

        bottom_right = (
            bottom_right,
            partitions_dictionary["bottom_right"][0] * img_height,
            partitions_dictionary["bottom_right"][1] * img_height,
            partitions_dictionary["bottom_right"][2] * img_width,
            partitions_dictionary["bottom_right"][3] * img_width,
            (x_cords_br, y_cords_br),
        )

        bottom_left = (
            bottom_left,
            partitions_dictionary["bottom_left"][0] * img_height,
            partitions_dictionary["bottom_left"][1] * img_height,
            partitions_dictionary["bottom_left"][2] * img_width,
            partitions_dictionary["bottom_left"][3] * img_width,
            (x_cords_bl, y_cords_bl),
        )

        return top_left, top_right, bottom_right, bottom_left
