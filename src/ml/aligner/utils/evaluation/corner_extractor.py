""" Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca """

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


import model


class GetCorners:
    def __init__(self, checkpoint_dir):
        self.model = model.ModelFactory.get_model("resnet", "document")
        # dummy_input=torch.randn(1, 3, 32, 32)
        # torch.onnx.export(
        #     self.model,  # Model to export
        #     dummy_input,  # Dummy input tensor
        #     "model_doc.onnx",  # Output file name
        #     export_params=True,  # Store the trained parameter weights inside the model file
        #     opset_version=11,  # ONNX version to export to (choose a suitable opset version)
        #     do_constant_folding=True,  # Whether to execute constant folding for optimization
        #     input_names=['input'],  # Name for the input tensor
        #     output_names=['output'],  # Name for the output tensor
        #     dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Allow variable batch size
        # )
        #

        model_data_dict = torch.load(checkpoint_dir, map_location="cpu")
        model_state_dict = self.model.state_dict()
        missing_layers_keys = set([x for x in model_state_dict.keys()]) - set(
            [x for x in model_data_dict.keys()]
        )
        missing_layers = {x: model_state_dict[x] for x in missing_layers_keys}
        model_data_dict.update(missing_layers)
        self.model.load_state_dict(model_data_dict)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

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
            image_array = np.copy(pil_image)
            pil_image = Image.fromarray(pil_image)
            test_transform = transforms.Compose(
                [transforms.Resize([32, 32]), transforms.ToTensor()]
            )
            img_temp = test_transform(pil_image)

            img_temp = img_temp.unsqueeze(0)
            if torch.cuda.is_available():
                img_temp = img_temp.cuda()

            model_prediction = self.model(img_temp).cpu().data.numpy()[0]

            model_prediction = np.array(model_prediction)
            # tl tr br bl
            x_cords = model_prediction[[0, 2, 4, 6]]
            y_cords = model_prediction[[1, 3, 5, 7]]

            doc_width = (
                self.calculate_euclidian_distance(
                    (x_cords[0], y_cords[0]), (x_cords[1], y_cords[1])
                )
                + self.calculate_euclidian_distance(
                    (x_cords[3], y_cords[3]), (x_cords[2], y_cords[2])
                )
            ) / 2
            doc_height = (
                self.calculate_euclidian_distance(
                    (x_cords[0], y_cords[0]), (x_cords[3], y_cords[3])
                )
                + self.calculate_euclidian_distance(
                    (x_cords[1], y_cords[1]), (x_cords[2], y_cords[2])
                )
            ) / 2

            # Extract the four corners of the image. Read "Region Extractor" in Section III of the paper for an explanation.

            top_left_y_lower_bound = max(
                0, (2 * y_cords[0] - (y_cords[3] + y_cords[0]) / 2)
            )
            top_left_y_upper_bound = (y_cords[3] + y_cords[0]) / 2
            top_left_x_lower_bound = max(
                0, (2 * x_cords[0] - (x_cords[1] + x_cords[0]) / 2)
            )
            top_left_x_upper_bound = (x_cords[1] + x_cords[0]) / 2

            top_right_y_lower_bound = max(
                0, (2 * y_cords[1] - (y_cords[1] + y_cords[2]) / 2)
            )
            top_right_y_upper_bound = (y_cords[1] + y_cords[2]) / 2
            top_right_x_lower_bound = (x_cords[1] + x_cords[0]) / 2
            top_right_x_upper_bound = min(
                image_array.shape[1] - 1,
                (x_cords[1] + (x_cords[1] - x_cords[0]) / 2),
            )

            bottom_right_y_lower_bound = (y_cords[1] + y_cords[2]) / 2
            bottom_right_y_upper_bound = min(
                image_array.shape[0] - 1,
                (y_cords[2] + (y_cords[2] - y_cords[1]) / 2),
            )
            bottom_right_x_lower_bound = (x_cords[2] + x_cords[3]) / 2
            bottom_right_x_upper_bound = min(
                image_array.shape[1] - 1,
                (x_cords[2] + (x_cords[2] - x_cords[3]) / 2),
            )

            bottom_left_y_lower_bound = (y_cords[0] + y_cords[3]) / 2
            bottom_left_y_upper_bound = min(
                image_array.shape[0] - 1,
                (y_cords[3] + (y_cords[3] - y_cords[0]) / 2),
            )
            bottom_left_x_lower_bound = max(
                0, (2 * x_cords[3] - (x_cords[2] + x_cords[3]) / 2)
            )
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

            top_left = image_array[
                int(
                    partitions_dictionary["top_left"][0] * image_array.shape[0]
                ) : int(
                    partitions_dictionary["top_left"][1] * image_array.shape[0]
                ),
                int(
                    partitions_dictionary["top_left"][2] * image_array.shape[1]
                ) : int(
                    partitions_dictionary["top_left"][3] * image_array.shape[1]
                ),
            ]

            top_right = image_array[
                int(
                    partitions_dictionary["top_right"][0]
                    * image_array.shape[0]
                ) : int(
                    partitions_dictionary["top_right"][1]
                    * image_array.shape[0]
                ),
                int(
                    partitions_dictionary["top_right"][2]
                    * image_array.shape[1]
                ) : int(
                    partitions_dictionary["top_right"][3]
                    * image_array.shape[1]
                ),
            ]

            bottom_right = image_array[
                int(
                    partitions_dictionary["bottom_right"][0]
                    * image_array.shape[0]
                ) : int(
                    partitions_dictionary["bottom_right"][1]
                    * image_array.shape[0]
                ),
                int(
                    partitions_dictionary["bottom_right"][2]
                    * image_array.shape[1]
                ) : int(
                    partitions_dictionary["bottom_right"][3]
                    * image_array.shape[1]
                ),
            ]

            bottom_left = image_array[
                int(
                    partitions_dictionary["bottom_left"][0]
                    * image_array.shape[0]
                ) : int(
                    partitions_dictionary["bottom_left"][1]
                    * image_array.shape[0]
                ),
                int(
                    partitions_dictionary["bottom_left"][2]
                    * image_array.shape[1]
                ) : int(
                    partitions_dictionary["bottom_left"][3]
                    * image_array.shape[1]
                ),
            ]

            # plt.imshow(top_left)
            # plt.show()
            # plt.imshow(top_right)
            # plt.show()
            # plt.imshow(bottom_right)
            # plt.show()
            # plt.imshow(bottom_left)
            # plt.show()

            x_cords = x_cords * image_array.shape[1]
            y_cords = y_cords * image_array.shape[0]

            top_left = (
                top_left,
                partitions_dictionary["top_left"][0] * image_array.shape[0],
                partitions_dictionary["top_left"][1] * image_array.shape[0],
                partitions_dictionary["top_left"][2] * image_array.shape[1],
                partitions_dictionary["top_left"][3] * image_array.shape[1],
                (x_cords[0], y_cords[0]),
            )

            top_right = (
                top_right,
                partitions_dictionary["top_right"][0] * image_array.shape[0],
                partitions_dictionary["top_right"][1] * image_array.shape[0],
                partitions_dictionary["top_right"][2] * image_array.shape[1],
                partitions_dictionary["top_right"][3] * image_array.shape[1],
                (x_cords[1], y_cords[1]),
            )

            bottom_right = (
                bottom_right,
                partitions_dictionary["bottom_right"][0]
                * image_array.shape[0],
                partitions_dictionary["bottom_right"][1]
                * image_array.shape[0],
                partitions_dictionary["bottom_right"][2]
                * image_array.shape[1],
                partitions_dictionary["bottom_right"][3]
                * image_array.shape[1],
                (x_cords[2], y_cords[2]),
            )

            bottom_left = (
                bottom_left,
                partitions_dictionary["bottom_left"][0] * image_array.shape[0],
                partitions_dictionary["bottom_left"][1] * image_array.shape[0],
                partitions_dictionary["bottom_left"][2] * image_array.shape[1],
                partitions_dictionary["bottom_left"][3] * image_array.shape[1],
                (x_cords[3], y_cords[3]),
            )

            return top_left, top_right, bottom_right, bottom_left
