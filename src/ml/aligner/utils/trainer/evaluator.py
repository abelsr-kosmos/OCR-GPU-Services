""" Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk """

import logging
from typing import Dict, List, Tuple, Union
from sys import prefix

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch.nn.functional as F


import torch
import wandb
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np


logger = logging.getLogger("iCARL")
from PIL import ImageDraw


def draw_multiple_bounding_boxes(image, bounding_boxes):
    """
    Draws multiple bounding boxes on an image.

    :param image_path: Path to the input image.
    :param bounding_boxes: List of bounding boxes, each defined as [lower_bound_y, upper_bound_y, lower_bound_x, upper_bound_x].
    :param output_path: Path to save the output image.
    """
    # Load the image
    # image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Loop through each bounding box and draw it on the image
    for bounding_box in bounding_boxes:
        lower_bound_y, upper_bound_y, lower_bound_x, upper_bound_x = (
            bounding_box
        )
        rect_coords = [
            lower_bound_x * 200,
            lower_bound_y * 200,
            upper_bound_x * 200,
            upper_bound_y * 200,
        ]
        try:
            draw.rectangle(rect_coords, outline="red", width=1)
        except:
            pass
    return image


def highlight_coordinates(image, coordinates, color, radius=1):
    """
    Highlights coordinates on an image by drawing small green circles around them.

    :param image_path: Path to the input image.
    :param coordinates: List of coordinates, each defined as (x, y).
    :param output_path: Path to save the output image.
    :param radius: Radius of the circle used to highlight the coordinates. Default is 5.
    """
    # Load the image
    # image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Loop through each coordinate and draw a small circle around it
    for x, y in coordinates:
        left_up_point = ((x * 200 - radius), (y * 200 - radius))
        right_down_point = ((x * 200 + radius), (y * 200 + radius))
        draw.ellipse(
            [left_up_point, right_down_point],
            outline=color,
            width=2,
            fill=color,
        )
    return image


class EvaluatorFactory:
    """
    This class is used to get different versions of evaluators
    """

    def __init__(self):
        pass

    @staticmethod
    def get_evaluator(
        testType="rmse", cuda=True, csv_path="experiment_result.csv"
    ):
        if testType == "rmse":
            return DocumentMseEvaluator(cuda, csv_path)
        if testType == "cross_entropy":
            return CompleteDocEvaluator(cuda)
        if testType == "rmse-corners":
            return CornerMseEvaluator(cuda)


class CornerMseEvaluator:
    """
    Evaluator class for softmax classification
    """

    def __init__(self, cuda):
        self.cuda = cuda
        self.table = wandb.Table(
            columns=["img", "path", "label", "coord", "loss"]
        )

    def cordinate_within_intervals(
        self, cordinate, x_interval, y_interval
    ) -> int:

        is_within_x = x_interval[0] <= cordinate[0] <= x_interval[1]
        is_within_y = y_interval[0] <= cordinate[1] <= y_interval[1]

        return int(is_within_x and is_within_y)

    def fill_table(self, imgs, results):
        for idx in range(len(imgs)):
            img = imgs[idx].cpu().data.numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img).resize((200, 200))
            # img=np.array(img)
            result = results[idx]

            coordinates = [result["coordinates"]]
            labels = [result["labels"]]
            path = result["path"]
            loss = result["loss"]

            # for bb_ in bb_s:
            img = highlight_coordinates(img, coordinates, "green")
            img = highlight_coordinates(img, labels, "blue")
            self.table.add_data(
                wandb.Image(np.array(img)),
                path,
                np.array(labels[0]),
                np.array(coordinates[0]),
                loss,
            )

    def evaluate(self, model, iterator, epoch, prefix, table):
        model.eval()
        lossAvg = None
        classification_results = []
        with torch.no_grad():
            for img, target, paths in tqdm(iterator):
                if self.cuda:
                    img, target = img.cuda(), target.cuda()

                response = model(Variable(img))

                loss_per_example = F.mse_loss(
                    response, Variable(target.float()), reduction="none"
                )
                loss_per_example = loss_per_example.mean(dim=1)
                loss = loss_per_example.mean()
                loss = torch.sqrt(loss)

                # model_prediction = self.model(img_temp)[0]

                model_prediction = np.array(response.cpu().data.numpy())

                classification_result = []
                for i in range(len(model_prediction)):
                    y_pred = model_prediction[i, :]
                    y_true = target.cpu().data.numpy()[i, :]
                    results = {
                        "coordinates": y_pred,
                        "path": paths[i],
                        "labels": y_true,
                        "loss": loss_per_example[i],
                    }
                    classification_result.append(results)

                # classification_result = self.evaluate_corners(x_cords, y_cords, target,paths)
                classification_results.extend(classification_result)
                if table:
                    self.fill_table(img, classification_result)

                if lossAvg is None:
                    lossAvg = loss
                else:
                    lossAvg += loss
                # logger.debug("Cur loss %s", str(loss))
        # df=pd.DataFrame(classification_results)

        # df.to_csv(r"/home/ubuntu/document_localization/Recursive-CNNs/predictions.csv")

        lossAvg /= len(iterator)

        wandb.log(
            {
                "epoch": epoch,
                prefix + "eval_loss": lossAvg,
                # prefix+"accuracy": accuracy,
            }
        )
        # logger.info("Avg Val Loss %s", str((lossAvg).cpu().data.numpy()))
        if table:
            wandb.log({prefix + "table": self.table})
        # return accuracy


class DocumentMseEvaluator:
    """
    Evaluator class for softmax classification
    """

    def __init__(self, cuda, csv_path, leeway):
        self.cuda = cuda
        self.table = wandb.Table(
            columns=["img", "tl", "tr", "br", "bl", "path", "total", "sqr-e"]
        )
        self.csv_path = csv_path
        self.leeway = leeway

    def cordinate_within_intervals(
        self, cordinate, x_interval, y_interval
    ) -> int:

        is_within_x = x_interval[0] <= cordinate[0] <= x_interval[1]
        is_within_y = y_interval[0] <= cordinate[1] <= y_interval[1]

        return int(is_within_x and is_within_y)

    def euclidean_distance_np(self, x1, y1, x2, y2):
        # Convert to NumPy arrays if they aren't already
        x1, y1, x2, y2 = np.array(x1), np.array(y1), np.array(x2), np.array(y2)
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_area_vect(
        self, y_lower_bound, y_upper_bound, x_lower_bound, x_upper_bound
    ):
        width = x_upper_bound - x_lower_bound
        height = y_upper_bound - y_lower_bound

        width = np.maximum(0, width)
        height = np.maximum(0, height)

        # Vectorized area calculation
        return width * height

    def fill_table(self, imgs, results, loss_per_batch):
        loss_per_batch = loss_per_batch.cpu().data.numpy()
        for idx in range(len(imgs)):
            img = imgs[idx].cpu().data.numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img).resize((200, 200))
            # img=np.array(img)
            result = results[idx]

            bb_s = [
                result["top_left"][2:],
                result["top_right"][2:],
                result["bottom_right"][2:],
                result["bottom_left"][2:],
            ]

            cordinates = [
                result["top_left"][0],
                result["top_right"][0],
                result["bottom_right"][0],
                result["bottom_left"][0],
            ]
            labels = [
                result["top_left"][1],
                result["top_right"][1],
                result["bottom_right"][1],
                result["bottom_left"][1],
            ]

            # for bb_ in bb_s:
            img = draw_multiple_bounding_boxes(img, bb_s)
            img = highlight_coordinates(img, cordinates, "green")
            img = highlight_coordinates(img, labels, "blue")

            path = result["path"]
            contains_tl = result["contains_tl"]
            contains_tr = result["contains_tr"]
            contains_br = result["contains_br"]
            contains_bl = result["contains_bl"]
            total = result["total_corners"]
            self.table.add_data(
                wandb.Image(np.array(img)),
                contains_tl,
                contains_tr,
                contains_br,
                contains_bl,
                path,
                total,
                loss_per_batch[idx],
            )

    def evaluate_corners(
        self,
        x_cords: np.ndarray,
        y_cords: np.ndarray,
        target: np.ndarray,
        paths: str,
        loss_per_example,
        leeway,
    ) -> Dict:

        target = target.cpu().data.numpy()

        x0, x1, x2, x3 = (
            x_cords[:, 0],
            x_cords[:, 1],
            x_cords[:, 2],
            x_cords[:, 3],
        )
        y0, y1, y2, y3 = (
            y_cords[:, 0],
            y_cords[:, 1],
            y_cords[:, 2],
            y_cords[:, 3],
        )

        target_x = target[:, [0, 2, 4, 6]]
        target_y = target[:, [1, 3, 5, 7]]
        loss_per_example = loss_per_example.cpu().data.numpy()

        doc_width = (
            self.euclidean_distance_np(x0, y0, x1, y1)
            + self.euclidean_distance_np(x3, y3, x2, y2)
        ) / 2
        doc_height = (
            self.euclidean_distance_np(x0, y0, x3, y3)
            + self.euclidean_distance_np(x1, y1, x2, y2)
        ) / 2

        # Top-left bounds
        top_left_y_lower_bound = np.maximum(0, (2 * y0 - (y3 + y0) / 2))
        top_left_y_upper_bound = (y3 + y0) / 2
        top_left_x_lower_bound = np.maximum(0, (2 * x0 - (x1 + x0) / 2))
        top_left_x_upper_bound = (x1 + x0) / 2

        # Top-right bounds
        top_right_y_lower_bound = np.maximum(0, (2 * y1 - (y1 + y2) / 2))
        top_right_y_upper_bound = (y1 + y2) / 2
        top_right_x_lower_bound = (x1 + x0) / 2
        top_right_x_upper_bound = np.minimum(1, (x1 + (x1 - x0) / 2))

        # Bottom-right bounds
        bottom_right_y_lower_bound = (y1 + y2) / 2
        bottom_right_y_upper_bound = np.minimum(1, (y2 + (y2 - y1) / 2))
        bottom_right_x_lower_bound = (x2 + x3) / 2
        bottom_right_x_upper_bound = np.minimum(1, (x2 + (x2 - x3) / 2))

        # Bottom-left bounds
        bottom_left_y_lower_bound = (y0 + y3) / 2
        bottom_left_y_upper_bound = np.minimum(1, (y3 + (y3 - y0) / 2))
        bottom_left_x_lower_bound = np.maximum(0, (2 * x3 - (x2 + x3) / 2))
        bottom_left_x_upper_bound = (x3 + x2) / 2

        partitions_dictionary = {
            "top_left": [
                top_left_y_lower_bound,
                top_left_y_upper_bound,
                top_left_x_lower_bound,
                top_left_x_upper_bound,
                (x_cords[:, 0], y_cords[:, 0]),
            ],
            "top_right": [
                top_right_y_lower_bound,
                top_right_y_upper_bound,
                top_right_x_lower_bound,
                top_right_x_upper_bound,
                (x_cords[:, 1], y_cords[:, 1]),
            ],
            "bottom_right": [
                bottom_right_y_lower_bound,
                bottom_right_y_upper_bound,
                bottom_right_x_lower_bound,
                bottom_right_x_upper_bound,
                (x_cords[:, 2], y_cords[:, 2]),
            ],
            "bottom_left": [
                bottom_left_y_lower_bound,
                bottom_left_y_upper_bound,
                bottom_left_x_lower_bound,
                bottom_left_x_upper_bound,
                (x_cords[:, 3], y_cords[:, 3]),
            ],
        }

        for key in partitions_dictionary.keys():
            current_bb = partitions_dictionary[key]
            predicted_cordinates = current_bb[4]

            y_lower_bound = current_bb[0]
            y_upper_bound = current_bb[1]
            x_lower_bound = current_bb[2]
            x_upper_bound = current_bb[3]

            BB_area_boolean_mask = (
                self.calculate_area_vect(
                    y_lower_bound, y_upper_bound, x_lower_bound, x_upper_bound
                )
                < 0.05
            )

            new_y_lower_bound = current_bb[0]
            new_y_upper_bound = current_bb[1]
            new_x_lower_bound = current_bb[2]
            new_x_upper_bound = current_bb[3]

            new_y_lower_bound[BB_area_boolean_mask] = (
                predicted_cordinates[1][BB_area_boolean_mask]
                - leeway * doc_height[BB_area_boolean_mask]
            )
            new_y_upper_bound[BB_area_boolean_mask] = (
                predicted_cordinates[1][BB_area_boolean_mask]
                + leeway * doc_height[BB_area_boolean_mask]
            )
            new_x_lower_bound[BB_area_boolean_mask] = (
                predicted_cordinates[0][BB_area_boolean_mask]
                - leeway * doc_width[BB_area_boolean_mask]
            )
            new_x_upper_bound[BB_area_boolean_mask] = (
                predicted_cordinates[0][BB_area_boolean_mask]
                + leeway * doc_width[BB_area_boolean_mask]
            )

            partitions_dictionary[key] = [
                new_y_lower_bound,
                new_y_upper_bound,
                new_x_lower_bound,
                new_x_upper_bound,
                predicted_cordinates,
            ]

        tl = (
            (partitions_dictionary["top_left"][0] <= target_y[:, 0])
            & (target_y[:, 0] <= partitions_dictionary["top_left"][1])
            & (partitions_dictionary["top_left"][2] <= target_x[:, 0])
            & (target_x[:, 0] <= partitions_dictionary["top_left"][3])
        )

        tr = (
            (partitions_dictionary["top_right"][0] <= target_y[:, 1])
            & (target_y[:, 1] <= partitions_dictionary["top_right"][1])
            & (partitions_dictionary["top_right"][2] <= target_x[:, 1])
            & (target_x[:, 1] <= partitions_dictionary["top_right"][3])
        )

        br = (
            (partitions_dictionary["bottom_right"][0] <= target_y[:, 2])
            & (target_y[:, 2] <= partitions_dictionary["bottom_right"][1])
            & (partitions_dictionary["bottom_right"][2] <= target_x[:, 2])
            & (target_x[:, 2] <= partitions_dictionary["bottom_right"][3])
        )

        bl = (
            (partitions_dictionary["bottom_left"][0] <= target_y[:, 3])
            & (target_y[:, 3] <= partitions_dictionary["bottom_left"][1])
            & (partitions_dictionary["bottom_left"][2] <= target_x[:, 3])
            & (target_x[:, 3] <= partitions_dictionary["bottom_left"][3])
        )

        result_dicts = []
        for idx in range(len(paths)):

            result_dict = {
                "path": paths[idx],
                "contains_tl": tl[idx],
                "contains_tr": tr[idx],
                "contains_br": br[idx],
                "contains_bl": bl[idx],
                "total_corners": tl[idx] + tr[idx] + br[idx] + bl[idx],
                "top_left": (
                    (
                        partitions_dictionary["top_left"][4][0][idx],
                        partitions_dictionary["top_left"][4][1][idx],
                    ),
                    (target_x[idx][0], target_y[idx][0]),
                    partitions_dictionary["top_left"][0][idx],
                    partitions_dictionary["top_left"][1][idx],
                    partitions_dictionary["top_left"][2][idx],
                    partitions_dictionary["top_left"][3][idx],
                ),
                "top_right": (
                    (
                        partitions_dictionary["top_right"][4][0][idx],
                        partitions_dictionary["top_right"][4][1][idx],
                    ),
                    (target_x[idx][1], target_y[idx][1]),
                    partitions_dictionary["top_right"][0][idx],
                    partitions_dictionary["top_right"][1][idx],
                    partitions_dictionary["top_right"][2][idx],
                    partitions_dictionary["top_right"][3][idx],
                ),
                "bottom_right": (
                    (
                        partitions_dictionary["bottom_right"][4][0][idx],
                        partitions_dictionary["bottom_right"][4][1][idx],
                    ),
                    (target_x[idx][2], target_y[idx][2]),
                    partitions_dictionary["bottom_right"][0][idx],
                    partitions_dictionary["bottom_right"][1][idx],
                    partitions_dictionary["bottom_right"][2][idx],
                    partitions_dictionary["bottom_right"][3][idx],
                ),
                "bottom_left": (
                    (
                        partitions_dictionary["bottom_left"][4][0][idx],
                        partitions_dictionary["bottom_left"][4][1][idx],
                    ),
                    (target_x[idx][3], target_y[idx][3]),
                    partitions_dictionary["bottom_left"][0][idx],
                    partitions_dictionary["bottom_left"][1][idx],
                    partitions_dictionary["bottom_left"][2][idx],
                    partitions_dictionary["bottom_left"][3][idx],
                ),
                "loss": loss_per_example[idx],
            }
            result_dicts.append(result_dict)
        return result_dicts

    def evaluate(self, model, iterator, epoch, prefix, table):
        model.eval()
        lossAvg = None
        classification_results = []
        with torch.no_grad():
            for img, target, paths in tqdm(iterator):
                if self.cuda:
                    img, target = img.cuda(), target.cuda()

                response = model(Variable(img))

                loss_per_example = F.mse_loss(
                    response, Variable(target.float()), reduction="none"
                )
                loss_per_example = loss_per_example.mean(dim=1)
                loss = loss_per_example.mean()
                loss = torch.sqrt(loss)

                # model_prediction = self.model(img_temp)[0]

                model_prediction = np.array(response.cpu().data.numpy())

                x_cords = model_prediction[:, [0, 2, 4, 6]]
                y_cords = model_prediction[:, [1, 3, 5, 7]]

                classification_result = self.evaluate_corners(
                    x_cords, y_cords, target, paths, loss_per_example
                )
                classification_results.extend(classification_result)
                if table:
                    self.fill_table(
                        img, classification_result, loss_per_example
                    )

                if lossAvg is None:
                    lossAvg = loss
                else:
                    lossAvg += loss
                # logger.debug("Cur loss %s", str(loss))
        df = pd.DataFrame(classification_results)

        df.to_csv(self.csv_path, index=False)

        lossAvg /= len(iterator)
        total_corners = df["total_corners"]
        accuracy = (np.sum(total_corners) / 4) / len(total_corners)

        accuracy_4 = np.sum(total_corners == 4) / len(total_corners)
        accuracy_3 = np.sum(total_corners >= 3) / len(total_corners)

        wandb.log(
            {
                "epoch": epoch,
                prefix + "eval_loss": lossAvg,
                prefix + "accuracy": accuracy,
                prefix + "4_corners_accuracy": accuracy_4,
                prefix + "3_corners_accuracy": accuracy_3,
            }
        )
        # logger.info("Avg Val Loss %s", str((lossAvg).cpu().data.numpy()))
        if table:
            wandb.log({prefix + "table": self.table})
        return accuracy, accuracy_4, accuracy_3


class CompleteDocEvaluator:
    """
    Evaluator class for softmax classification
    """

    def __init__(self, cuda):
        self.cuda = cuda

    def evaluate(self, model, iterator, epoch, prefix, table=True):
        model.eval()

        test_table = wandb.Table(
            columns=[
                "img",
                "directory",
                "prediction",
                "label",
                "dataset",
                "path",
            ]
        )

        lossAvg = None
        all_targets = []
        all_predictions = []
        loss_function = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for img, target, directory, dataset, path in tqdm(iterator):
                if self.cuda:
                    img, target = img.cuda(), target.cuda()

                response = model(Variable(img))

                loss = loss_function(response, Variable(target.float()))

                if lossAvg is None:
                    lossAvg = loss
                else:
                    lossAvg += loss

                predictions = torch.argmax(response, dim=1).cpu().numpy()
                target = torch.argmax(target, dim=1).cpu().numpy()
                all_targets.extend(target)
                all_predictions.extend(predictions)
                img = np.transpose(img.cpu().numpy(), (0, 2, 3, 1))
                if table:
                    for idx in range(len(target)):
                        numpy_img = wandb.Image(img[idx])
                        test_table.add_data(
                            numpy_img,
                            directory[idx],
                            predictions[idx],
                            target[idx],
                            dataset[idx],
                            path[idx],
                        )

        lossAvg /= len(iterator)

        # Compute metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(
            all_targets, all_predictions, average="binary"
        )
        recall = recall_score(all_targets, all_predictions, average="binary")
        f1 = f1_score(all_targets, all_predictions, average="binary")

        if table:
            wandb.log({prefix + "table": test_table})
            matrix = wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_targets,
                preds=all_predictions,
                class_names=["Full", "Incomplete"],
            )

            wandb.log({prefix + "confussion_matrix": matrix})
        # Log metrics to wandb
        wandb.log(
            {
                "epoch": epoch,
                prefix + "eval_loss": lossAvg.cpu().numpy(),
                prefix + "eval_accuracy": accuracy,
                prefix + "eval_precision": precision,
                prefix + "eval_recall": recall,
                prefix + "eval_f1": f1,
            }
        )
