import math
import os.path
from collections import defaultdict
from typing import Tuple, List, Dict, Union

import cv2
import numpy as np

from src.constant import DNA_ORIGAMI
from src.Image import Image
from src.utils import mkdir_if_not_exists, process_labels, compute_angle_between_horizontal, find_clockwise_order
from src.typeHint import LabelsType, PointImageType, PointCoordinateType


class Component(Image):
    """
    Component object can represent DNA origami chip, and also augmented data because they both require image array
    and label data.
    """

    def __init__(self, img_path: str, label_path: str):
        super().__init__(img_path)

        self.image_centre: np.array  # refer to TL
        self.chip_centre: Tuple[int, int] | None = None  # refer to TL
        self.chip_w: int = 0
        self.chip_h: int = 0

        # label txt only contains one row of data for the bounding box
        self.labels: LabelsType = defaultdict(list)  # refer to TL

        # origami chip box relative to the centre of the chip
        # refer to chip centre
        # cartisian coordinate
        self.box: np.ndarray | None = None

        self.initial_scale: bool = False

        with open(label_path, "r") as file:
            for line in file:
                values = line.strip().split(" ")[: -1]

                # corners are in the order of clockwise from the top left point
                # NOTE: The manually labeled chips should be in a clear orientation with four corners at top left,
                # top right, bottom right and bottom left positions. Tejas' cropping function and the YOLOv5-OBB model
                # detection both can reach this.

                # (x, y) in (width, height)
                # Two objects: DNA-origami and active-site
                self.labels[values[-1]].append(np.array(values[: -1], dtype=np.float64).reshape(-1, 2))

        self.img_centre = np.divide(self.img_size[: 2], 2)

        if DNA_ORIGAMI in self.labels.keys():
            self.chip_centre = self.find_chip_center()
            self.chip_w, self.chip_h = self.compute_chip_size(self.labels[DNA_ORIGAMI][0])

            # initial rotation of the bottom side of the chip respect to the horizontal direction
            self.chip_rotation: float = compute_angle_between_horizontal(labels=self.labels[DNA_ORIGAMI][0])

        # morphology
        if img_path is not None:
            self.morphology: str = img_path.split("/")[-1].split("_")[1]
        else:
            self.morphology = "N/A"

        # dynamic storage
        # scale range may only contain 10 or 15 options, and these are quite few for a data augmentation with
        # 1500 or even more tasks, which means that the dynamic storage can save time on the expense of space
        self.scaled_image: Dict[float, np.ndarray] = dict()  # area scales: image
        self.scaled_labels: Dict[float, Dict[str, List[np.ndarray]]] = dict()

        self.flipped_image: Dict[str, np.ndarray] = dict()  # flip: image(initial scale)
        self.flipped_label: Dict[str, Dict[str, List[np.ndarray]]] = dict()

        self.chip_wh_cache: Dict[float, Tuple[float, float]] = dict()  # area scales: (chip_w, chip_h)

    @staticmethod
    def compute_chip_size(chip_label: np.ndarray) -> Tuple[float, float]:
        # compute width and height
        temp_chip_label = chip_label.copy()
        temp_chip_label_w = temp_chip_label[np.argsort(temp_chip_label[:, 1])]
        temp_chip_label_h = temp_chip_label[np.argsort(temp_chip_label[:, 0])]

        return (np.linalg.norm(temp_chip_label_w[0] - temp_chip_label_w[1]),
                np.linalg.norm(temp_chip_label_h[0] - temp_chip_label_h[1]))

    def find_chip_center(self, chip_label: np.ndarray | None = None) -> Tuple[float, float]:
        if chip_label is None:
            chip_label = self.labels[DNA_ORIGAMI][0]

        chip_centre_x, chip_centre_y = np.mean(chip_label, axis=0)

        self.box = self.from_TL_to_centre((chip_centre_x, chip_centre_y), chip_label, cartesian=False)

        return chip_centre_x, chip_centre_y

    @staticmethod
    def convert_TL_to_centre(centre: PointCoordinateType,
                             component_labels: LabelsType,
                             cartesian: bool = False) -> LabelsType:
        """

        :param centre: (x, y)
        :param component_labels:
        :param cartesian:
        :return:
        """
        return process_labels(centre, component_labels, Component.from_TL_to_centre, cartesian)

    @staticmethod
    def from_TL_to_centre(centre: PointCoordinateType | np.ndarray,
                          labels: np.ndarray,
                          cartesian: bool) -> np.ndarray:
        """

        :param centre: (x, y)
        :param labels:
        :return:
        """
        if cartesian:
            return np.array([[pos[0] - centre[0], - pos[1] + centre[1]] for pos in labels])
        else:
            return np.array([[pos[0] - centre[0], pos[1] - centre[1]] for pos in labels])

    @staticmethod
    def convert_centre_to_TL(new_coordinates: PointCoordinateType,
                             component_labels: LabelsType,
                             cartesian: bool) -> LabelsType:
        """

        :param cartesian:
        :param new_coordinates: (x, y)
        :param component_labels:
        :return:
        """
        return process_labels(new_coordinates, component_labels, Component.__from_centre_to_TL, cartesian)

    @staticmethod
    def __from_centre_to_TL(new_centre: PointCoordinateType,
                            labels: np.ndarray,
                            cartesian: bool) -> np.ndarray:
        """

        :param new_centre: (x, y) wrt the background image top left corner
        :param labels:
        :return:
        """
        res = labels + np.array(new_centre)

        if cartesian:
            res *= np.array([1, -1])

        return res

    def update_resizing_res(self, side_scale: float):
        """
        Update attributes which are changed due to the resizing operation.
        :param side_scale: for the side instead of the area
        :return:
        """
        if not self.resize_into_flag:
            raise Exception(f"Error: Incorrect update the component information. May want to use 'add_resizing_res'")

        self.img_centre = np.divide(self.img_size, 2)
        self.labels = self.rescale_label(self.labels, side_scale)
        self.chip_centre = np.multiply(self.chip_centre, side_scale)
        self.chip_h *= side_scale
        self.chip_w *= side_scale
        self.box = self.from_TL_to_centre(self.chip_centre, self.labels[DNA_ORIGAMI][0], cartesian=False)

        # recover the flag
        self.resize_into_flag = False

    def add_resizing_res(self,
                         size: PointImageType,
                         side_scale: float,
                         decimal_places: int = 2) -> Tuple[np.ndarray, Dict[str, List[np.ndarray]]]:
        """
        Add data generated due to the scale.
        :param size: (x, y)
        :param side_scale:
        :param decimal_places:
        :return:
        """
        component_img = cv2.resize(self.read(), size)
        component_label = self.rescale_label(self.labels, side_scale)

        scale_area = round(side_scale ** 2, decimal_places)
        self.scaled_image[scale_area] = component_img
        self.scaled_labels[scale_area] = component_label
        self.chip_wh_cache[scale_area] = self.compute_chip_size(component_label[DNA_ORIGAMI][0])

        return component_img, component_label

    @staticmethod
    def rescale_label(labels: LabelsType, scale: float) -> LabelsType:
        rescaled_labels = defaultdict(list)

        for label_type, value_list in labels.items():
            for label in value_list:
                rescaled_labels[label_type].append(np.dot(scale, label))

        return rescaled_labels

    def add_flipping_res(self, flip: str,
                         component_img: np.array,
                         component_label: Dict[str, List[np.ndarray]]):
        self.flipped_image[flip] = component_img
        self.flipped_label[flip] = component_label

    @staticmethod
    def order_labels_in_clockwise_order(labels: LabelsType) -> LabelsType:
        res = defaultdict(list)

        for key, value in labels.items():
            for label in value:
                res[key].append(find_clockwise_order(label))

        return res

    @staticmethod
    def crop(component_img: np.ndarray,
             component_labels: LabelsType,
             dilation: int = 10,
             debug: bool = False,
             pos:Tuple[int, int] | None = None) -> Tuple[np.ndarray, LabelsType]:
        """

        :param component_labels:
        :param component_img:
        :param dilation:
        :param debug:
        :return:
        """
        # Create the canvas to contain the cropped chip
        canvas = np.zeros_like(component_img, dtype=np.uint8)
        chip_label = component_labels[DNA_ORIGAMI][0].astype(np.int32)

        # Create the mask to crop the chip based on the manually labeled box
        mask = np.zeros_like(component_img[:, :, 0], dtype=np.uint8)
        cv2.fillPoly(mask, [chip_label], 255)

        # Dilate the kernel due to the imperfect labeling
        kernel = np.ones((dilation, dilation), dtype=np.uint8)
        dilated_mask = cv2.dilate(mask, kernel)

        # Find the tight bounding box
        y_range, x_range = np.where(dilated_mask == 255)
        min_x, max_x = np.min(x_range), np.max(x_range)
        min_y, max_y = np.min(y_range), np.max(y_range)

        # Crop the canvas to the bounding box
        canvas[dilated_mask == 255] = component_img[dilated_mask == 255]
        cropped_canvas = canvas[min_y:max_y + 1, min_x:max_x + 1]

        # Update the label coordinates
        # new_label = chip_label.copy()
        # new_label[:, 0] -= min_x
        # new_label[:, 1] -= min_y

        labels_copy = component_labels.copy()
        for key, value in labels_copy.items():
            for labels in value:
                labels[:, 0] -= min_x
                labels[:, 1] -= min_y

        # for debugging
        if debug:
            # pts = np.array(new_label).reshape((-1, 1, 2)).astype(np.int32)
            # cv2.polylines(cropped_canvas, [pts], True, (0, 0, 255), 2)
            img = Image.plot_labels(cropped_canvas, labels_copy)
            cv2.imwrite(f"../debug/debug_5_cropped_image_{pos}.png", img)

        return cropped_canvas, labels_copy

    @staticmethod
    def convert_to_chip_centre_cartesian(labels: LabelsType) -> LabelsType:
        chip_label = labels[DNA_ORIGAMI][0]
        chip_centre = tuple(np.mean(chip_label, axis=0))

        return Component.convert_TL_to_centre(chip_centre, labels, cartesian=True)

    def draw_box(self,
                 flag: str,
                 scale: float = None,
                 component_img: np.array = None,
                 component_label: LabelsType = None,
                 save_directory: str = "../debug"):
        mkdir_if_not_exists(save_directory)

        if component_img is None and component_label is None and scale is None:
            # plot initial image
            img = self.read()
            label_place_holder = self.labels
        elif component_img is not None and component_label is not None:
            # plot processed one provided required data
            img = component_img.copy()
            label_place_holder = component_label
        elif scale:
            # plot just rescaled one
            img = self.scaled_image[scale].copy()
            label_place_holder = self.scaled_labels[scale]
        else:
            raise Exception(f"Error: Incorrect combination of parameters is given")

        img = Image.plot_labels(img, label_place_holder)
        cv2.imwrite(os.path.join(save_directory, f"debug_{flag}_{self.img_name}.png"), img)
