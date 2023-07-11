import argparse
from copy import copy
from typing import Tuple, List, Dict

import cv2
import numpy as np

from src.Image import Image


class Component(Image):
    image_centre: np.array
    chip_centre: Tuple[float, float] = None
    corners: np.array

    initial_scale: bool = False

    # dynamic storage
    scaled_image: Dict[float, np.array] = dict()
    scaled_labels: Dict[float, np.array] = dict()

    flipped_image: Dict[str, np.array] = dict()
    flipped_label: Dict[str, np.array] = dict()

    def __init__(self, img_path: str, label_path: str):
        super().__init__(img_path)

        # label txt only contains one row of data for the bounding box
        with open(label_path, "r") as file:
            label_data = file.read()
            corner_coordinate = [float(n) for n in label_data.split()[:8]]

        # (x, y) in (width, height)
        self.corners = np.array([[corner_coordinate[i], corner_coordinate[i + 1]] for i in range(0, 8, 2)])
        self.img_centre = np.divide(self.img_size[: 2], 2)
        self.chip_centre = self.__find_chip_center()

    def __find_chip_center(self) -> Tuple[float, float]:
        x_y = np.array(self.corners)
        centre_x, centre_y = x_y.mean(axis=0)

        return centre_x, centre_y

    @staticmethod
    def convert_TL_to_centre(component_size: Tuple[int], component_label: np.array):
        return np.array([[pos[0] - component_size[1] / 2, pos[1] - component_size[0] / 2] for pos in component_label])

    @staticmethod
    def convert_centre_to_TL(component_size: Tuple[int], component_label: np.array):
        return np.array([[pos[0] - component_size[1] / 2, pos[1] - component_size[0] / 2] for pos in component_label])

    def update_resizing(self, scale: float):
        """

        :param scale:
        :return:
        """
        if not self.resize_into_flat:
            raise Exception(f"Error: Incorrect update the component information. May want to use 'add_resizing_res'")

        self.img_centre = np.divide(self.img_size[: 2], 2)
        self.corners = np.array([pos * scale for pos in self.corners], dtype=np.int32)
        self.chip_centre = np.multiply(self.chip_centre, scale)

    def add_resizing_res(self, size: Tuple[int], scale: float):
        component_img = cv2.resize(self.read(), size)
        component_label = np.array([pos * scale for pos in self.corners], dtype=np.int32)

        self.scaled_image[scale] = component_img
        self.scaled_labels[scale] = component_label

        return component_img, component_label

    def add_flipping_res(self, flip: str,
                         component_img: np.array,
                         component_label: np.array):
        self.flipped_image[flip] = component_img
        self.flipped_label[flip] = component_label

    def draw_box(self,
                 flag: str,
                 scale: float = None,
                 component_img: np.array = None,
                 component_label: np.array = None):
        if component_img is None and component_label is None and not scale:
            img = self.read()
            pts = self.corners.reshape((-1, 1, 2)).astype(np.int32)
        elif component_img is not None and component_label is not None:
            img = copy(component_img)
            pts = component_label.reshape((-1, 1, 2)).astype(np.int32)
        elif scale:
            img = copy(self.scaled_image[scale])
            pts = self.scaled_labels[scale].reshape((-1, 1, 2)).astype(np.int32)
        else:
            raise Exception(f"Error: Incorrect combination of parameters is given")

        cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        cv2.imwrite(f"debug_{flag}_{self.img_name}.png", img)
