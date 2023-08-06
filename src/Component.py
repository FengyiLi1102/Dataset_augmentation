import os.path
from collections import defaultdict
from typing import Tuple, List, Dict, Union

import cv2
import numpy as np

from src.constant import DNA_ORIGAMI
from src.Image import Image
from src.utils import mkdir_if_not_exists, process_labels


class Component(Image):
    """
    Component object can represent DNA origami chip, and also augmented data because they both require image array
    and label data.
    """

    def __init__(self, img_path: str, label_path: str):
        super().__init__(img_path)

        self.image_centre: np.array
        self.chip_centre: Tuple[float, float]

        # label txt only contains one row of data for the bounding box
        self.labels: Dict[str, List[np.ndarray]] = defaultdict(list)

        with open(label_path, "r") as file:
            for line in file:
                values = line.strip().split(" ")[: -1]

                # (x, y) in (width, height)
                # Two objects: DNA-origami and active-site
                self.labels[values[-1]].append(np.array(values[: -1], dtype=np.float64).reshape(-1, 2))
                # self.corners = np.array(label_data.split()[:8], dtype=np.float64).reshape(-1, 2)

        self.img_centre = np.divide(self.img_size[: 2], 2)

        if DNA_ORIGAMI in self.labels.keys():
            self.chip_centre = self.__find_chip_center()

        # morphology
        if img_path is not None:
            self.morphology: str = img_path.split("/")[-1].split("_")[1]
        else:
            self.morphology = "N/A"

        self.initial_scale: bool = False

        # dynamic storage
        self.scaled_image: Dict[float, np.ndarray] = dict()
        self.scaled_labels: Dict[float, Dict[str, List[np.ndarray]]] = dict()

        self.flipped_image: Dict[str, np.ndarray] = dict()
        self.flipped_label: Dict[str, Dict[str, List[np.ndarray]]] = dict()

    def __find_chip_center(self) -> Tuple[float, float]:
        x_y = self.labels[DNA_ORIGAMI][0]
        centre_x, centre_y = x_y.mean(axis=0)

        return centre_x, centre_y

    @staticmethod
    def convert_TL_to_centre(component_size: Tuple[int, int], component_labels: Dict[str, List[np.ndarray]]) \
            -> Dict[str, List[np.ndarray]]:
        return process_labels(component_size, component_labels, Component.__from_TL_to_centre)

    @staticmethod
    def __from_TL_to_centre(img_size: Tuple[int, int], labels: np.ndarray) -> np.ndarray:
        return np.array([[pos[0] - img_size[1] / 2, pos[1] - img_size[0] / 2] for pos in labels])

    @staticmethod
    def convert_centre_to_TL(new_coordinates: Tuple[int, int], component_labels: Dict[str, List[np.ndarray]]) \
            -> Dict[str, List[np.ndarray]]:
        """

        :param new_coordinates: (x, y)
        :param component_labels:
        :return:
        """
        return process_labels(new_coordinates, component_labels, Component.__from_centre_to_TL)

    @staticmethod
    def __from_centre_to_TL(new_coordinates: Tuple[int, int], labels: np.ndarray) -> np.ndarray:
        """

        :param new_coordinates: (x, y)
        :param labels:
        :return:
        """
        return labels + np.array(new_coordinates)

    def update_resizing_res(self, scale: float):
        """
        Update attributes which are changed due to the resizing operation
        :param scale:
        :return:
        """
        if not self.resize_into_flag:
            raise Exception(f"Error: Incorrect update the component information. May want to use 'add_resizing_res'")

        self.img_centre = np.divide(self.img_size[: 2], 2)
        self.labels = self.__rescale_label(self.labels, scale)
        self.chip_centre = np.multiply(self.chip_centre, scale)

    def add_resizing_res(self, size: Tuple[int], scale_side: float) \
            -> Tuple[np.ndarray, Dict[str, List[np.ndarray]]]:
        """
        Add data generated due to the scale.
        :param size: (x, y)
        :param scale_side:
        :return:
        """
        component_img = cv2.resize(self.read(), size)
        component_label = self.__rescale_label(self.labels, scale_side)

        scale_area = round(scale_side ** 2, 1)
        self.scaled_image[scale_area] = component_img
        self.scaled_labels[scale_area] = component_label

        return component_img, component_label

    @staticmethod
    def __rescale_label(labels: Dict[str, List[np.ndarray]], scale: float) -> Dict[str, List[np.ndarray]]:
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

    def draw_box(self,
                 flag: str,
                 scale: float = None,
                 component_img: np.array = None,
                 component_label: Dict[str, List[np.ndarray]] = None,
                 save_directory: str = "../debug"):
        mkdir_if_not_exists(save_directory)

        if component_img is None and component_label is None and not scale:
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

        Image.plot_labels(img, label_place_holder)
        cv2.imwrite(os.path.join(save_directory, f"debug_{flag}_{self.img_name}.png"), img)
