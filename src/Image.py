from __future__ import annotations

from copy import copy
from typing import Tuple, Dict, List

import cv2
import numpy as np

from src.typeHint import LabelsType


class Image:
    """

    """
    def __init__(self, img_path: str):
        self.__image: np.ndarray = cv2.imread(img_path)
        self.img_name: str = "_".join(img_path.split("/")[-1].split(".")[:-1])
        self.img_size: Tuple[int, int] = self.__image.shape[: 2]
        self.ext: str = img_path[-3:]
        self.resize_into_flag: bool = False

    def resize_into(self, width: int, height: int):
        self.__image = cv2.resize(self.__image, (width, height))
        self.img_size = self.__image.shape[: 2]
        self.resize_into_flag = True

    def show(self):
        cv2.imshow(self.img_name, self.__image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def read(self) -> np.array:
        return self.__image.copy()

    def save(self):
        cv2.imwrite(f"debug_{self.img_name}.png", self.__image)

    def set_image(self, image: np.array):
        self.__image = image

    @staticmethod
    def plot_labels(img: np.ndarray,
                    labels: LabelsType,
                    color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        canvas = img.copy()

        for label_type, value_list in labels.items():
            for label in value_list:
                pts = label.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(canvas, [pts], True, color=color, thickness=2)

        return canvas
