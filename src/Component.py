import argparse
from typing import Tuple, List

import numpy as np

from src.Image import Image


class Component(Image):
    centre: Tuple[float, float] = None
    corners: List[Tuple[float, float]]
    jobs: List[Tuple]

    def __init__(self, img_path: str, label_path: str):
        super().__init__(img_path)

        # label txt only contains one row of data for the bounding box
        with open(label_path, "r") as file:
            label_data = file.read()
            corner_coordinate = [float(n) for n in label_data.split()]

        self.corners = [(corner_coordinate[i], corner_coordinate[i + 1]) for i in range(0, 8, 2)]
        self.centre = self.__find_center()

    def __find_center(self) -> Tuple[float, float]:
        x_y = np.array(self.corners)
        centre_x, centre_y = x_y.mean(axis=0)

        return centre_x, centre_y

