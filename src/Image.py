from __future__ import annotations

from copy import copy
from typing import Tuple, List, Dict, Union

import cv2
import numpy as np


class Image:
    """

    """
    img_name: str
    ext: str
    __image: np.array
    img_size: Tuple

    resize_into_flag: bool = False

    # dynamic storing augmented images for future potential use
    # img_augmented: Dict[List, np.array] = dict()

    def __init__(self,
                 img_path: Union[str, None],
                 img: np.array = None,
                 image_name_ext: str = None):
        if img_path:
            self.__image = cv2.imread(img_path)
            self.img_name = "_".join(img_path.split("/")[-1].split(".")[:-1])
            self.img_size = self.__image.shape
            self.ext = img_path[-3:]
        else:
            if img is None or image_name_ext is None:
                raise ValueError("Cannot create Image object due to missing data")

            self.__image = img
            self.img_name = image_name_ext[: -4]
            self.ext = image_name_ext[-3:]
            self.img_size = self.__image.shape

    def resize_into(self, width: int, height: int):
        self.__image = cv2.resize(self.__image, (width, height))
        self.img_size = self.__image.shape
        self.resize_into_flag = True

    def show(self):
        cv2.imshow(self.img_name, self.__image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def read(self) -> np.array:
        return copy(self.__image)

    def save(self):
        cv2.imwrite(f"debug_{self.img_name}.png", self.__image)
