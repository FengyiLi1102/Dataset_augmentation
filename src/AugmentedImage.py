from typing import List, Union, Dict

import numpy as np

from src.Component import Component
from src.Image import Image


class AugmentedImage(Image):
    def __init__(self,
                 category: str,
                 img_path: str = None,
                 label_path: str = None,
                 img: np.ndarray = None,
                 img_name: str = None,
                 label: Union[np.ndarray, List[np.ndarray]] = None,
                 data: List[Dict] = None):
        self._category = category
        self._labels: List[np.ndarray] = []

        # image
        if img_path is not None:
            super().__init__(img_path)
        else:
            if img is None or img_name is None:
                raise Exception("Cannot create Augmented Image object due to the lack of image data")

            self.set_image(img)
            self.img_name = img_name
            self.img_size = self.read().shape
            self.ext = img_name[-3:]

        self._component_id: List[int] = []
        self._background_id: List[int] = []
        self._scale: List[float] = []
        self._flip: List[str] = []
        self._rotate: List[int] = []

        for record in data:
            self._component_id.append(record["Component_id"])
            self._background_id.append(record["Background_id"])
            self._scale.append(record["Component_scale"])
            self._flip.append(record["Flip"])
            self._rotate.append(record["Rotate"])

        self._labelTxt: str = self.img_name + ".txt"

        # labels
        if label_path is not None:
            # label txt may contain several rows of data for the bounding box
            with open(label_path, "r") as file:
                for line in file:
                    coordinates_str = line.split()[: 8]

                    # (x, y) in (width, height) in [[x, y], [x, y], ...] for each chip
                    self._labels.append(np.array(coordinates_str, dtype=np.float64).reshape(-1, 2))

        if label is not None:
            # multiple chips in one background
            if type(label) is list:
                self._labels = label
            elif type(label) is np.ndarray:
                self._labels.append(label)
            else:
                raise TypeError(f"Incorrect label type is given: expect list or ndarray but get {type(label)}")
        else:
            raise Exception("Cannot create Augmented Image object due to the lack of label data")

    @property
    def category(self):
        return self._category

    @property
    def component_id(self):
        return self._component_id

    @property
    def background_id(self):
        return self._background_id

    @property
    def scale(self):
        return self._scale

    @property
    def flip(self):
        return self._flip

    @property
    def rotate(self):
        return self._rotate

    @property
    def label_name(self):
        return self._labelTxt

    @property
    def labels(self):
        return self._labels
