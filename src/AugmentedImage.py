from collections import defaultdict
from typing import List, Dict

import numpy as np

from src.Image import Image


class AugmentedImage(Image):
    def __init__(self,
                 category: str,
                 img_path: str = None,
                 label_path: str = None,
                 img: np.ndarray = None,
                 img_name: str = None,
                 labels: List[Dict[str, List[np.ndarray]]] = None,
                 data: List[Dict] = None):
        self._category = category
        self._labels: Dict[str, List[np.ndarray]] = defaultdict(list)

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
        self._background_id: int = 0
        self._scale: float = .0
        self._flip: List[str] = []
        self._rotate: List[int] = []

        try:
            for record in data:
                self._component_id.append(record["Component_id"])
                self._background_id = record["Background_id"]
                self._scale = record["Component_scale"]
                self._flip.append(record["Flip"])
                self._rotate.append(record["Rotate"])

            self._labelTxt: str = self.img_name + ".txt"
        except Exception as e:
            raise Exception(f"Cannot create the Augmented Image object due to the lack of augmented data,"
                            f" given error {e}")

        # labels
        if label_path is not None:
            # label txt may contain several rows of data for the bounding box
            with open(label_path, "r") as file:
                for line in file:
                    values = line.strip().split(" ")[: -1]

                    # (x, y) in (width, height)
                    # Two objects: DNA-origami and active-site
                    self._labels[values[-1]].extend(np.array(values[: -1], dtype=np.float64).reshape(-1, 2))

        if labels is not None:
            # multiple chips in one background
            if type(labels) is list:
                for one_chip_label in labels:
                    for label_type, label in one_chip_label.items():
                        self._labels[label_type] = label
            else:
                raise TypeError(f"Incorrect label type is given: expect list or ndarray but get {type(labels)}")
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
