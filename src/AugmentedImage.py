import numpy as np

from src.Component import Component


class AugmentedImage(Component):
    _category: str
    _component_id: int
    _background_id: int
    _scale: float
    _flip: str
    _rotate: int
    _labelTxt: str

    def __init__(self,
                 category: str,
                 img_path: str = None,
                 label_path: str = None,
                 img: np.array = None,
                 img_name: str = None,
                 label: np.array = None):
        super().__init__(img_path=img_path,
                         label_path=label_path,
                         img=img,
                         img_name=img_name,
                         label=label)

        self._category = category

        # augmented_11_100_2.20_n_5.png
        values = self.img_name.split("_")

        self._component_id = int(values[1])
        self._background_id = int(values[2])
        self._scale = float(values[3])
        self._flip = values[4]
        self._rotate = int(values[5])
        self._labelTxt = self.img_name + ".txt"

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
