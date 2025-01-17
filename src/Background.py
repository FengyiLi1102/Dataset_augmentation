import os.path
import re
from typing import Union, List, Dict

import cv2
import numpy as np

from src.Image import Image
from src.utils import mkdir_if_not_exists
from src.typeHint import LabelsType


class Background(Image):
    background_textures = ["clean", "noisy", "noisyL", "messy"]

    def __init__(self,
                 img_path: str | None,
                 mosaic_size: int = 0,
                 img: np.ndarray = None,
                 img_name: str = None):
        if img_path:
            super().__init__(img_path=img_path)

        self.texture: str  # clean, noisy, messy

        if not img_path:
            self.set_image(img)
            self.img_name = img_name

        # classify into different textures
        self.texture = re.split("[_.]", self.img_name)[1]

        if mosaic_size:
            self.resize_into(mosaic_size, mosaic_size)

    @staticmethod
    def draw_box(save_name: str,
                 background_img: np.ndarray,
                 all_labels: List[LabelsType],
                 save_dir: str = "../debug"):
        mkdir_if_not_exists(save_dir)

        for one_chip_label in all_labels:
            Image.plot_labels(background_img, one_chip_label)

        cv2.imwrite(os.path.join(save_dir, f"debug_{save_name}.png"), background_img)

