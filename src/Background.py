import os.path
import re
from typing import Union, List

import cv2
import numpy as np

from src.Image import Image
from src.utils import mkdir_if_not_exists


class Background(Image):
    background_textures = ["clean", "noisy", "noisyL", "messy"]

    def __init__(self,
                 img_path: Union[str, None],
                 mosaic_size: int = 0):
        super().__init__(img_path=img_path)
        self.texture: str  # clean, noisy, messy

        # classify into different textures
        self.texture = re.split("[_.]", self.img_name)[1]

        if mosaic_size:
            self.resize_into(mosaic_size, mosaic_size)

    @staticmethod
    def draw_box(save_name: str,
                 background_img: np.ndarray,
                 labels: List[np.ndarray],
                 save_dir: str = "../debug"):
        mkdir_if_not_exists(save_dir)

        for label in labels:
            pts = label.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(background_img, [pts], True, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(save_dir, f"debug_{save_name}.png"), background_img)
