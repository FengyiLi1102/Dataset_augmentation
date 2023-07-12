import argparse
import re

import cv2
import numpy as np

from src.Image import Image


class Background(Image):
    texture: str  # clean, noisy, messy
    __resized_image: np.array

    background_textures = ["clean", "noisy", "noisyL", "messy"]

    def __init__(self, img_path: str, mosaic_size: int = 0):
        super().__init__(img_path)
        # category into different textures
        idx = 1 if not mosaic_size else 2
        self.texture = re.split("[_.]", self.img_name)[idx]

        if mosaic_size:
            self.resize_into(mosaic_size, mosaic_size)

    @staticmethod
    def draw_box(save_name: str, background_img: np.array, label: np.array):
        pts = label.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(background_img, [pts], True, (0, 0, 255), 2)
        cv2.imwrite(f"debug_{save_name}.png", background_img)
