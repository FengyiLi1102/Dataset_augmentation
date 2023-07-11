import argparse
import re

import cv2
import numpy as np

from src.Image import Image


class Background(Image):
    texture: str  # clean, noisy, messy
    __resized_image: np.array

    background_textures = ["clean", "noisy", "noisyL", "messy"]

    def __init__(self, img_path, b_size=None):
        super().__init__(img_path)
        # category into different textures
        self.texture = re.split("[/_.]", self.img_name)[1]

        if b_size:
            self.resize_into(b_size, b_size)

    @staticmethod
    def draw_box(save_name: str, background_img: np.array, label: np.array):
        pts = label.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(background_img, [pts], True, (0, 0, 255), 2)
        cv2.imwrite(f"debug_{save_name}.png", background_img)
