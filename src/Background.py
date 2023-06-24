import argparse

import cv2
import numpy as np

from src.Image import Image


class Background(Image):
    texture: str  # clean, noisy, messy
    __resized_image: np.array

    background_textures = ["clean", "noisy", "noisyL", "messy"]

    def __init__(self, img_path, b_size):
        super().__init__(img_path)
        # category into different textures
        self.texture = self.img_name.split(".")[0].split("_")[-1]

        self.resize_into(b_size, b_size)
