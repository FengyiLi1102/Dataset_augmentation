import os.path
import re
from typing import Union

import cv2
import numpy as np

from src.Image import Image
from src.utils import mkdir_if_not_exists


class Background(Image):
    texture: str  # clean, noisy, messy
    # __resized_image: np.array

    background_textures = ["clean", "noisy", "noisyL", "messy"]

    def __init__(self,
                 img_path: Union[str, None],
                 mosaic_size: int = 0,
                 img: np.array = None,
                 img_name: str = None,
                 texture: str = None):
        super().__init__(img_path=img_path)
        if not img_path:
            if img is None or img_name is None or texture is None:
                raise ValueError(f"Cannot fast create the Background object due to incorrect data provide")

            # complete Image attributes
            self.fast_init(img, img_name)

            self.texture = texture
        else:
            # category into different textures
            idx = 1 if not mosaic_size else 2
            self.texture = re.split("[_.]", self.img_name)[idx]

        if mosaic_size:
            self.resize_into(mosaic_size, mosaic_size)

    @staticmethod
    def draw_box(save_name: str,
                 background_img: np.array,
                 label: np.array,
                 save_dir: str = "../debug"):
        mkdir_if_not_exists(save_dir)

        pts = label.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(background_img, [pts], True, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_dir, f"debug_{save_name}.png"), background_img)
