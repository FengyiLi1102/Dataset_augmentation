import numpy as np
import argparse


class Augmentor:
    # Given images for augmentation
    input_images_arr: np.array = None

    def __int__(self, args: argparse.Namespace):
