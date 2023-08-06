import argparse
import os
from typing import List

import numpy as np


class ArgumentParser:
    parser: argparse.ArgumentParser

    def __init__(self):
        self.parser = argparse.ArgumentParser("Dataset augmentation parameters")

        # ================================================================================================= #
        # Functionality
        self.parser.add_argument("--function", "-f",
                                 choices=["crop_origami", "run", "generate_fake_backgrounds", "create_cache"],
                                 default="run")

        # ========================================= >>> GENERAL <<< ======================================= #
        # General parameters
        self.parser.add_argument("--img_path", "-ip",
                                 type=str,
                                 default=r"data",
                                 help="Directory of raw mosaic and component images")
        self.parser.add_argument("--save_path", "-sp",
                                 type=str,
                                 default=r"dataset",
                                 help="Directory to save processed images")
        self.parser.add_argument("--dataset_path", "-dp",
                                 type=str,
                                 default=os.getcwd(),
                                 help="Processed images including fake backgrounds and cropped components")
        self.parser.add_argument("--cache_scan",
                                 action="store_true",
                                 help="Flag to indicate scanning from provided cache files instead of images the "
                                      "directory")
        self.parser.add_argument("--cache_save_dir",
                                 type=str,
                                 default="cache")

        # ====================================== >>> BACKGROUND <<< ======================================= #
        # For generating fake backgrounds without origami chips
        self.parser.add_argument("--cache_bg_path",
                                 type=str,
                                 default="")
        self.parser.add_argument("--background_size", "-bg_size",
                                 type=np.int16,
                                 default=2560,
                                 help="Expected generated fake mosaic size")
        self.parser.add_argument("--mosaic_size", "-ms_size",
                                 type=int,
                                 default=320,
                                 help="Size of the mosaic mosaics")
        self.parser.add_argument("--bg_number", "-bg_n",
                                 type=np.int16,
                                 default=200)
        self.parser.add_argument("--bg_ratio", "-bg_r",
                                 type=np.int8,
                                 nargs=4,
                                 default=[3, 3, 3, 1],
                                 help="Number ratio among clean : noisyL : noisy : messy")
        self.parser.add_argument("--kernel_size", "-ks",
                                 type=int,
                                 default=5,
                                 help="Averaging kernel size for decease seam visibility")

        # ====================================== >>> COMPONENT <<< ======================================== #
        # For cropping DNA origami from given raw images to make component
        self.parser.add_argument("--cache_chip_path",
                                 type=str,
                                 default="")
        self.parser.add_argument("--config_path", "-cp",
                                 type=str,
                                 default=r"config/params.json")
        self.parser.add_argument("--inflation", "-infl",
                                 type=int,
                                 default=300,
                                 help="Extra pixels captured away from the detected DNA origami")

        # ====================================== >>> AUGMENTATION <<< ===================================== #
        # For producing augmentation images
        self.parser.add_argument("--dataset_name",
                                 type=str,
                                 default="one_chip_dataset")
        self.parser.add_argument("--initial_scale", "-is",
                                 type=float,
                                 default=1 / 4,
                                 help="Resize the components with a suitable size compared to backgrounds")
        self.parser.add_argument("--scale_range", "-sr",
                                 nargs=2,
                                 type=float,
                                 default=[0.4, 2.0],
                                 help="Scale range for origami")
        self.parser.add_argument("--scale_increment",
                                 type=float,
                                 default=0.1)
        self.parser.add_argument("--scaling_mode",
                                 choices=["fixed", "random", "larger", "smaller"],
                                 default="random",
                                 help="Setting for scale of the origami:\n"
                                      "fixed -> as the initial_scale\n"
                                      "random -> random scales within scale range \n"
                                      "larger -> more scale-up origami \n"
                                      "smaller -> more scale-down origami")
        self.parser.add_argument("--aug_number",
                                 type=int,
                                 default=10)
        self.parser.add_argument("--backgrounds",
                                 choices=["random", "messy", "clean", "noisy", "noisyL"],
                                 default="random")
        self.parser.add_argument("--components",
                                 choices=["random", "irregular", "regular", "broken", "sheared"],
                                 default="random")
        self.parser.add_argument("--rotation_increment",
                                 type=int,
                                 default=5)
        self.parser.add_argument("--difficult",
                                 type=int,
                                 default=0)
        self.parser.add_argument("--debug",
                                 action="store_true",
                                 help="Debug mode to see augmentation process with images for each step.")
        self.parser.add_argument("--cache",
                                 action="store_true",
                                 help="Create a cache for the augmented dataset.")
        self.parser.add_argument("--patience",
                                 type=int,
                                 default=2500,
                                 help="For multiple chips randomly embedded in the background, if the scale of the"
                                      " chip is too large, the algorithm will be hard to find a suitable location to"
                                      " place all of them. Therefore, a step should be set for the while loop to skip"
                                      " this task if the algorithm takes too long.")
        self.parser.add_argument("--n_chip",
                                 type=int,
                                 nargs=2,
                                 default=[1, 4],
                                 help="Range of number of chips to embed")

        # ====================================== >>> TRAINING <<< ========================================= #
        # Training
        self.parser.add_argument("--mode",
                                 choices=["augmentation", "simple"],
                                 default="augmentation",
                                 help="augmentation: require training_ratio to split the dataset into proper DOTA "
                                      " format \n"
                                      "simple: only augment the dataset with images and labels")
        self.parser.add_argument("--training_ratio",
                                 type=np.int8,
                                 nargs=3,
                                 default=[8, 2, 0],
                                 help="Number ratio of training, validation and testing")

        # ======================================== >>> Cache <<< =========================================== #
        self.parser.add_argument("--cache_name",
                                 type=str,
                                 help="Type [cropped, background] of the cache expected to generate")

    def parse_args(self):
        return self.parser.parse_args()

    @classmethod
    def run_default(cls) -> argparse.Namespace:
        """
        Produce mosaic images, and crop DNA origami chips.
        :return:
        """
        arg_parser = cls()

        return arg_parser.parser.parse_args()

    @classmethod
    def test_args(cls, function: str):
        arg_parser = cls()

        return arg_parser.parser.parse_args(["--function", f"{function}",
                                             "-ip", "data",
                                             "-sp", "test_dataset",
                                             "-dp", "test_dataset",
                                             "--bg_number", '10',
                                             "--aug_number", '10',
                                             ])

    @classmethod
    def test_aug(cls, function):
        arg_parser = cls()

        return arg_parser.parser.parse_args(["--function", f"{function}",
                                             "-ip", "../data",
                                             "-sp", "../test_dataset",
                                             "-dp", "../test_dataset",
                                             "--dataset_name", "test",
                                             "--bg_number", '10',
                                             "--aug_number", '20',
                                             "--cache",
                                             "--n_chip", '5', '10',
                                             "--patience", '2'])

    def find_all_choices(self, param: str) -> List:
        _choices = None

        for action in self.parser._actions:
            if action.dest == param:
                _choices = action.choices

                return _choices

        raise Exception(f"Given parameter {param} is not found.")


if __name__ == "__main__":
    arg_p = ArgumentParser()
    print(arg_p.find_all_choices("label"))
