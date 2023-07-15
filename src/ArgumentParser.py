import argparse

import numpy as np


class ArgumentParser:
    parser: argparse.ArgumentParser

    def __init__(self):
        self.parser = argparse.ArgumentParser("Dataset augmentation parameters")

        # Functionality
        self.parser.add_argument("--function", "-f",
                                 choices=["crop_origami", "run", "generate_fake_backgrounds"],
                                 default="run")

        # General parameters
        self.parser.add_argument("--img_path", "-ip",
                                 type=str,
                                 default=r"data",
                                 help="Directory of raw background and component images")
        self.parser.add_argument("--save_path", "-sp",
                                 type=str,
                                 default=r"dataset",
                                 help="Directory to save processed images")
        self.parser.add_argument("--dataset_path", "-dp",
                                 type=str,
                                 default=r"dataset",
                                 help="Processed images including fake backgrounds and cropped components")

        # For generating fake backgrounds without origami chips
        self.parser.add_argument("--cache_bg_type",
                                 type=str,
                                 choices=["mosaic", "fake_bg", "none"],
                                 default="none",
                                 help="mosaic: prepared background mosaics; \n"
                                      "fake_bg: generated fake backgrounds; \n"
                                      "none: no cache file")
        self.parser.add_argument("--cache_bg_path",
                                 type=str)
        self.parser.add_argument("--init_background", "-init_bg",
                                 action="store_true",
                                 help="Initialise backgrounds from background mosaics")
        self.parser.add_argument("--background_size", "-bg_size",
                                 type=np.int16,
                                 default=2560,
                                 help="Expected generated fake background size")
        self.parser.add_argument("--mosaic_size", "-ms_size",
                                 type=int,
                                 default=320,
                                 help="Size of the background mosaics")
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

        # For cropping DNA origami from given raw images to make component
        self.parser.add_argument("--cache_chip_type",
                                 type=str,
                                 choices=["raw_img", "cropped", "none"],
                                 default="none",
                                 help="raw_img: raw AFM images or cropping origami; \n"
                                      "cropped: cropped origami chip; \n"
                                      "none: no cache file")
        self.parser.add_argument("--cache_chip_path",
                                 type=str)
        self.parser.add_argument("--config_path", "-cp",
                                 type=str,
                                 default=r"config/params.json")
        self.parser.add_argument("--inflation", "-infl",
                                 type=int,
                                 default=300,
                                 help="Extra pixels captured away from the detected DNA origami")

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
                                 default=[0.6, 2.0],
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
                                 default=1000)
        self.parser.add_argument("--backgrounds",
                                 choices=["random", "messy", "clean", "noisy", "noisyL"],
                                 default="noisy")
        self.parser.add_argument("--components",
                                 choices=["random", "irregular", "regular", "broken", "sheared"],
                                 default="random")
        self.parser.add_argument("--rotation_increment",
                                 type=int,
                                 default=5)
        self.parser.add_argument("--label",
                                 type=str,
                                 default="DNA-origami")
        self.parser.add_argument("--difficult",
                                 type=int,
                                 default=0)
        self.parser.add_argument("--debug",
                                 action="store_true")

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

    def parse_args(self):
        return self.parser.parse_args()

    @classmethod
    def run_default(cls) -> argparse.Namespace:
        """
        Produce background images, and crop DNA origami chips.
        :return:
        """
        arg_parser = cls()

        return arg_parser.parser.parse_args()

    @classmethod
    def test_args(cls, function: str):
        arg_parser = cls()

        return arg_parser.parser.parse_args(["--function", f"{function}",
                                            "-init_b",
                                             "-ip", "data",
                                             "-sp", "test_dataset",
                                             "-cp", "config/params.json",
                                             "-dp", "dataset",
                                             "--bg_number", '10',
                                             "--aug_number", '10',
                                             "--mode", "augmentation"])

    @classmethod
    def test_aug(cls):
        arg_parser = cls()

        return arg_parser.parser.parse_args()
