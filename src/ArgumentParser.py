import argparse

import numpy as np


class ArgumentParser:
    parser: argparse.ArgumentParser

    def __init__(self):
        self.parser = argparse.ArgumentParser("Dataset augmentation parameters")

        self.parser.add_argument("--function", "-f",
                                 choices=["produce_chip", "run"],
                                 default="run")

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
                                 help="Processed images including ready-for-training data")

        # For generating fake backgrounds without origami chips
        self.parser.add_argument("--init_background", "-init_b",
                                 action="store_true",
                                 help="Initialise backgrounds from background mosaics")
        self.parser.add_argument("--b_size",
                                 type=int,
                                 default=320,
                                 help="Size of the background mosaics")
        self.parser.add_argument("--number", "-n",
                                 type=np.int16,
                                 default=16)
        self.parser.add_argument("--ratio", "-r",
                                 type=np.int8,
                                 nargs=4,
                                 default=[5, 2, 2, 1],
                                 help="Number ratio among clean : noisyL : noisy : messy")
        self.parser.add_argument("--size", "-s",
                                 type=np.int16,
                                 default=2560)
        self.parser.add_argument("--kernel_size", "-ks",
                                 type=int,
                                 default=5,
                                 help="Averaging kernel size for decease seam visibility")

        # For cropping DNA origami from given raw images to make component
        self.parser.add_argument("--config_path", "-cp",
                                 type=str,
                                 default=r"config/params.json")
        self.parser.add_argument("--inflation", "-infl",
                                 type=int,
                                 default=300,
                                 help="Extra pixels captured away from the detected DNA origami")

        # For producing augmentation images
        self.parser.add_argument("--initial_scale", "-is",
                                 type=float,
                                 default=1/6,
                                 help="Resize the components with a suitable size compared to backgrounds")
        self.parser.add_argument("--scale_range", "-sr",
                                 nargs=2,
                                 type=float,
                                 default=[0.4, 1.6],
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
                                 default="random")
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

        # Training
        self.parser.add_argument("--training_ratio",
                                 type=np.int8,
                                 nargs=3,
                                 default=[6, 2, 2])


    @classmethod
    def run_default(cls) -> argparse.Namespace:
        """
        Produce background images, and crop DNA origami chips.
        :return:
        """
        arg_parser = cls()

        return arg_parser.parser.parse_args()

    @classmethod
    def test_args(cls):
        arg_parser = cls()

        return arg_parser.parser.parse_args(["-init_b",
                                             "-ip", "../data",
                                             "-sp", "../dataset",
                                             "-cp", "../config/params.json",
                                             "-dp", "../dataset",
                                             "-ks", '10',
                                             "--aug_number", '5'])

    @classmethod
    def test_aug(cls):
        arg_parser = cls()

        return arg_parser.parser.parse_args()
