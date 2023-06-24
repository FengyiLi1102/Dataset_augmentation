import argparse

import numpy as np


class ArgumentParser:
    parser: argparse.ArgumentParser

    def __init__(self):
        self.parser = argparse.ArgumentParser("Dataset augmentation parameters")

        self.parser.add_argument("--img_path", "-ip",
                                 type=str,
                                 default=r"data")
        self.parser.add_argument("--save_path", "-sp",
                                 type=str,
                                 default=r"dataset")

        # For generate fake backgrounds without origami chips
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

    @classmethod
    def default(cls) -> argparse.Namespace:
        arg_parser = cls()

        return arg_parser.parser.parse_args()

    @classmethod
    def arg_background(cls):
        arg_parser = cls()

        return arg_parser.parser.parse_args(["-init_b", "-sp", "../test_dataset"])
