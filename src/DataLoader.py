from __future__ import annotations

import json
import sqlite3
import argparse
import glob
import os.path
from collections import defaultdict
from typing import List, Dict, Tuple

import cv2
import pandas as pd
from tqdm import tqdm

from src.Background import Background
from src.Component import Component
from src.DNALogging import DNALogging
from src.Image import Image

import logging.config

from src.constant import BACKGROUND

DNALogging.config_logging()
logger = logging.getLogger(__name__)


class DataLoader:
    # TODO: save once loaded images in the format of .npy or pickle for future quick loading
    img_root_path: str  # root of images required to be processes: background mosaics and raw images containing chips
    dataset_root_path: str  # root of prepared images: backgrounds and cropped components

    background_img: Dict[str, List[Background]] = defaultdict(list)
    name_background: Dict[str, Background] = dict()

    component_img: List[Component] = []
    name_component: Dict[str, Component] = dict()

    raw_input_img: List[Image] = []

    @classmethod
    def initialise(cls,
                   img_path: str,
                   dataset_path: str) -> DataLoader:
        data_loader = cls()

        data_loader.img_root_path = img_path
        data_loader.dataset_root_path = dataset_path

        return data_loader

    def load_backgrounds(self, mosaic_size: int):
        """
        Load
        :param mosaic_size: 0 -> prepared backgrounds; else -> background mosaics
        :return:
        """
        try:
            if mosaic_size:
                background_img_paths = sorted(glob.glob(os.path.join(self.img_root_path, "background/*")),
                                              key=lambda x: int(''.join(filter(str.isdigit, x))))
                logger.info(
                    f">>> Load {len(background_img_paths)} background mosaics from {self.img_root_path}/background")
            else:
                background_img_paths = sorted(glob.glob(os.path.join(self.dataset_root_path, "background/*")),
                                              key=lambda x: int(''.join(filter(str.isdigit, x))))
                logger.info(
                    f">>> Load {len(background_img_paths)} existing backgrounds from {self.img_root_path}/background")
        except Exception as e:
            raise Exception(f"Error: Incorrect information {e} given to load image paths")

        for img_path in tqdm(background_img_paths):
            img = Background(img_path, mosaic_size)
            self.background_img[img.texture].append(img)
            self.name_background[img.img_name] = img

        return self

    def load_raw_components(self):
        """
        Load raw images containing DNA origami chips to crop them for further data augmentation.
        :return:
        """

        try:
            raw_img_paths = sorted(glob.glob(os.path.join(self.img_root_path, "component/*")),
                                   key=lambda x: int(''.join(filter(str.isdigit, x))))
        except Exception as e:
            raise Exception(f"Error: Image paths cannot be correctly extracted from given paths with error {e}")

        logger.info(
            f">>> Load {len(raw_img_paths)} raw images for cropping components from {self.img_root_path}/component")

        for img_path in tqdm(raw_img_paths):
            img = Image(img_path)
            self.raw_input_img.append(img)

        return self

    def load_components(self):
        try:
            component_img_paths = sorted(glob.glob(os.path.join(self.dataset_root_path, "component/images/*")),
                                         key=lambda x: int(''.join(filter(str.isdigit, x))))
            component_label_paths = sorted(glob.glob(os.path.join(self.dataset_root_path, "component/labels/*")),
                                           key=lambda x: int(''.join(filter(str.isdigit, x))))
        except Exception as e:
            raise Exception(f"Error: Image paths cannot be correctly extracted from given paths with error {e}")

        n_imgs = len(component_img_paths)
        n_labels = len(component_label_paths)

        if n_imgs != n_labels:
            raise Exception(f"Error: Missing data between images and labels. Should be {n_imgs} = {n_labels}")

        logger.info(f">>> Load {len(component_img_paths)} cropped components from {self.dataset_root_path}/component")
        for img_path, label in tqdm(zip(component_img_paths, component_label_paths)):
            img = Component(img_path, label)
            self.component_img.append(img)
            self.name_component[img.img_name] = img

        return self
