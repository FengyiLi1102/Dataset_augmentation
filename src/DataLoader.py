from __future__ import annotations

import glob
import os.path
import pickle
from collections import defaultdict
from datetime import datetime
from typing import List, Dict

from tqdm import tqdm

from src.Background import Background
from src.Component import Component
from src.DNALogging import DNALogging
from src.Image import Image

import logging.config

from src.utils import mkdir_if_not_exists

DNALogging.config_logging()
logger = logging.getLogger(__name__)


class DataLoader:
    # TODO: save once loaded images in the format of .npy or pickle for future quick loading
    img_root_path: str  # root of images required to be processes: background mosaics and raw images containing chips
    dataset_root_path: str  # root of prepared images: backgrounds and cropped components
    save_path: str

    background_img: Dict[str, List[Background]] = defaultdict(list)
    name_background: Dict[str, Background] = dict()

    component_img: List[Component] = []
    name_component: Dict[str, Component] = dict()

    raw_input_img: List[Image] = []

    @classmethod
    def initialise(cls,
                   img_path: str = None,
                   dataset_path: str = None,
                   save_path: str = None) -> DataLoader:
        data_loader = cls()

        data_loader.img_root_path = img_path
        data_loader.dataset_root_path = dataset_path
        data_loader.save_path = save_path

        return data_loader

    def load_cached_files(self, cache_type: str, cache_path: str):
        if not os.path.exists(cache_path):
            raise FileExistsError(f'Given cache file {cache_path.split("/")[-1]} cannot be found')

        with open(cache_path, "rb") as f:
            logger.info(f">>> Load data from cache {cache_path.split('/')[-1]}")

            if cache_type == "fake_bg":
                self.background_img, self.name_background = pickle.load(f)
            elif cache_type == "mosaic":
                self.background_img = pickle.load(f)
            elif cache_type == "raw_img":
                self.raw_input_img = pickle.load(f)
            else:
                self.component_img, self.name_component = pickle.load(f)

        logger.info(f">>> Finish loading dataset")

        return self

    def load_backgrounds(self, mosaic_size: int):
        """
        Load
        :param mosaic_size: 0 -> prepared backgrounds; else -> background mosaics
        :return:
        """
        mode_name = "backgrounds" if not mosaic_size else "mosaics"

        logger.info(f">>> Start to load {mode_name}")

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

            if mosaic_size == 0:
                # only used for augmentation
                self.name_background[img.img_name] = img

        logger.info(f">>> Finish loading {mode_name}")

        logger.info(f">>> Start to create the cache for {mode_name}")

        # cache the backgrounds file into pickle for future fast loading
        prefix = "ready_backgrounds" if mosaic_size == 0 else "mosaics"
        mkdir_if_not_exists(self.save_path)
        cache_save_path = os.path.join(self.save_path, f'{prefix}_{datetime.now().strftime("%Y_%m_%d_%H:%M")}.pkl')

        with open(cache_save_path, "wb") as f:
            if mosaic_size == 0:
                pickle.dump((self.background_img, self.name_background), f)
            else:
                pickle.dump(self.background_img, f)

        logger.info(f">>> Finish creating the cache file")

        return self

    def load_raw_components(self):
        """
        Load raw images containing DNA origami chips to crop them for further data augmentation.
        :return:
        """
        logger.info(">>> Start to load raw images for cropping components")

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

        # cache the raw images file into pickle for future fast loading
        mkdir_if_not_exists(self.save_path)
        cache_save_path = os.path.join(self.save_path, f'rawComponent_{datetime.now().strftime("%Y_%m_%d_%H:%M")}.pkl')

        with open(cache_save_path, "wb") as f:
            pickle.dump(self.raw_input_img, f)

        return self

    def load_components(self):
        logger.info(">>> Start to load cropped components")

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
            raise Exception(f"Error: Missing data between images and labels. Should be {n_imgs} = {n_labels} \n"
                            f"This is currently required users to firstly label manually all the components by \n"
                            f"some software such as RectLabel.")

        logger.info(f">>> Load {len(component_img_paths)} cropped components from {self.dataset_root_path}/component")
        for img_path, label in tqdm(zip(component_img_paths, component_label_paths)):
            img = Component(img_path, label)
            self.component_img.append(img)
            self.name_component[img.img_name] = img

        logger.info(">>> Finish loading cropped components")

        logger.info(">>> Start to create the cache for cropped components")

        # cache the cropped origami file into pickle for future fast loading
        mkdir_if_not_exists(self.save_path)
        cache_save_path = os.path.join(self.save_path, f'cropped_{datetime.now().strftime("%Y_%m_%d_%H:%M")}.pkl')

        with open(cache_save_path, "wb") as f:
            pickle.dump((self.component_img, self.name_component), f)

        logger.info(">>> Finish creating the cache")

        return self
