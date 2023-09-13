from __future__ import annotations

import glob
import os.path
import pickle
from collections import defaultdict
from datetime import datetime
from typing import List, Dict

from rich.progress import track

from src.constant import BACKGROUND, MOSAICS, RAW, COMPONENT, CROPPED
from src.Background import Background
from src.Component import Component
from src.DNALogging import DNALogging
from src.Image import Image

import logging.config

from src.utils import mkdir_if_not_exists

DNALogging.config_logging()
logger = logging.getLogger(__name__)


class DataLoader:
    # TODO: refactor all load functions into one
    img_root_path: str  # root of images required to be processes: mosaic mosaics and raw images containing chips
    dataset_root_path: str  # root of prepared images: backgrounds and cropped components
    save_path: str
    cache_save_dir: str

    bg_or_mosc_img: Dict[str, List[Background]] = defaultdict(list)
    name_bg_or_mosc: Dict[str, Background] = dict()

    name_component: Dict[str, Component] = dict()

    name_raw_input: Dict[str, Image] = dict()

    @classmethod
    def initialise(cls,
                   img_path: str = None,
                   dataset_path: str = None,
                   save_path: str = None,
                   cache_save_dir: str = "cache") -> DataLoader:
        data_loader = cls()

        data_loader.img_root_path = img_path
        data_loader.dataset_root_path = dataset_path
        data_loader.save_path = save_path
        data_loader.cache_save_dir = cache_save_dir

        mkdir_if_not_exists(data_loader.save_path)

        return data_loader

    def load_cached_files(self, cache_type: str, cache_path: str):
        if not os.path.exists(cache_path):
            raise FileExistsError(f'Given cache file {cache_path} cannot be found')

        with open(cache_path, "rb") as f:
            logger.info(f">>> Load data from cache {cache_path}")

            if cache_type in [BACKGROUND, MOSAICS]:
                self.name_bg_or_mosc = pickle.load(f)

                for _, bg in self.name_bg_or_mosc.items():
                    self.bg_or_mosc_img[bg.texture].append(bg)
            elif cache_type == RAW:
                self.name_raw_input = pickle.load(f)
            else:
                # cropped
                self.name_component = pickle.load(f)

        logger.info(f">>> Finish loading {cache_type} dataset")

        return self

    def load_backgrounds(self, mosaic_size: int):
        """
        Load
        :param mosaic_size: 0 -> prepared backgrounds; else -> mosaic mosaics
        :return:
        """
        mode_name = "backgrounds" if not mosaic_size else MOSAICS

        logger.info(f">>> Start to load {mode_name}")

        if mosaic_size:
            img_paths = sorted(glob.glob(os.path.join(self.img_root_path, f"{MOSAICS}/*")),
                               key=lambda x: int(''.join(filter(str.isdigit, x))))
            logger.info(f">>> Load {len(img_paths)} background mosaics from {self.img_root_path}/{MOSAICS}")
        else:
            img_paths = sorted(glob.glob(os.path.join(self.dataset_root_path, f"{BACKGROUND}/*")),
                               key=lambda x: int(''.join(filter(str.isdigit, x))))
            logger.info(f">>> Load {len(img_paths)} existing backgrounds from {self.img_root_path}/{BACKGROUND}")

        for img_path in track(img_paths):
            img = Background(img_path, mosaic_size)
            self.bg_or_mosc_img[img.texture].append(img)
            self.name_bg_or_mosc[img.img_name] = img

        logger.info(f">>> Finish loading {mode_name}")

        logger.info(f">>> Start to create the cache for {mode_name}")

        # cache the backgrounds file into pickle for future fast loading
        prefix = "background" if mosaic_size == 0 else "mosaics"

        root_path = self.img_root_path if mosaic_size else self.save_path
        cache_save_path = os.path.join(root_path, self.cache_save_dir)

        mkdir_if_not_exists(cache_save_path)

        cache_save_path = os.path.join(cache_save_path, f'{prefix}_{datetime.now().strftime("%Y_%m_%d_%H:%M")}.pkl')

        with open(cache_save_path, "wb") as f:
            pickle.dump(self.name_bg_or_mosc, f)

        logger.info(f">>> Finish creating the cache file")

        return self

    def load_raw_components(self):
        """
        Load raw images containing DNA origami chips to crop them for further data augmentation.
        :return:
        """
        logger.info(">>> Start to load raw images for cropping components")

        try:
            raw_img_paths = sorted(glob.glob(os.path.join(self.img_root_path, f"{RAW}/*")),
                                   key=lambda x: int(''.join(filter(str.isdigit, x))))
        except Exception as e:
            raise Exception(f"Error: Image paths cannot be correctly extracted from given paths with error {e}")

        logger.info(
            f">>> Load {len(raw_img_paths)} raw images for cropping components from {self.img_root_path}/{RAW}")

        for img_path in track(raw_img_paths):
            img = Image(img_path=img_path)
            self.name_raw_input[img.img_name] = img  # only for creating the cache

        cache_save_path = os.path.join(self.img_root_path, self.cache_save_dir)
        mkdir_if_not_exists(cache_save_path)
        cache_save_path = os.path.join(cache_save_path, f'{RAW}_{datetime.now().strftime("%Y_%m_%d_%H:%M")}.pkl')

        with open(cache_save_path, "wb") as f:
            pickle.dump(self.name_raw_input, f)

        logger.info(">>> Finish creating the cache")

        return self

    def load_cropped_components(self, dir_name: str = CROPPED, images: str = "images", labels: str = "labelTxt"):
        logger.info(">>> Start to load cropped components")

        try:
            component_img_paths = sorted(glob.glob(os.path.join(self.dataset_root_path, f"{dir_name}/{images}/*")),
                                         key=lambda x: int(''.join(filter(str.isdigit, x))))
            component_label_paths = sorted(glob.glob(os.path.join(self.dataset_root_path, f"{dir_name}/{labels}/*")),
                                           key=lambda x: int(''.join(filter(str.isdigit, x))))
        except Exception as e:
            raise Exception(f"Error: Image paths cannot be correctly extracted from given paths with error {e}")

        n_imgs = len(component_img_paths)
        n_labels = len(component_label_paths)

        if n_imgs != n_labels:
            raise Exception(f"Error: Missing data between images and labels. Should be {n_imgs} = {n_labels} \n"
                            f"This is currently required users to firstly label manually all the components by \n"
                            f"some software such as RectLabel.")

        logger.info(f">>> Load {len(component_img_paths)} components from {self.dataset_root_path}/{dir_name}")

        for img_path, label in track(zip(component_img_paths, component_label_paths)):
            img = Component(img_path=img_path, label_path=label)
            self.name_component[img.img_name] = img

        logger.info(">>> Finish loading components")

        logger.info(">>> Start to create the cache for components")

        # cache the cropped origami file into pickle for future fast loading
        cache_save_path = os.path.join(self.save_path, self.cache_save_dir)
        mkdir_if_not_exists(cache_save_path)
        cache_save_path = os.path.join(cache_save_path, f'{CROPPED}_{datetime.now().strftime("%Y_%m_%d_%H:%M")}.pkl')

        with open(cache_save_path, "wb") as f:
            pickle.dump(self.name_component, f)

        logger.info(">>> Finish creating the cache")

        return self
