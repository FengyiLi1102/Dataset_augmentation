from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple, Union

import numpy as np
from rich.progress import track

from src.Background import Background
from src.DNALogging import DNALogging
import random

import logging

from src.DatabaseManager import DatabaseManager
from src.Task import Task
from src.constant import BACKGROUND, CROPPED, GENERATE_FAKE_BACKGROUND, CROP_ORIGAMI, DNA_ORIGAMI, AUGMENTED, TRAINING, \
    VALIDATION, TESTING, V, H, N, SIMPLE, GENERATE_EXTENDED_STRUCTURE
from src.utils import mkdir_if_not_exists, ratio_to_number
from src.typeHint import PointImageType

DNALogging.config_logging()
logger = logging.getLogger(__name__)


class TaskAssigner:
    def __init__(self, task_type: str):
        # General
        self.save_path: str = ""
        self.cache_save_dir: str = ""

        if task_type == GENERATE_FAKE_BACKGROUND:
            # Backgrounds
            self.background_task_pipeline: Dict[str, List] = dict()

            """
            Each mosaic mosaic has flip or 180 degree rotation operations.
            flip: vertical - 'v' / horizontal - 'h'
            rotation: 180
            """
            self.operations_backgrounds = [V, H, 180]
            self.num_per_side: int = 0
            self.num_mosaic_in_background: int = 0
            self.kernel_size: int = 0
        elif task_type == CROP_ORIGAMI:
            # Components
            self.config: Dict = dict()
            self.cropping_inflation: int = 0
        elif task_type in [GENERATE_EXTENDED_STRUCTURE, AUGMENTED]:
            self.dataset_name: str = task_type
            self.initial_scale: float = .0
            self.augmentation_task_pipeline: List[Task] = []
            self.cache: bool = False
            self.expected_num: int = 0
            self.difficult: int = 0
            self.decimal_place: int = 2
            self.mode: str = SIMPLE

            if task_type == GENERATE_EXTENDED_STRUCTURE:
                self.stitch_size: PointImageType
                self.n_stitch: int = 0
                self.gap_w: float = .0
                self.gap_h: float = .0
                self.stitch_size: Tuple | None = None
            else:
                self.component_name: str = CROPPED

                # [required_scale, background_texture, component_texture, position, flip, rotation]
                self.height_domain: int = 0
                self.width_domain: int = 0

                self.patience: int = 0
                self.max_try: int = 0
                self.n_split: List[int] = []

                self.patience: int = 100

    @classmethod
    def background_task(cls,
                        args: argparse.Namespace):
        if int(sum(args.bg_ratio)) != 10:
            raise Exception(
                f"Error: Given ratios are not valid. Sum of them should be 10 instead of {int(sum(args.bg_ratio))}.")

        task_assigner = cls(args.function)

        task_assigner.save_path = args.save_path
        mkdir_if_not_exists(task_assigner.save_path)

        num_for_textures = [int(args.bg_number * r / 10) for r in args.bg_ratio]
        task_assigner.num_per_side = int(args.background_size / args.mosaic_size)
        task_assigner.num_mosaic_in_background = task_assigner.num_per_side ** 2
        task_assigner.kernel_size = args.kernel_size
        task_assigner.cache_save_dir = args.cache_save_dir

        logger.info(">>> Start to assign tasks for each mosaics to generate backgrounds")
        # Assign tasks for producing backgrounds
        for idx, texture in enumerate(Background.background_textures):
            logger.info(f">>> Assign {num_for_textures[idx]} tasks for generating {texture} backgrounds")
            task_assigner.background_task_pipeline[texture] = [
                random.choices(task_assigner.operations_backgrounds, k=task_assigner.num_mosaic_in_background)
                for _ in range(num_for_textures[idx])
            ]

        return task_assigner

    @classmethod
    def component_task(cls, args: argparse.Namespace) -> TaskAssigner:
        task_assigner = cls(args.function)

        task_assigner.save_path = args.save_path
        task_assigner.cropping_inflation = args.inflation
        task_assigner.cache_save_dir = args.cache_save_dir

        # load configuration parameters for cropping origami chips to make component
        with open(args.config_path, "r") as config_file:
            config = json.load(config_file)
            task_assigner.config = config

        return task_assigner

    @classmethod
    def extended_structure_task(cls, args: argparse.Namespace, db: DatabaseManager) -> TaskAssigner:
        """
        For augmenting structures with chips stitched together, the rotation for each chip is the same, and the size
        is very close because the original size of each chip is not the same and the aspect ratio should be also kept.
        The augmentation only allows to produce all structures with the same number of stitched chips.
        :param args:
        :param db:
        :return:
        """
        if np.any([args.stitch[0] <= 0, args.stitch[1] <= 0]):
            raise Exception(f"Error: Incorrect given stitching size")

        n_chip_l, n_chip_h = args.n_chip
        if n_chip_l <= 0 or n_chip_h <= 0 or n_chip_h < n_chip_l:
            raise Exception(f"Error: Incorrect range of chip number is given.")

        if args.scale_increment <= 0 or args.rotation_increment <= 0:
            raise ValueError(f"Given invalid increment for scale or rotation or both two.")

        task_assigner = cls(args.function)

        task_assigner.stitch_size = tuple(args.stitch)
        task_assigner.n_stitch = int(args.stitch[0] * args.stitch[1])  # number of individual chips per structure
        task_assigner.dataset_name = args.dataset_name

        task_assigner.cache = args.cache
        task_assigner.difficult = args.difficult
        task_assigner.expected_num = args.aug_number
        task_assigner.decimal_place = TaskAssigner.__num_of_decimal_place(args.scale_increment)
        task_assigner.gap_w = args.gap_w
        task_assigner.gap_h = args.gap_h

        task_assigner.save_path = args.save_path
        task_assigner.cache_save_dir = args.cache_save_dir

        # resize the component image into a fixed size based on the origami size
        # default value is 640
        task_assigner.initial_scale = args.initial_scale

        task_assigner.augmentation_task_pipeline, _ = Task.initialise_list(SIMPLE, args.aug_number)

        # ids for components
        # FIXME
        ids_component = TaskAssigner.__ids_texture_morphology(args.components, CROPPED, db)

        for _, task in track(enumerate(task_assigner.augmentation_task_pipeline)):
            # All chips in a structure share the same orientation and very similar size.
            # Choose Components by id with morphology
            # task.component_id = np.asarray(random.choices(ids_component, k=task_assigner.n_stitch)).reshape(
            #     task_assigner.stitch_size)
            task.component_id = np.asarray([random.choice(ids_component)] * task_assigner.n_stitch).reshape(
                task_assigner.stitch_size)

            # flip
            task.flip = np.asarray(random.choices([V, H, N], k=task_assigner.n_stitch)).reshape(
                task_assigner.stitch_size)

        return task_assigner

    @classmethod
    def augmented_task(cls, args: argparse.Namespace, db: DatabaseManager) -> TaskAssigner:
        task_assigner = cls(args.function)

        task_assigner.dataset_name = args.dataset_name
        n_chip_l, n_chip_h = args.n_chip

        task_assigner.cache = args.cache
        task_assigner.difficult = args.difficult
        task_assigner.patience = args.patience
        task_assigner.expected_num = args.aug_number

        # maximum number of attempt to finish the target number of tasks
        task_assigner.max_try = int(args.aug_number * (args.buffer + 1))

        task_assigner.mode = args.mode
        task_assigner.save_path = args.save_path
        task_assigner.cache_save_dir = args.cache_save_dir
        task_assigner.component_name = args.component_name

        # resize the components into a suitable size compared with the existing backgrounds
        task_assigner.initial_scale = args.initial_scale

        task_assigner.augmentation_task_pipeline, n_split = Task.initialise_list(args.mode, task_assigner.max_try,
                                                                                 args.training_ratio)

        if n_split:
            task_assigner.n_split = ratio_to_number(args.training_ratio, task_assigner.expected_num)

        # magnify or shrink the origami
        min_scale, max_scale = args.scale_range
        num_interval = int((max_scale - 1) / args.scale_increment + 1)

        # flip
        flip_options = [V, H, N]

        # rotation
        max_range = 180 + args.rotation_increment

        for idx, task in track(enumerate(task_assigner.augmentation_task_pipeline)):
            # Choose Background image by id with textures
            TaskAssigner.choose_texture(BACKGROUND, args.backgrounds, task, db)

            # rescale
            # for each AFM image, the scale is a fixed value covering the whole image, and there should not be any same
            # kind of objects with dramatically different scale but can accept tiny change between +/- 0.2
            # TODO: if the number of chips is larger -> the scale of the chip should be smaller -> able to fill
            # TODO: in the background
            if args.scaling_mode == "fixed":
                required_scale = 1
            elif args.scaling_mode == "random":
                # if args.n_chip > 3:
                #     adapted_max_scale = TaskAssigner.adaptive_scale_generator(task_assigner.n_chip,
                #                                                               task_assigner.initial_scale, min_scale,
                #                                                               max_scale)
                required_scale = random.choice(np.linspace(min_scale, max_scale, num=num_interval))
            elif args.scaling_mode in ["larger", "smaller"]:
                required_scale = TaskAssigner.biased_random(args.scaling_mode, min_scale, max_scale, num_interval)
            else:
                raise Exception(f"Error: Incorrect scaling model {args.scaling_mode} is given.")

            task.required_scale = required_scale

            # generate number of chips in one background
            n_chip = random.randint(n_chip_l, n_chip_h)
            task.n_chip = n_chip

            for n in range(n_chip):
                # Choose Component image by id with textures
                TaskAssigner.choose_texture(task_assigner.component_name, args.components, task, db)

                # flip
                task.flip.append(random.choice(flip_options))

                # rotation
                task.rotation.append(random.choice(range(-180, max_range, args.rotation_increment)))

                """
                Check if equivalent augmentation exists with same component, similar scale and special combination of
                flip and rotation. This is equivalent to each other if:
                h_flip + positive rotate (D) = v_flip + negative rotate (180 - D)
                where D > 0
                """
                while True:
                    if TaskAssigner.equivalence_check_for_one_chip(task_assigner.augmentation_task_pipeline[: idx + 1],
                                                                   args.scale_increment):
                        # same or equivalent
                        # here we change the rotation to avoid
                        task.rotation[-1] = random.choice(range(-180, max_range, args.rotation_increment))
                    else:
                        break

        return task_assigner

    @staticmethod
    def equivalence_check_for_one_chip(pipeline: List[Task], scale_interval: float):
        if len(pipeline) == 1:
            return False

        new_added_task = pipeline[-1]

        # [0r, 2c, 4f, 5r]
        for previous_task in pipeline[: -1]:
            # same component and similar scale
            if previous_task.component_id[0] == new_added_task.component_id[0] and abs(
                    previous_task.required_scale - new_added_task.required_scale) <= 2 * scale_interval:
                # exactly same flip + rotate
                if previous_task.flip[0] == new_added_task.flip[0] and previous_task.rotation[0] == \
                        new_added_task.rotation[0]:
                    return True
                elif previous_task.flip[0] + new_added_task.flip[0] == 1 and int(abs(
                        previous_task.rotation[0] - new_added_task.rotation[0])) == 180:
                    # special combination
                    return True
                else:
                    return False
            else:
                return False

    @staticmethod
    def __ids_texture_morphology(requirement: str,
                                 table_name: str,
                                 db: DatabaseManager) -> List[int]:
        ids = db.select_table(table_name).get_unique_values("id")
        target_col = "Texture" if table_name == BACKGROUND else "Morphology"

        if requirement == "random":
            return ids
        else:
            grouped_ids = db.select_table(table_name).group_by_column("id", target_col)
            return grouped_ids[requirement]

    @staticmethod
    def __rescale_generator(scaling_mode: str,
                            min_scale: float,
                            max_scale: float,
                            num_interval: int,
                            decimal_places: int = 1) -> float:
        if scaling_mode == "fixed":
            res = 1.0
        elif scaling_mode == "random":
            res = random.choice(np.linspace(min_scale, max_scale, num=num_interval))
        elif scaling_mode in ["larger", "smaller"]:
            res = TaskAssigner.biased_random(scaling_mode, min_scale, max_scale, num_interval)
        else:
            raise Exception(f"Error: Incorrect scaling model {scaling_mode} is given.")

        return round(res, decimal_places)

    @staticmethod
    def __number_chip_generator(init_scale: float,
                                rescale: float,
                                n_stitch: int,
                                range_l: int,
                                range_h: int) -> int:
        final_scale = (init_scale ** 2) * rescale * n_stitch
        ideal_volume = int(1 / final_scale)

        if ideal_volume < 1:
            # cannot to fill the background with even one origami
            return -1
        else:
            if ideal_volume < range_l:
                return random.randint(1, ideal_volume)
            elif range_h > ideal_volume:
                return random.randint(range_l, ideal_volume)
            else:
                return random.randint(range_l, range_h)

    @staticmethod
    def biased_random(direction: str,
                      min_scale: float,
                      max_scale: float,
                      num: int) -> float:
        if direction == "larger":
            threshold = 0.7
        else:
            threshold = 0.3

        if random.random() < threshold:
            return random.choice(np.linspace(1, max_scale, num=num))
        else:
            return random.choice(np.linspace(min_scale, 1, num=num))

    @staticmethod
    def __num_of_decimal_place(value: float | int) -> int:
        value_str = str(value)
        decimal_point_index = value_str.find('.')

        if decimal_point_index == -1:
            return 0

        return len(value_str) - decimal_point_index - 1

    @staticmethod
    def adaptive_scale_generator(n_chip: int,
                                 initial_scale: float,
                                 min_s: float,
                                 max_s: float) -> Tuple[float, float]:
        """
        When multiple chips are embedded in the background, if the final scale of the chip is too larger, there will be
        hardly possible to find a location to place the chip. Therefore, the scale range for the chip should be modified
        based on the final number of chips in the background.
        :param n_chip:
        :param initial_scale:
        :param min_s:rr
        :param max_s:
        :return:
        """
        expected_chip_volume = int((1 / initial_scale) ** 2)

        # this is not mathematical proven but just an assumption
        # chip can rotate, and it may be considered as a circle with the radius as the half of its diagonal
        safe_chip_volume = expected_chip_volume / 2

        abs_max_s = round((1 / 8) / (initial_scale ** 2), 1)  # reasonable maximum scale
        abs_min_s = 0.3

        if n_chip <= round(safe_chip_volume * 0.3):
            r_min_s = min_s if min_s >= abs_min_s else abs_max_s
            r_max_s = abs_max_s
            return r_min_s, r_max_s
        elif n_chip <= round(safe_chip_volume * 0.6):
            r_max_s = round(1 / round(safe_chip_volume * 0.6) * safe_chip_volume, 1)
            return abs_min_s, r_max_s

    @staticmethod
    def choose_texture(table_name: str, given: str, task: Task, db: DatabaseManager):
        ids = db.select_table(table_name).get_unique_values("id")
        target_col = "Texture" if table_name == BACKGROUND else "Morphology"

        if given == "random":
            if table_name == BACKGROUND:
                task.background_id = random.choice(ids)
            else:
                task.component_id.append(random.choice(ids))
        else:
            grouped_ids = db.select_table(table_name).group_by_column("id", target_col)

            if table_name == BACKGROUND:
                task.background_id = random.choice(grouped_ids[given])
            else:
                task.component_id.append(random.choice(grouped_ids[given]))
