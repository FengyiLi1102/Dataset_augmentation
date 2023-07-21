from __future__ import annotations

import argparse
import json
import os.path
from typing import Dict, List

import numpy as np

from src.Background import Background
from src.DNALogging import DNALogging
import random

import logging

from src.DatabaseManager import DatabaseManager
from src.Task import Task
from src.constant import BACKGROUND, COMPONENT, CROPPED
from src.utils import mkdir_if_not_exists

DNALogging.config_logging()
logger = logging.getLogger(__name__)


class TaskAssigner:
    # General
    save_path: str
    mode: str
    cache_save_dir: str

    # Backgrounds
    background_task_pipeline: Dict[str, List] = dict()

    """
    Each mosaic mosaic has flip or 180 degree rotation operations.
    flip: vertical - 'v' / horizontal - 'h'
    rotation: 180
    """
    operations_backgrounds = ['v', 'h', 180]
    num_per_side: int
    num_mosaic_in_background: int
    kernel_size: int

    # Components
    config: Dict = dict()
    cropping_inflation: int

    # Augmentation
    dataset_name: str = "augmented"

    # [required_scale, background_texture, component_texture, position, flip, rotation]
    initial_scale: float
    augmentation_task_pipeline: List[Task]

    height_domain: int
    width_domain: int

    label: str
    difficult: int

    @classmethod
    def background_task(cls,
                        args: argparse.Namespace):
        if int(sum(args.bg_ratio)) != 10:
            raise Exception(
                f"Error: Given ratios are not valid. Sum of them should be 10 instead of {int(sum(args.bg_ratio))}.")

        task_assigner = cls()

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
        task_assigner = cls()

        task_assigner.save_path = args.save_path
        task_assigner.cropping_inflation = args.inflation
        task_assigner.cache_save_dir = args.cache_save_dir

        # load configuration parameters for cropping origami chips to make component
        with open(args.config_path, "r") as config_file:
            config = json.load(config_file)
            task_assigner.config = config

        return task_assigner

    @classmethod
    def augmented_task(cls, args: argparse.Namespace, db: DatabaseManager) -> TaskAssigner:
        task_assigner = cls()

        task_assigner.mode = args.mode
        task_assigner.dataset_name = args.dataset_name
        task_assigner.save_path = args.save_path
        task_assigner.label = args.label
        task_assigner.difficult = args.difficult
        task_assigner.cache_save_dir = args.cache_save_dir

        # resize the components into a suitable size compared with the existing backgrounds
        task_assigner.initial_scale = args.initial_scale

        task_assigner.augmentation_task_pipeline = Task.initialise_list(args.mode, args.aug_number,
                                                                        args.training_ratio)

        # new scales for the origami
        min_scale, max_scale = args.scale_range
        num_interval = int((max_scale - 1) / args.scale_increment + 1)

        # flip
        flip_options = ['v', 'h', 'n']

        # rotation
        max_range = 180 + args.rotation_increment

        for task in task_assigner.augmentation_task_pipeline:
            # rescale
            if args.scaling_mode == "fixed":
                task.required_scale = 1
            elif args.scaling_mode == "random":
                task.required_scale = random.choice(np.linspace(min_scale, max_scale, num=num_interval))
            elif args.scaling_mode in ["larger", "smaller"]:
                task.required_scale = TaskAssigner.biased_random(args.scaling_mode, min_scale, max_scale, num_interval)
            else:
                raise Exception(f"Error: Incorrect scaling model {args.scaling_mode} is given.")

            # Choose Background and Component image in id with textures
            TaskAssigner.choose_texture(BACKGROUND, args.backgrounds, task, db)
            TaskAssigner.choose_texture(CROPPED, args.components, task, db)

            # flip
            task.flip = random.choice(flip_options)

            # rotation
            task.rotation = random.choice(range(-180, max_range, args.rotation_increment))

            # Check if equivalent augmentation exists with same component, similar scale and special combination of
            # flip and rotation. This is equivalent to each other if:
            # h_flip + positive rotate (D) = v_flip + negative rotate (180 - D)
            # where D > 0
            while True:
                if TaskAssigner.equivalence_check(task_assigner.augmentation_task_pipeline, args.scale_increment):
                    # same or equivalent
                    # here we chang the rotation to avoid
                    task.rotation = random.choice(range(-180, max_range, args.rotation_increment))
                else:
                    break

        return task_assigner

    @staticmethod
    def equivalence_check(pipeline: List[Task], scale_interval: float):
        new_added_task = pipeline[-1]

        # [0r, 2c, 4f, 5r]
        for previous_task in pipeline[: -1]:
            # same component and similar scale
            if previous_task.component_id == new_added_task.component_id and abs(
                    previous_task.required_scale - new_added_task.required_scale) <= 2 * scale_interval:
                # exactly same flip + rotate
                if previous_task.flip == new_added_task.flip and previous_task.rotation == new_added_task.rotation:
                    return True
                elif ord(previous_task.flip) + ord(new_added_task.flip) == 222 and int(abs(
                        previous_task.rotation - new_added_task.rotation)) == 180:
                    # special combination
                    return True
                else:
                    return False
            else:
                return False

    @staticmethod
    def choose_texture(table_name: str, given: str, task: Task, db: DatabaseManager):
        ids = db.select_table(table_name).get_unique_values("id")
        target_col = "Texture" if table_name == BACKGROUND else "Morphology"

        if given == "random":
            if table_name == BACKGROUND:
                task.background_id = random.choice(ids)
            else:
                task.component_id = random.choice(ids)
        else:
            grouped_ids = db.select_table(table_name).group_by_column("id", target_col)

            if table_name == BACKGROUND:
                task.background_id = random.choice(grouped_ids[given])
            else:
                task.component_id = random.choice(grouped_ids[given])

    @staticmethod
    def biased_random(direction: str, min_scale: float, max_scale: float, num: int) -> float:
        if direction == "larger":
            threshold = 0.7
        else:
            threshold = 0.3

        if random.random() < threshold:
            return random.choice(np.linspace(1, max_scale, num=num))
        else:
            return random.choice(np.linspace(min_scale, 1, num=num))
