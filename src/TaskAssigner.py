from __future__ import annotations

import argparse
import json
import math
import os.path
from typing import Dict, List, Tuple

import numpy as np

from src.Background import Background
from src.DNALogging import DNALogging
from src.DataLoader import DataLoader
from tqdm import tqdm
import random

import logging

from src.DatabaseManager import DatabaseManager
from src.Task import Task
from src.constant import BACKGROUND, COMPONENT, AUGMENTATION

DNALogging.config_logging()
logger = logging.getLogger(__name__)


class TaskAssigner:
    # General
    save_path: str

    # Backgrounds
    background_task_pipeline: Dict[str, List] = dict()

    """
    Each background mosaic has flip or 180 degree rotation operations.
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
        if int(sum(args.ratio)) != 10:
            raise Exception(
                f"Error: Given ratios are not valid. Sum of them should be 10 instead of {int(sum(args.ratio))}.")

        task_assigner = cls()

        task_assigner.save_path = args.save_path
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        num_for_textures = [int(args.number * r / 10) for r in args.ratio]
        task_assigner.num_per_side = int(args.size / args.b_size)
        task_assigner.num_mosaic_in_background = task_assigner.num_per_side ** 2
        task_assigner.kernel_size = args.kernel_size

        logger.info(">>> Start to assign tasks for each mosaics to generate backgrounds")
        # Assign tasks for producing backgrounds
        for idx, texture in enumerate(Background.background_textures):
            task_assigner.__background_assign_helper(texture, task_assigner.num_mosaic_in_background,
                                                     num_for_textures[idx])

        return task_assigner

    def __background_assign_helper(self, texture: str,
                                   n_mo_backgrounds: int,
                                   n_produced: int):
        # Assign tasks for producing backgrounds
        print(f">>> Assign {n_produced} tasks for generating {texture} backgrounds")
        self.background_task_pipeline[texture] = [
            random.choices(self.operations_backgrounds, k=n_mo_backgrounds) for _ in range(n_produced)]

    @classmethod
    def component_task(cls, args: argparse.Namespace) -> TaskAssigner:
        task_assigner = cls()

        task_assigner.save_path = args.save_path
        task_assigner.cropping_inflation = args.inflation

        # load configuration parameters for cropping origami chips to make component
        with open(args.config_path, "r") as config_file:
            config = json.load(config_file)
            task_assigner.config = config

        return task_assigner

    @classmethod
    def augmented_task(cls, args: argparse.Namespace, db: DatabaseManager) -> TaskAssigner:
        task_assigner = cls()

        task_assigner.save_path = args.save_path
        task_assigner.label = args.label
        task_assigner.difficult = args.difficult

        # resize the components into a suitable size compared with the existing backgrounds
        task_assigner.initial_scale = args.initial_scale

        task_assigner.augmentation_task_pipeline = Task.initialise_list(AUGMENTATION, args.aug_number)

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
            TaskAssigner.choose_texture(COMPONENT, args.components, task, db)

            # flip
            task.flip = random.choice(flip_options)

            # position
            # task.position = task_assigner.__random_position(task, db, args)

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

    # @staticmethod
    # def random_position(task: Task,
    #                     db: DatabaseManager,
    #                     args: argparse.Namespace) -> Tuple[int, int]:
    #     img_height, img_width = db.select_table(COMPONENT).query_data(f"id = {task.component_id}", ["Height", "Width"])
    #     background_size = args.size
    #
    #     scaled_height = background_size * args.initial_scale
    #     adjusted_scale = scaled_height / img_height
    #     scaled_width = int(adjusted_scale * img_width)  # common scaled width
    #     scaled_height = int(scaled_height)
    #     width = int(scaled_width * task.required_scale)  # adjusted-scaled width
    #     height = int(scaled_height * task.required_scale)
    #
    #     # half of the diagonal as the minimum distance from the centre of the component to the edge of the background
    #     half_diagonal = math.ceil(math.sqrt(width ** 2 + height ** 2) / 2)
    #
    #     min_domain, max_domain = half_diagonal, background_size - half_diagonal
    #
    #     return int(random.uniform(min_domain, max_domain)), int(random.uniform(min_domain, max_domain))

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
    def choose_texture(table_name: str, given_texture: str, task: Task, db: DatabaseManager):
        ids = db.select_table(table_name).get_unique_values("id")

        if given_texture == "random":
            if table_name == BACKGROUND:
                task.background_id = random.choice(ids)
            else:
                task.component_id = random.choice(ids)
        else:
            texture_ids = db.select_table(table_name).group_by_column("id", "Texture")

            if table_name == BACKGROUND:
                task.background_id = random.choice(texture_ids[given_texture])
            else:
                task.component_id = random.choice(texture_ids[given_texture])

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
