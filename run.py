import argparse
import os.path

import numpy as np

from src.ArgumentParser import ArgumentParser
from src.Augmentor import Augmentor
from src.DataLoader import DataLoader
from src.DatabaseManager import DatabaseManager
from src.TaskAssigner import TaskAssigner
from src.constant import DNA_AUGMENTATION


def crop_origami(args: argparse.Namespace):
    db = DatabaseManager(os.path.join("data", DNA_AUGMENTATION))
    db.scan_and_update(args.dataset_path)
    data_loader = DataLoader.initialise(args.img_path).load_raw_components()
    component_task_assigner = TaskAssigner.component_task(args)
    Augmentor.produce_components(data_loader, component_task_assigner, db)


def run(args: argparse.Namespace):
    db = DatabaseManager(os.path.join("data", DNA_AUGMENTATION))
    db.scan_and_update(args.dataset_path)

    if args.init_background:
        # initialise backgrounds
        data_loader = DataLoader.initialise(args.img_path, args.dataset_path).load_backgrounds(args.b_size)
        background_task_assigner = TaskAssigner.background_task(args)
        # Augmentor.produce_backgrounds(data_loader, background_task_assigner, db)
    else:
        # use existing backgrounds
        data_loader = DataLoader.initialise(args.img_path, args.dataset_path).load_backgrounds(0)

    data_loader.load_components()

    augmented_task_assigner = TaskAssigner.augmented_task(args, db)
    # Augmentor.produce_augmented(data_loader, augmented_task_assigner, db)

    db.close_connection()


def main(args: argparse.Namespace):
    if args.function == "run":
        run(args)
    elif args.function == "crop_origami":
        crop_origami(args)
    else:
        raise NameError(f"Function name {args.function} cannot be found")


if __name__ == "__main__":
    main(ArgumentParser.run_default())
