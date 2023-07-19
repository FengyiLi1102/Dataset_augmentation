import argparse
import os.path

from src.ArgumentParser import ArgumentParser
from src.Augmentor import Augmentor
from src.DataLoader import DataLoader
from src.DatabaseManager import DatabaseManager
from src.TaskAssigner import TaskAssigner
from src.constant import DNA_AUGMENTATION, GENERATE_FAKE_BACKGROUND, CROP_ORIGAMI, RUN


def generate_fake_backgrounds(args: argparse.Namespace, db: DatabaseManager):
    if args.cache_bg_type == "none":
        data_loader = DataLoader.initialise(args.img_path, args.dataset_path, args.save_path) \
            .load_backgrounds(args.mosaic_size)
    else:
        data_loader = DataLoader.initialise().load_cached_files(args.cache_bg_type, args.cache_bg_path)

    # generate tasks
    background_task_assigner = TaskAssigner.background_task(args)

    # augmentor processes tasks to generate fake images
    Augmentor.produce_backgrounds(data_loader, background_task_assigner, db)


def crop_origami(args: argparse.Namespace, db: DatabaseManager):
    if args.cache_chip_type == "none":
        data_loader = DataLoader.initialise(args.img_path, args.dataset_path, args.save_path).load_raw_components()
    else:
        data_loader = DataLoader.initialise().load_cached_files(args.cache_chip_type, args.cache_chip_path)

    # generate tasks
    component_task_assigner = TaskAssigner.component_task(args)

    # augmentor processes tasks to generate fake images
    Augmentor.produce_components(data_loader, component_task_assigner, db)


def run(args: argparse.Namespace, db: DatabaseManager):
    data_loader = DataLoader.initialise(args.img_path, args.dataset_path, args.save_path)

    if args.cache_bg_type == "none":
        data_loader.load_backgrounds(0)
    elif args.cache_bg_type == "background":
        data_loader.load_cached_files(args.cache_bg_type, args.cache_bg_path)
    else:
        raise Exception(f"Error: Only prepared backgrounds can be loaded for dataset augmentation")

    if args.cache_chip_type == "none":
        data_loader.load_components()
    elif args.cache_chip_type == "cropped":
        data_loader.load_cached_files(args.cache_chip_type, args.cache_chip_path)
    else:
        raise Exception(f"Error: Only cropped components can be loaded for dataset augmentation")

    # generate tasks for producing augmented training data
    augmented_task_assigner = TaskAssigner.augmented_task(args, db)
    Augmentor.produce_augmented(data_loader, augmented_task_assigner, db, debug=args.debug)


def main(args: argparse.Namespace, db: DatabaseManager):
    if args.function == "run":
        run(args, db)
    elif args.function == "crop_origami":
        crop_origami(args, db)
    elif args.function == "generate_fake_backgrounds":
        generate_fake_backgrounds(args, db)
    else:
        raise NameError(f"Function name {args.function} cannot be found")


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    # connect to database
    database = DatabaseManager(os.path.join(f"data/{DNA_AUGMENTATION}"), training_dataset_name=args.dataset_name)
    database.scan_and_update(args.dataset_path, args.img_path, load_cache=args.cache_scan)

    main(args, database)

    # main(ArgumentParser.test_args(RUN), database)

    database.close_connection()
