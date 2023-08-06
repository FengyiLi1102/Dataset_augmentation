import argparse
import os.path
from typing import Union, Tuple

from src.ArgumentParser import ArgumentParser
from src.Augmentor import Augmentor
from src.DataLoader import DataLoader
from src.DatabaseManager import DatabaseManager
from src.TaskAssigner import TaskAssigner
from src.constant import DNA_AUGMENTATION, GENERATE_FAKE_BACKGROUND, CROP_ORIGAMI, RUN, CROPPED, BACKGROUND, \
    CREATE_CACHE, RAW, MOSAICS


def database_update(args: argparse.Namespace):
    database = DatabaseManager(os.path.join(f"data/{DNA_AUGMENTATION}"), training_dataset_name=args.dataset_name)
    database.scan_and_update(args.dataset_path, args.img_path, load_cache=args.cache_scan)

    return database


def generate_fake_backgrounds(args: argparse.Namespace, db: DatabaseManager):
    if args.cache_bg_path == "":
        data_loader = DataLoader.initialise(args.img_path, args.dataset_path, args.save_path, args.cache_save_dir) \
            .load_backgrounds(args.mosaic_size)
    else:
        data_loader = DataLoader.initialise(cache_save_dir=args.cache_save_dir).load_cached_files(args.cache_bg_type,
                                                                                                  args.cache_bg_path)

    # generate tasks
    background_task_assigner = TaskAssigner.background_task(args)

    # augmentor processes tasks to generate fake images
    Augmentor.produce_backgrounds(data_loader, background_task_assigner, db)


def crop_origami(args: argparse.Namespace, db: DatabaseManager):
    if args.cache_chip_path == "":
        data_loader = DataLoader.initialise(args.img_path, args.dataset_path, args.save_path, args.cache_save_dir) \
            .load_raw_components()
    else:
        data_loader = DataLoader.initialise(cache_save_dir=args.cache_save_dir).load_cached_files(args.cache_chip_type,
                                                                                                  args.cache_chip_path)

    # generate tasks
    component_task_assigner = TaskAssigner.component_task(args)

    # augmentor processes tasks to generate fake images
    Augmentor.produce_components(data_loader, component_task_assigner, db)


def run(args: argparse.Namespace, db: DatabaseManager):
    data_loader = DataLoader.initialise(args.img_path, args.dataset_path, args.save_path, args.cache_save_dir)

    data_loader = load_data(args.cache_bg_path, data_loader, True)
    data_loader = load_data(args.cache_chip_path, data_loader, False)

    # generate tasks for producing augmented training data
    augmented_task_assigner = TaskAssigner.augmented_task(args, db)
    Augmentor.produce_augmented(data_loader, augmented_task_assigner, db, debug=args.debug)


def create_cache(args):
    data_loader = DataLoader.initialise(dataset_path=args.dataset_path, save_path=args.save_path,
                                        cache_save_dir=args.cache_save_dir)

    if args.cache_name == CROPPED:
        data_loader.load_cropped_components()
    elif args.cache_name == RAW:
        data_loader.load_raw_components()
    elif args.cache_name == MOSAICS:
        data_loader.load_backgrounds(args.mosaic_size)
    else:
        data_loader.load_backgrounds(0)


def load_data(path: str, data_loader: DataLoader, is_background: bool):
    if path == "":
        if is_background:
            data_loader.load_backgrounds(0)
        else:
            data_loader.load_cropped_components()
    else:
        cache_type = path.split("/")[-1].split("_")[0]

        if cache_type in [BACKGROUND, CROPPED]:
            data_loader.load_cached_files(cache_type, path)
        else:
            raise Exception(f"Error: Only prepared {cache_type} dataset can be loaded for the augmentation")

    return data_loader


def main(args: argparse.Namespace) -> Tuple[bool, Union[DatabaseManager, None]]:
    db: Union[DatabaseManager, None] = None
    db_flag: bool = False

    if args.function in [RUN, CROP_ORIGAMI, GENERATE_FAKE_BACKGROUND]:
        db = database_update(args)
        db_flag = True

    if args.function == RUN:
        run(args, db)
    elif args.function == CROP_ORIGAMI:
        crop_origami(args, db)
    elif args.function == GENERATE_FAKE_BACKGROUND:
        generate_fake_backgrounds(args, db)
    elif args.function == CREATE_CACHE:
        create_cache(args)
    else:
        raise NameError(f"Function name {args.function} cannot be found")

    return db_flag, db


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    db_flag, database = main(args)

    if db_flag:
        database.close_connection()
