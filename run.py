import argparse
import os.path


from src.ArgumentParser import ArgumentParser
from src.Augmentor import Augmentor
from src.DataLoader import DataLoader
from src.DatabaseManager import DatabaseManager
from src.TaskAssigner import TaskAssigner
from src.constant import DNA_AUGMENTATION, GENERATE_FAKE_BACKGROUND, CROP_ORIGAMI, RUN


def generate_fake_backgrounds(args: argparse.Namespace):
    # connect to database
    db = DatabaseManager(os.path.join(f"data/{DNA_AUGMENTATION}"))
    db.scan_and_update(args.dataset_path)

    # load required data images
    data_loader = DataLoader.initialise(args.img_path, args.dataset_path).load_backgrounds(args.mosaic_size)

    # generate tasks
    background_task_assigner = TaskAssigner.background_task(args)

    # augmentor processes tasks to generate fake images
    Augmentor.produce_backgrounds(data_loader, background_task_assigner, db)


def crop_origami(args: argparse.Namespace):
    # connect to database
    db = DatabaseManager(f"data/{DNA_AUGMENTATION}")
    db.scan_and_update(args.dataset_path)

    # load required data images
    data_loader = DataLoader.initialise(args.img_path, args.dataset_path).load_raw_components()

    # generate tasks
    component_task_assigner = TaskAssigner.component_task(args)

    # augmentor processes tasks to generate fake images
    Augmentor.produce_components(data_loader, component_task_assigner, db)


def run(args: argparse.Namespace):
    db = DatabaseManager(os.path.join("data", DNA_AUGMENTATION), training_dataset_name=args.dataset_name)
    db.scan_and_update(args.dataset_path)

    # load prepared fake backgrounds and components
    data_loader = DataLoader.initialise(args.img_path, args.dataset_path).load_backgrounds(0).load_components()

    # generate tasks for producing augmented training data
    augmented_task_assigner = TaskAssigner.augmented_task(args, db)
    Augmentor.produce_augmented(data_loader, augmented_task_assigner, db, debug=args.debug)

    db.close_connection()


def main(args: argparse.Namespace):
    if args.function == "run":
        run(args)
    elif args.function == "crop_origami":
        crop_origami(args)
    elif args.function == "generate_fake_backgrounds":
        generate_fake_backgrounds(args)
    else:
        raise NameError(f"Function name {args.function} cannot be found")


if __name__ == "__main__":
    # parser = ArgumentParser()
    # args = parser.parse_args()
    # main(args)

    # Test
    main(ArgumentParser.test_args(RUN))
