import argparse

from src.DataLoader import DataLoader
from src.JobAssigner import JobAssigner


def run(args: argparse.Namespace):
    job_assigner = JobAssigner(args)
    job_assigner.allocate(DataLoader.load_images_labels(args.img_path, args.log_path), args)

