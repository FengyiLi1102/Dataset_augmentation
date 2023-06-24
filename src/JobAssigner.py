from __future__ import annotations

import argparse
import os.path
from typing import Dict, List

import numpy as np

from src.Background import Background
from src.DataLoader import DataLoader
from tqdm import tqdm
import random


class JobAssigner:
    # General
    save_path: str

    # Backgrounds
    background_job_pipeline: Dict[str, List] = dict()

    """
    Each background mosaic has flip or 180 degree rotation operations.
    flip: vertical - 'v' / horizontal - 'h'
    rotation: 180
    """
    operations_backgrounds = ['v', 'h', 180]
    num_per_side: int
    num_mosaic_in_background: int

    # Components

    @classmethod
    def background_job(cls, data_loader: DataLoader, args: argparse.Namespace):
        if int(sum(args.ratio)) != 10:
            raise Exception(
                f"Error: Given ratios are not valid. Sum of them should be 10 instead of {int(sum(args.ratio))}.")

        job_assigner = cls()

        job_assigner.save_path = args.save_path
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        num_for_textures = [int(args.number * r / 10) for r in args.ratio]
        job_assigner.num_per_side = int(args.size / args.b_size)
        job_assigner.num_mosaic_in_background = job_assigner.num_per_side ** 2

        print(">>> Start to assign jobs for generating backgrounds")
        # Assign jobs for producing backgrounds
        for idx, texture in tqdm(enumerate(Background.background_textures)):
            job_assigner.__background_assign_helper(texture, job_assigner.num_mosaic_in_background,
                                                    num_for_textures[idx])

        return job_assigner

    def __background_assign_helper(self, texture: str, n_mo_backgrounds: int, n_produced: int):
        # Assign jobs for producing backgrounds
        print(f">>> Assign {n_produced} jobs for generating {texture} backgrounds")
        self.background_job_pipeline[texture] = [
            random.choices(self.operations_backgrounds, k=n_mo_backgrounds) for _ in range(n_produced)]

    def allocate(self, data_loader: DataLoader, args: argparse.Namespace):
        pass
