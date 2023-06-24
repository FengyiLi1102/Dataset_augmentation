import os.path
import random
from typing import List

import cv2
import numpy as np
import argparse
from tqdm import tqdm

from src.DataLoader import DataLoader
from src.DatabaseManager import DatabaseManager
from src.Image import Image
from src.JobAssigner import JobAssigner
from src.constant import DNA_AUGMENTATION, BACKGROUND


class Augmentor:

    @classmethod
    def produce_backgrounds(cls, data_loader: DataLoader, job_assigner: JobAssigner):
        save_path = os.path.join(job_assigner.save_path, BACKGROUND)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        max_id = cls.__get_current_num(BACKGROUND)
        counter = max_id + 1 if max_id != 0 else max_id

        # connect to the database
        db = DatabaseManager(DNA_AUGMENTATION)

        print(f">>> Start to produce background images")
        for texture, jobs in tqdm(job_assigner.background_job_pipeline.items(),
                                  total=len(job_assigner.background_job_pipeline)):
            print(f">>> Producing total {len(jobs)} {texture} backgrounds")

            for one_job in jobs:
                concatenated_image = cls.__mosaics_in_row(one_job[0: job_assigner.num_per_side],
                                                          data_loader, texture)

                for start_img in range(job_assigner.num_per_side, job_assigner.num_mosaic_in_background,
                                       job_assigner.num_per_side):
                    row_image = cls.__mosaics_in_row(one_job[start_img: start_img + job_assigner.num_per_side],
                                                     data_loader, texture)
                    concatenated_image = cv2.vconcat([concatenated_image, row_image])

                save_name = f"fake_{texture}_background_{counter}.png"
                cv2.imwrite(os.path.join(save_path, save_name), concatenated_image)
                db.select_table("background").insert_data(Background_name=save_name, Texture=texture)

                counter += 1
        db.close_connection()

    @staticmethod
    def __get_current_num(type_name: str) -> int:
        db = DatabaseManager(DNA_AUGMENTATION)
        max_id = db.select_table(type_name).read_max_id()
        db.close_connection()

        return max_id

    @staticmethod
    def __mosaics_in_row(jobs_row: List, data_loader: DataLoader, texture: str) -> List[np.array]:
        augmented_images = []

        for operation in jobs_row:
            augmented_images.append(Augmentor.operate(operation, random.choice(data_loader.background_img[texture])))

        return cv2.hconcat(augmented_images)

    @staticmethod
    def operate(operation_type, image: Image) -> np.array:
        if operation_type in ['v', 'h']:
            # flip
            return Augmentor.__flip(image.read(), operation_type)
        elif -180 <= operation_type <= 180:
            # rotate
            return Augmentor.__rotate(image.read(), operation_type)
        else:
            raise Exception(f"Error: Incorrect operation {operation_type} is given")

    @staticmethod
    def __flip(image: np.array, direction: str) -> np.array:
        if direction == 'v':
            return cv2.flip(image, 0)
        else:
            return cv2.flip(image, 1)

    @staticmethod
    def __rotate(image: np.array, operation_type: int) -> np.array:
        # positive angle -> counter clock-wise
        # negative angle -> clock-wise
        rows, cols = image.shape[:2]
        center = (cols / 2, rows / 2)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, operation_type, scale)  # rotation matrix

        return cv2.warpAffine(image, M, (cols, rows))
