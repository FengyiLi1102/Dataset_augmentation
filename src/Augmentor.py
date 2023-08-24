import math
import os.path
import pickle
import logging
import random
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Union, Tuple

import cv2
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from rich.progress import track

from src.AugmentedImage import AugmentedImage
from src.ArgumentParser import ArgumentParser
from src.Background import Background
from src.Component import Component
from src.DataLoader import DataLoader
from src.DatabaseManager import DatabaseManager
from src.Image import Image
from src.TaskAssigner import TaskAssigner
from src.constant import BACKGROUND, TRAINING, SIMPLE, AUGMENTATION, VALIDATION, RUN, CROPPED, TESTING, V, H, N, \
    DNA_ORIGAMI, ACTIVE_SITE, GENERATE_EXTENDED_STRUCTURE
from src.DNALogging import DNALogging
from src.utils import mkdir_if_not_exists, process_labels, compute_m_from_angle, compute_m_c, \
    compute_angle_between_horizontal, find_clockwise_order, compute_new_centre_row
from src.typeHint import *

DNALogging.config_logging()
logger = logging.getLogger(__name__)


class Augmentor:
    # Augmentation
    patience: int = 0
    debug: bool = False

    @classmethod
    def produce_backgrounds(cls,
                            data_loader: DataLoader,
                            task_assigner: TaskAssigner,
                            db: DatabaseManager) -> None:
        """

        :param data_loader:
        :param task_assigner:
        :param db:
        :return:
        """
        save_path = os.path.join(task_assigner.save_path, BACKGROUND)
        mkdir_if_not_exists(save_path)

        max_id = cls.__get_current_num(BACKGROUND, db)
        counter = max_id + 1 if max_id != 0 else max_id

        # cache for future fast load
        name_background: Dict[str, Background] = dict()

        logger.info(">>> Start to produce background images")
        for texture, tasks in task_assigner.background_task_pipeline.items():
            logger.info(f">>> Producing total {len(tasks)} {texture} backgrounds")

            for one_task in tasks:
                concat_whole_image = cls.__mosaics_in_row(one_task[0: task_assigner.num_per_side],
                                                          data_loader.bg_or_mosc_img[texture],
                                                          task_assigner.kernel_size)

                for start_img in range(task_assigner.num_per_side, task_assigner.num_mosaic_in_background,
                                       task_assigner.num_per_side):
                    concat_row_image = cls.__mosaics_in_row(one_task[start_img: start_img + task_assigner.num_per_side],
                                                            data_loader.bg_or_mosc_img[texture],
                                                            task_assigner.kernel_size)

                    concat_whole_image = cls.smooth_seam(concat_whole_image, concat_row_image,
                                                         task_assigner.kernel_size, axis=0)

                save_name = f"fake_{texture}_background_{counter}"
                save_name_png = f"{save_name}.png"
                cv2.imwrite(os.path.join(save_path, save_name_png), concat_whole_image)

                # add into the cache
                fast_data = {
                    "img": concat_whole_image,
                    "img_name": save_name_png
                }

                img = Background(img_path=None, **fast_data)
                name_background[save_name] = img

                # update the database
                db.select_table(BACKGROUND).insert_data(Background_name=save_name_png, Texture=texture)

                counter += 1

        # create the cache
        cache_save_path = os.path.join(task_assigner.save_path, task_assigner.cache_save_dir)

        mkdir_if_not_exists(cache_save_path)

        with open(os.path.join(cache_save_path, f'background_{datetime.now().strftime("%Y_%m_%d_%H:%M")}.pkl'),
                  "wb") as f:
            pickle.dump(name_background, f)

    @staticmethod
    def __get_current_num(type_name: str,
                          db: DatabaseManager) -> int:
        max_id = db.select_table(type_name).read_max_id()

        return max_id

    @staticmethod
    def __mosaics_in_row(task_in_row: List,
                         mosaics_pool: List[Background],
                         k_size: int) -> List[np.array]:
        augmented_images = []

        try:
            for operation in task_in_row:
                augmented_images.append(Augmentor.operate(operation, random.choice(mosaics_pool)))
        except IndexError:
            raise Exception(f"Not enough mosaics loaded: only with {len(mosaics_pool)}")

        h_concat_images = augmented_images[0]
        for img in augmented_images[1:]:
            h_concat_images = Augmentor.smooth_seam(h_concat_images, img, k_size)

        return h_concat_images

    @staticmethod
    def smooth_seam(img_1: np.array,
                    img_2: np.array,
                    k_size: int,
                    axis: int = 1) -> np.array:
        """
        Only for horizontal and vertical direction.
        TODO: require for straight lines in any directions
        :param img_1: left or top
        :param img_2: right or bottom
        :param k_size: kernel size for defining the averaging filter
        :param axis: 1 -> horizontal, 0 -> vertical
        :return:
        """
        if axis not in [0, 1]:
            raise IndexError(f"Incorrect index {axis} is given")

        dimension = img_1.shape[axis]
        kernel = np.ones((1, k_size), np.float32) / k_size
        kernel = kernel if axis else kernel.T

        if axis:
            temp_concatenated_image = cv2.hconcat([img_1, img_2])

            temp_concatenated_image[:, int(dimension - k_size - 1): int(dimension + k_size)] = cv2.filter2D(
                temp_concatenated_image[:, int(dimension - k_size - 1): int(dimension + k_size)], -1, kernel)
        else:
            temp_concatenated_image = cv2.vconcat([img_1, img_2])

            temp_concatenated_image[int(dimension - k_size - 1): int(dimension + k_size), :] = cv2.filter2D(
                temp_concatenated_image[int(dimension - k_size - 1): int(dimension + k_size), :], -1, kernel)

        return temp_concatenated_image

    @staticmethod
    def operate(operation_type,
                image: Image) -> np.array:
        if operation_type in [V, H, N]:
            # flip
            return Augmentor.__flip(image.read(), operation_type)[0]
        elif -180 <= operation_type <= 180:
            # rotate
            return Augmentor._rotate(image.read(), operation_type)[0]
        else:
            raise Exception(f"Error: Incorrect operation {operation_type} is given")

    @staticmethod
    def __flip(img: np.ndarray,
               direction: int,
               labels: LabelsType | None = None) -> Tuple[np.ndarray, LabelsType | None]:
        flipped_labels = None

        if direction == N:
            # no flip
            flipped_image = img

            if labels:
                flipped_labels = labels
        else:
            if direction == V:
                flipped_image = cv2.flip(img, 0)

                if labels:
                    flipped_labels = process_labels(img, labels, Augmentor.__flip_position_v, False)
            elif direction == H:
                flipped_image = cv2.flip(img, 1)

                if labels:
                    flipped_labels = process_labels(img, labels, Augmentor.__flip_position_h, False)
            else:
                raise ValueError(f"Given flip option {direction} is invalid. Please use one of V, H, and N. \n"
                                 f"V: vertical; H: horizontal; N: Not flip")

            flipped_labels = Component.order_labels_in_clockwise_order(flipped_labels)

        return flipped_image, flipped_labels

    @staticmethod
    def __flip_position_v(img: np.ndarray, positions: np.ndarray, cartesian: bool = False) -> np.ndarray:
        return np.array([[pos[0], img.shape[0] - pos[1]] for pos in positions])

    @staticmethod
    def __flip_position_h(img: np.ndarray, positions: np.ndarray, cartesian: bool = False) -> np.ndarray:
        return np.array([[img.shape[1] - pos[0], pos[1]] for pos in positions])

    @staticmethod
    def _rotate(img: np.ndarray,
                angle: float,
                labels: LabelsType | None = None) -> Tuple[np.ndarray, LabelsType | None, np.ndarray]:
        # positive angle -> counter clock-wise
        # negative angle -> clock-wise
        rows, cols = img.shape[: 2]
        center = (cols / 2, rows / 2)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M_c = M.copy()

        # Calculate the size of the new image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((rows * sin) + (cols * cos))
        new_h = int((rows * cos) + (cols * sin))

        # Adjust the rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated_labels = defaultdict(list)

        # if labels given
        if labels is not None:
            rotation_matrix_2D = M_c[:, :2]  # 2 x 2

            for label_type, label_list in labels.items():
                for label in label_list:
                    # Reshape the label to 2 x 4
                    _label = label.T
                    rotated_label = np.dot(rotation_matrix_2D, _label)  # 2 x 4
                    # refer to the centre of image
                    # 4 x 2
                    rotated_labels[label_type].append(rotated_label.T + np.array([new_w / 2, new_h / 2]))

        return cv2.warpAffine(img, M, (new_w, new_h)), rotated_labels, M

    # NOTE: The following function is adapted from Tejas' previous work.
    # This function was originally written by Tejas Narayan, and I may do some change on it.
    # Source: https://github.com/ic-dna-storage/tn21-ic-msc-project/blob/main/code/image_analysis/cv/models/dna_origami.py#L187C11-L187C11
    @staticmethod
    def produce_components(data_loader: DataLoader,
                           task_assigner: TaskAssigner,
                           db: DatabaseManager):
        """
        Note: Cropping components will not create the cache file because the programme cannot automatically decide
        which texture the component is. The cache will be only created when the manual work is done and components are
        firstly loaded.
        :param data_loader:
        :param task_assigner:
        :param db:
        :return:
        """
        # https://pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/
        find_chips_config = task_assigner.config["find_dna_chip"]
        total_chips = 0
        index_in_name = Augmentor.__get_current_num(CROPPED, db)

        save_path = os.path.join(task_assigner.save_path, CROPPED, "images")
        mkdir_if_not_exists(save_path)

        logger.info(f">>> Start to crop DNA origami to produce component")

        for input_img in list(data_loader.name_raw_input.values()):
            logger.info(f"Process image {input_img.img_name}")

            img = input_img.read()
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            blurred = cv2.GaussianBlur(img_grey, (find_chips_config["blur_radius"], find_chips_config["blur_radius"]),
                                       0)

            otsu_threshold, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
            best_dilate_iteration = find_chips_config["dilate_iterations"]
            best_erode_iteration = find_chips_config["erode_iterations"]

            dilated = cv2.dilate(th, kernel, iterations=best_dilate_iteration)
            eroded = cv2.erode(dilated, kernel, iterations=best_erode_iteration)

            contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # find the biggest contour by area
            cs = sorted(contours, key=cv2.contourArea, reverse=True)

            cutoff = cv2.contourArea(cs[0]) * find_chips_config["proportion_max"]
            cs = [c for c in cs if cv2.contourArea(c) > cutoff]
            num_found_chips = len(cs)

            logger.info(f"Found {num_found_chips} chips")

            total_chips += num_found_chips

            for c in cs:
                # find the rotated minRectangle for that contour
                rect = cv2.minAreaRect(c)

                # https://stackoverflow.com/questions/18207181/opencv-python-draw-minarearect-rotatedrect-not-implemented
                # Inflate the minimum area rectangle by 75px on all sides because we're not perfect
                ((x, y), (w, h), angle) = rect
                rect = ((x, y), (w + task_assigner.cropping_inflation, h + task_assigner.cropping_inflation), angle)

                box = cv2.boxPoints(rect)
                box = np.intp(box)

                # draw the biggest contour in green
                x, y, w, h = cv2.boundingRect(c)

                img_copy = img.copy()

                contour_color, contour_thickness = (0, 255, 0), 20
                cv2.rectangle(
                    img_copy,
                    (x, y),
                    (x + w, y + h),
                    color=contour_color,
                    thickness=contour_thickness,
                )

                # draw the minAreaRectangle in blue
                min_rect_color, min_rect_thickness = (255, 0, 0), 20
                cv2.drawContours(
                    image=img_copy,
                    contours=[box],
                    contourIdx=0,
                    color=min_rect_color,
                    thickness=min_rect_thickness,
                )

                found_chip = Augmentor._crop_rect(img, rect)

                component_name = f"component_{index_in_name}"
                img_name = f"cropped_TEXTURE_" + component_name + ".png"
                cv2.imwrite(os.path.join(save_path, img_name), found_chip)

                # update record in the database
                db.select_table(CROPPED).insert_data(Raw_image=input_img.img_name + "." + input_img.ext,
                                                     Sample=img_name,
                                                     Morphology="TEXTURE", Height=found_chip.shape[0],
                                                     Width=found_chip.shape[1])

                index_in_name += 1

        logger.info(f"All {total_chips} are saved in {save_path}")

    # NOTE: The following function is adapted from Tejas' previous work.
    # This function was originally written by Tejas Narayan, and I did some change on it.
    # Source: https://github.com/ic-dna-storage/tn21-ic-msc-project/blob/main/code/image_analysis/cv/models/dna_origami.py#L116C10-L116C10
    @staticmethod
    def _crop_rect(img: np.array, rect) -> np.array:
        # get the perimeter of the small rectangle
        center, size, angle = rect

        # Convert into a tuple of ints
        center, size = tuple(map(int, center)), tuple(map(int, size))

        # get row and col num in img
        height, width = img.shape[:2]

        m = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, m, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)

        return img_crop

    @classmethod
    def produce_extended_structures(cls,
                                    data_loader: DataLoader,
                                    task_assigner: TaskAssigner,
                                    db: DatabaseManager,
                                    debug: bool = False):
        cls.debug = debug

        # Debug: display each task information
        if debug:
            for row in task_assigner.augmentation_task_pipeline:
                print(row)

        # Saving directory
        save_path_root = os.path.join(task_assigner.save_path, task_assigner.dataset_name)
        Augmentor.__save_directory(task_assigner.mode, save_path_root)

        # cache for future fast loading
        name_augmented: Dict[str, AugmentedImage] = dict()

        logger.info(f">>> Start to produce {task_assigner.stitch_size} extended structures for a task of "
                    f"{task_assigner.expected_num}")
        for counter, task in tqdm.tqdm(enumerate(task_assigner.augmentation_task_pipeline)):
            this_rotation = 0

            # all structures -> one structure -> multiple chips -> labels: type <-> multiple
            finished_labels: List[List[LabelsType]] = [[] for _ in range(task.n_structure)]

            # for multiple chips in one structure
            pos_in_struc = [index for index, _ in
                            np.ndenumerate(np.arange(task_assigner.n_stitch).reshape(task_assigner.stitch_size))]

            component_img: np.ndarray | None = None
            component_labels: LabelsType | None = None

            stitched_row_img: np.ndarray | None = None  # image of the structure
            stitched_label: StitchLabelType = dict()  # label of the structure
            stitched_row_label: np.ndarray | None = None  # bounding box enclosing the row structure

            # m and c values for each row
            m_c_arr: np.ndarray = np.zeros(task_assigner.stitch_size, dtype=np.float64)

            pixel_to_nm: float = task_assigner.initial_scale / 70

            # store structure part along the row
            # e.g. 2 x 2 structure will be stored with 2 parts with a size of 1 x 2
            # both for stitching along the y-axis
            structure_row_img: List[np.ndarray] = []  # store row structure image
            structure_row_label: List[np.ndarray] = []  # store row structure label

            # ================================ >>> Start for one chip <<< ================================== #
            # for each chip in a structure
            for idx_chip in range(task_assigner.n_stitch):
                # if the stitch structure has more than one chips
                stitch_pos = pos_in_struc[idx_chip]

                # extract the data
                this_component_id = task.component_id[stitch_pos]
                this_flip = task.flip[stitch_pos]

                # find the corresponding component from its id
                component = cls.__id_to_image(CROPPED, this_component_id, data_loader.name_component, db)

                if debug:
                    component.draw_box("0_first")

                # scale into initial size if not yet
                if not component.initial_scale:
                    # based on the height of the chip
                    adjusted_scale = task_assigner.initial_scale / component.chip_h

                    # resize the component image
                    component.resize_into(
                        int(component.img_size[1] * adjusted_scale), int(component.img_size[0] * adjusted_scale))
                    component.update_resizing_res(adjusted_scale)

                    component.initial_scale = True

                    # Debug: show initial component images
                    if debug:
                        component.draw_box("1_init")

                # chips may have tiny difference on the scale within an acceptable range
                # In the extended structure, this tiny amount is not known now.
                # Assume that even 10% of the difference in area will destroy the structure.
                new_scale = round(np.random.uniform(0.9, 1.1), task_assigner.decimal_place)
                new_side_scale = np.sqrt(new_scale)

                # Note: in width and height for cv2.resize
                scaled_size = tuple(int(x) for x in np.dot(component.img_size, new_side_scale))[::-1]
                component_img = cv2.resize(component.read(), scaled_size)
                component_labels = Component.rescale_label(component.labels, new_side_scale)

                # compute the relation between the pixel and real length of one chip in the background
                # assume the real length of each chip is the same of 70 nm
                if not pixel_to_nm:
                    chip_w, _ = Component.compute_chip_size(component_labels[DNA_ORIGAMI][0])
                    # pixel_to_nm = chip_w / 70
                    pixel_to_nm = 10

                # debug: show resized component with its labels
                if debug:
                    component.draw_box(f"2_resize_{task.required_scale:.2f}", component_img=component_img,
                                       component_label=component_labels)

                # flip
                # if this_flip != N:
                component_img, component_labels = cls.__flip(component_img, this_flip, labels=component_labels)
                chip_rotation = compute_angle_between_horizontal(labels=component_labels[DNA_ORIGAMI][0])

                # debug: show resized and flipped component with its surrounding box
                if debug:
                    component.draw_box(f"3_flipped_{this_flip}", component_img=component_img,
                                       component_label=component_labels)

                # rotate
                component_img_centre = np.divide(component_img.shape[: 2][::-1], 2)
                component_label_to_centre = Component.convert_TL_to_centre(component_img_centre, component_labels)
                additional_rotation = this_rotation - chip_rotation
                component_img, component_labels, _ = Augmentor._rotate(component_img, additional_rotation,
                                                                       component_label_to_centre)

                # debug: show resized, flipped and rotated component with its surrounding box
                if debug:
                    component.draw_box(f"4_rotated_{additional_rotation:.2f}", component_img=component_img,
                                       component_label=component_labels)

                component_img, component_labels = component.crop(component_img, component_labels, debug=True,
                                                                 pos=stitch_pos)

                if stitch_pos == (0, 0):
                    stitched_row_img = component_img
                    stitched_label[stitch_pos] = component_labels
                    chip_label = component_labels[DNA_ORIGAMI][0]
                    chip_label_centre_d = Component.from_TL_to_centre(np.mean(chip_label, axis=0), chip_label,
                                                                      cartesian=True)
                    m_c_arr[0] = compute_m_c(chip_label_centre_d[-1], chip_label_centre_d[-2], cartesian=True)
                    continue
                else:
                    if stitch_pos[1] == 0:
                        # if this chip starts from a new row
                        # first store the earlier row stitched image.
                        structure_row_img.append(stitched_row_img)

                        # store the larger labeling box enclosing row structure
                        structure_row_label.append(stitched_row_label)

                        # reset
                        stitched_row_img = component_img
                        stitched_label[stitch_pos] = component_labels

                        # compute the bottom line equation with m and c
                        chip_label = component_labels[DNA_ORIGAMI][0]
                        chip_label_centre_d = Component.from_TL_to_centre(np.mean(chip_label, axis=0), chip_label,
                                                                          cartesian=True)
                        m_c_arr[stitch_pos[0]] = compute_m_c(chip_label_centre_d[-1], chip_label_centre_d[-2],
                                                             cartesian=True)
                        continue

                    stitched_row_img, stitched_label[stitch_pos], stitched_row_label = (
                        Augmentor.__stitch_along_row(
                            component_img,
                            component_labels,
                            stitched_row_img,
                            stitched_label,
                            stitch_pos,
                            m_c_arr[stitch_pos[0]],
                            task_assigner.gap_w * pixel_to_nm
                        )
                    )

                    if debug:
                        cv2.imwrite(f"../debug/debug_6_{counter}_row_{idx_chip}.png", stitched_row_img)

                    # after all the row structures are finished and start to stitch them along y-axis
                    if idx_chip != task_assigner.n_stitch:
                        continue

                # stitch all chips along y-axis
                component_img, component_labels = Augmentor.__stitch_along_col(structure_row_img,
                                                                               stitched_label,
                                                                               task_assigner.gap_h * pixel_to_nm)

                # add new records for updating the database
                # TODO: design a table for extended structure
                # new_record = {
                #     "Component_id": this_component_id,
                #     "Background_id": task.background_id,
                #     "Component_scale": round(this_required_area_scale, 2),
                #     "Flip": this_flip,
                #     "Rotate": this_rotation,
                # }
                # db_records.append(new_record)

                # ============================== End of processing one structure =============================== #

            # generate the image name
            save_name = "augmented_" + str(task_assigner.stitch_size) + "_" + str(counter)
            # component_id_str = []
            # scale_str = []
            # flip_str = []
            # rotate_str = []
            #
            # for chip in db_records:
            #     component_id_str.append(str(chip["Component_id"]))
            #     scale_str.append(str(chip["Component_scale"]))
            #     flip_str.append(chip["Flip"])
            #     rotate_str.append(str(chip["Rotate"]))

            # for value_list in [component_id_str, scale_str, flip_str, rotate_str]:
            #     save_name += "_"
            #     save_name += ",".join(value_list)

            # debug: show final result
            if debug:
                canvas = Image.plot_labels(component_img, component_labels)
                mkdir_if_not_exists("../debug")
                cv2.imwrite(f"../debug/debug_{save_name}.png", canvas)

            # images
            save_name_png = f"{save_name}.png"
            cv2.imwrite(os.path.join(save_path_root, "images", save_name_png), component_img)

            # labels
            save_name_txt = f"{save_name}.txt"
            with open(os.path.join(save_path_root, "labelTxt", save_name_txt), "w") as f:
                for label_type, value_list in component_labels.items():
                    # iterate one type of labels
                    for label in value_list:
                        # get one pair of coordinates
                        for coordinates in label:
                            x, y = coordinates
                            f.write(f"{x:.6f}" + " ")
                            f.write(f"{y:.6f}" + " ")

                        f.write(label_type + " ")
                        f.write(str(task_assigner.difficult))
                        f.write("\n")

            # update the database
            # for record in db_records:
            #     record["Image_name"] = save_name_png
            #     record["Category"] = category
            #     record["LabelTxt"] = save_name_txt
            #
            #     db.select_table(task_assigner.dataset_name).insert_data(**record)

            # add it into the cache
            # fast_create_data = {
            #     "img": augmented_img,
            #     "img_name": save_name_png,
            #     "labels": finished_labels,
            #     "data": db_records
            # }
            #
            # img = AugmentedImage(category, **fast_create_data)
            # name_augmented[save_name] = img

            if counter % 200 == 0:
                logger.info(f"Finish {counter} tasks ...")

            # ====================================== Finish one task ==================================== #

        # create the cache
        # TODO: (perf) create separate image and label caches -> save space
        # if task_assigner.cache:
        #     cache_save_path = os.path.join(task_assigner.save_path, task_assigner.cache_save_dir)
        #
        #     mkdir_if_not_exists(cache_save_path)
        #
        #     cache_name = f'{task_assigner.dataset_name}_{datetime.now().strftime("%Y_%m_%d_%H:%M")}.pkl'
        #     with open(os.path.join(cache_save_path, cache_name), "wb") as f:
        #         pickle.dump(name_augmented, f)
        #
        #     logger.info(f">>> Create a cache file for {cache_name}")

    @classmethod
    def produce_augmented(cls,
                          data_loader: DataLoader,
                          task_assigner: TaskAssigner,
                          db: DatabaseManager,
                          debug: bool = False):
        cls.patience = task_assigner.patience
        cls.debug = debug

        background_size = data_loader.bg_or_mosc_img["clean"][0].img_size[0]
        scaled_height = background_size * task_assigner.initial_scale  # initial_scale is for the side not for area

        split_to_folder = {
            TRAINING: "train",
            VALIDATION: "val",
            TESTING: "test"
        }

        # Debug: display each task information
        if debug:
            for row in task_assigner.augmentation_task_pipeline:
                print(row)

        # Saving directory
        save_path_root = os.path.join(task_assigner.save_path, task_assigner.dataset_name)
        Augmentor.__save_directory(task_assigner.mode, save_path_root)

        # cache for future fast loading
        name_augmented: Dict[str, AugmentedImage] = dict()

        finished_number: int = 0  # total number of tasks (images) actually finished
        counter: int = 0  # total number of tasks that have been processed

        flags: List[bool] = [False, False, False]

        logger.info(f">>> Start to augment the dataset for a target of {task_assigner.expected_num}")
        for task in tqdm.tqdm(task_assigner.augmentation_task_pipeline):
            # check if this type of the task has been finished
            if flags[task.split]:
                continue

            # background
            background = cls.__id_to_image(BACKGROUND, task.background_id, data_loader.name_bg_or_mosc, db)
            background_img = background.read()

            augmented_img = background_img.copy()

            this_required_area_scale = task.required_scale
            this_required_side_scale = np.sqrt(this_required_area_scale)

            # all structures -> one structure -> multiple chips -> labels: type <-> multiple
            finished_labels: List[List[LabelsType]] = [[] for _ in range(task.n_structure)]
            pos_recorder: np.ndarray = np.zeros(augmented_img.shape[: 2])

            # for only one or multiple chips in one structure
            db_records: List[Dict[str, str | int | float]] | None = []
            pos_in_struc = None

            skip_flag: bool = False
            task_labels: np.ndarray | None = None  # contain all structure labels in one background image

            # ================================== >>> Start for one structure <<< ==================================== #
            # for each structure
            for idx_struc in range(task.n_structure):
                this_rotation = task.rotation[idx_struc]

                # ================================ >>> Start for one chip <<< ================================== #
                # for each chip in a structure
                for idx_chip in range(task_assigner.n_stitch):
                    # extract the data
                    this_component_id = task.component_id[idx_struc][idx_chip]
                    this_flip = task.flip[idx_struc][idx_chip]

                    # find the corresponding component from its id
                    component = cls.__id_to_image(CROPPED, this_component_id, data_loader.name_component, db)

                    # scale into initial size if not yet
                    if not component.initial_scale:
                        # based on the height of the chip
                        adjusted_scale = scaled_height / component.chip_h

                        # resize the component image
                        component.resize_into(
                            int(component.img_size[1] * adjusted_scale), int(component.img_size[0] * adjusted_scale))
                        component.update_resizing_res(adjusted_scale)

                        component.initial_scale = True

                        # Debug: show initial component images
                        if debug:
                            component.draw_box("init")

                    # Rescale the component image
                    if this_required_area_scale in component.scaled_image:
                        # have rescaled before, directly apply from the storage
                        component_img = component.scaled_image[this_required_area_scale]
                        component_labels = component.scaled_labels[this_required_area_scale]
                    elif this_required_area_scale == 1.0:
                        component_img = component.read()
                        component_labels = component.labels
                    else:
                        # new scale
                        if task_assigner.n_stitch > 1:
                            # chips may have tiny difference on the scale within an acceptable range
                            # In the extended structure, this tiny amount is not known at the moment.
                            # Assume that even 10% of difference in area will destroy the whole structure
                            new_scale = round(
                                np.abs(np.random.normal(this_required_area_scale, this_required_area_scale / 2)),
                                task_assigner.decimal_place
                            )

                            if abs(new_scale - this_required_area_scale) / this_required_area_scale < 0.1:
                                this_required_area_scale = new_scale
                                this_required_side_scale = np.sqrt(new_scale)

                        # Note: in width and height for cv2.resize
                        scaled_size = tuple(
                            int(x) for x in np.dot(component.img_size[: 2], this_required_side_scale)
                        )[::-1]
                        component_img, component_labels = (
                            component.add_resizing_res(scaled_size,
                                                       this_required_side_scale,
                                                       decimal_places=task_assigner.decimal_place)
                        )

                    # debug: show resized component with its labels
                    if debug:
                        component.draw_box(f"resize_{task.required_scale:.2f}", component_img=component_img,
                                           component_label=component_labels)

                    # compute the relation between the pixel and real length of one chip in the background
                    # assume the real length of each chip is the same of 70 nm
                    pixel_to_nm = component.chip_wh_cache[this_required_area_scale][0] / 70

                    # check if component is larger than the background
                    Augmentor.__is_larger(component_img, background_img, error_flag=True)

                    # flip
                    if this_flip == N:
                        # no flip
                        pass
                    elif this_required_area_scale == 1.0:
                        # just flip without rescale (still initial scale)
                        if this_flip in component.flipped_image:
                            # from the storage
                            component_img = component.flipped_image[this_flip]
                            component_labels = component.flipped_label[this_flip]
                        else:
                            # not flipped before
                            component_img, component_labels = cls.__flip(component_img, this_flip,
                                                                         labels=component_labels)
                            component.add_flipping_res(this_flip, component_img, component_labels)
                    else:
                        # rescaled and flip
                        component_img, component_labels = cls.__flip(component_img, this_flip, labels=component_labels)

                    # debug: show resized and flipped component with its surrounding box
                    if debug:
                        component.draw_box(f"flipped_{task.flip}", component_img=component_img,
                                           component_label=component_labels)

                    # rotate
                    component_centre = component_img.shape[::-1][: 2] // 2
                    component_label_to_centre = Component.convert_TL_to_centre(component_centre, component_labels)
                    additional_rotation = this_rotation - component.chip_rotation
                    component_img, component_labels, _ = Augmentor._rotate(component_img, additional_rotation,
                                                                           component_label_to_centre)

                    # debug: show resized, flipped and rotated component with its surrounding box
                    if debug:
                        component.draw_box(f"rotated_{task.rotation}", component_img=component_img,
                                           component_label=component_labels)

                    # Generate the random position firstly without overlapping
                    # Embed the component on the background
                    # TODO: (stitch) compute the position for each chip in the structure
                    augmented_img, final_labels, pos_recorder = Augmentor.__embed_component(component_img,
                                                                                            augmented_img,
                                                                                            component_labels,
                                                                                            pos_recorder)
                    # Too hard to finish the task, skip this one
                    if augmented_img is None:
                        skip_flag = True
                        break

                    finished_labels[idx_struc].append(final_labels)  # labels for one component

                    # add new records for updating the database
                    if task_assigner.n_stitch == 1:
                        new_record = {
                            "Component_id": this_component_id,
                            "Background_id": task.background_id,
                            "Component_scale": round(this_required_area_scale, 2),
                            "Flip": this_flip,
                            "Rotate": this_rotation,
                        }
                        db_records.append(new_record)

                    # =========================== End of processing one chip / structure ============================ #

            # Skip the task
            if skip_flag:
                counter += 1

                # clean the unfinished task
                task_assigner.augmentation_task_pipeline.remove(task)
                continue

            # Save the image and its labels to the category it belongs to (training, validation, testing)
            category: str = ""

            if task_assigner.mode == AUGMENTATION:
                category = split_to_folder[task.split]

            # generate the image name
            save_name = "augmented_" + str(db_records[0]["Background_id"])
            component_id_str = []
            scale_str = []
            flip_str = []
            rotate_str = []

            for chip in db_records:
                component_id_str.append(str(chip["Component_id"]))
                scale_str.append(str(chip["Component_scale"]))
                flip_str.append(chip["Flip"])
                rotate_str.append(str(chip["Rotate"]))

            for value_list in [component_id_str, scale_str, flip_str, rotate_str]:
                save_name += "_"
                save_name += ",".join(value_list)

            # debug: show final result
            if debug:
                Background.draw_box(save_name, augmented_img, finished_labels)

            # images
            save_name_png = f"{save_name}.png"
            cv2.imwrite(os.path.join(save_path_root, category, "images", save_name_png), augmented_img)

            # labels
            save_name_txt = f"{save_name}.txt"
            with open(os.path.join(save_path_root, category, "labelTxt", save_name_txt), "w") as f:
                for one_chip_label in finished_labels:
                    # get all labels including chip location and active sites for one chip
                    for label_type, value_list in one_chip_label.items():
                        # iterate one type of labels
                        for label in value_list:
                            # get one label
                            for coordinates in label:
                                x, y = coordinates
                                f.write(f"{x:.6f}" + " ")
                                f.write(f"{y:.6f}" + " ")

                            f.write(label_type + " ")
                            f.write(str(task_assigner.difficult))
                            f.write("\n")

            # update the database
            for record in db_records:
                record["Image_name"] = save_name_png
                record["Category"] = category
                record["LabelTxt"] = save_name_txt

                db.select_table(task_assigner.dataset_name).insert_data(**record)

            # add it into the cache
            fast_create_data = {
                "img": augmented_img,
                "img_name": save_name_png,
                "labels": finished_labels,
                "data": db_records
            }

            img = AugmentedImage(category, **fast_create_data)
            name_augmented[save_name] = img

            finished_number += 1
            if finished_number == task_assigner.n_split[0]:
                # finish training part
                flags[0] = True

            if finished_number == sum(task_assigner.n_split[: 2]):
                flags[1] = True

            if finished_number == sum(task_assigner.n_split):
                flags[-1] = True

            if np.all(flags):
                logger.info(f">>> Successfully finish target {finished_number} tasks as expected.")
                break

            if finished_number % 200 == 0:
                logger.info(f"Finish {finished_number} tasks ...")

            counter += 1
            # ====================================== Finish one task ==================================== #

        # create the cache
        # TODO: (perf) create separate image and label caches -> save space
        if task_assigner.cache:
            cache_save_path = os.path.join(task_assigner.save_path, task_assigner.cache_save_dir)

            mkdir_if_not_exists(cache_save_path)

            cache_name = f'{task_assigner.dataset_name}_{datetime.now().strftime("%Y_%m_%d_%H:%M")}.pkl'
            with open(os.path.join(cache_save_path, cache_name), "wb") as f:
                pickle.dump(name_augmented, f)

            logger.info(f">>> Create a cache file for {cache_name}")

        # statistics
        if counter > task_assigner.expected_num:
            logger.warning(f"Totally finish {finished_number} tasks but expected {task_assigner.expected_num}.")

            if counter == task_assigner.max_try:
                logger.warning(f"The maximum attempt of {task_assigner.max_try} has reached.")

    @staticmethod
    def __stitch_along_row(chip_img: np.ndarray,
                           chip_labels: LabelsType,
                           stitched_img: np.ndarray,
                           stitched_labels: StitchLabelType,
                           stitch_pos: Tuple[int, int],
                           m_c_row: np.ndarray,
                           gap: float) -> Tuple[np.ndarray, LabelsType, np.ndarray]:
        neighbouring_origami_pos = (stitch_pos[0], stitch_pos[1] - 1)

        # based on the bottom line of the chip at the leftest position of the row
        # first compute the centre of the new chip wrt to the centre of the stitched image
        neighbouring_component_labels = stitched_labels[neighbouring_origami_pos]
        neighbour_chip_label = neighbouring_component_labels[DNA_ORIGAMI][0]
        neighbour_chip_label = Component.from_TL_to_centre(np.mean(neighbour_chip_label, axis=0),
                                                           neighbour_chip_label, cartesian=True)

        chip_label = chip_labels[DNA_ORIGAMI][0]
        chip_label = Component.from_TL_to_centre(np.mean(chip_label, axis=0), chip_label, cartesian=True)

        new_centre_x, new_centre_y = compute_new_centre_row(m_c_row[0], m_c_row[1], chip_label[-1],
                                                            neighbour_chip_label[-2], neighbour_chip_label[-1], gap)

        # stitch this new chip into the provided image
        stitched_img, stitched_labels, bounding_box = Augmentor.__embed_in_structure_row(
            neighbouring_origami_pos,
            stitch_pos,
            stitched_img,
            stitched_labels,
            chip_img,
            chip_labels,
            (new_centre_x, -new_centre_y)
        )

        return stitched_img, stitched_labels, bounding_box

    @staticmethod
    def __stitch_along_col(structure_row_img: List[np.ndarray],
                           stitch_labels: StitchLabelType,
                           gap: float) -> Tuple[np.ndarray, LabelsType]:
        # stitch row structures vertically
        # Calculate the canvas width and height
        max_width = max(rect.shape[1] for rect in structure_row_img)
        total_height = sum(rect.shape[0] for rect in structure_row_img) + gap * (len(structure_row_img) - 1)

        # Create an empty canvas
        canvas = np.zeros((total_height, max_width, 3), dtype=np.uint8)

        # Initialize the y-coordinate for placing rectangles
        y_offset = 0

        # Iterate through the rectangles and place them on the canvas
        for rect in structure_row_img:
            height, width = rect.shape[: 2]
            canvas[y_offset: y_offset + height, : width] = rect
            y_offset += height + gap

        # refactor labels in the LabelsType for cache creating or further data augmentation
        structure_labels = defaultdict(list)

        # DNA origami
        top_left = (0, 0)
        top_right = (max_width, 0)
        bottom_left = (0, total_height)
        bottom_right = (max_width, total_height)

        structure_labels[DNA_ORIGAMI].append(np.array([top_left, top_right, bottom_right, bottom_left]))

        # Active sites
        for _, chip_label in stitch_labels.items():
            for label_type, labels_list in chip_label.items():
                if label_type == ACTIVE_SITE:
                    for as_label in labels_list:
                        structure_labels[ACTIVE_SITE].append(as_label)

        return canvas, structure_labels

    @classmethod
    def __embed_in_structure_row(cls,
                                 neighbouring_origami_pos: PointImageType,
                                 stitch_pos: PointImageType,
                                 stitched_img: np.ndarray,
                                 existing_labels: StitchLabelType,
                                 stitched_chip: np.ndarray,
                                 stitched_label: LabelsType,
                                 chip_centre_to_neighbour_centre: PointCoordinateType) \
            -> Tuple[np.ndarray, LabelsType, np.ndarray]:
        # compute the centre of the neighbouring chip wrt the TL of the stitched image
        point_O = np.mean(stitched_label[DNA_ORIGAMI][0], axis=0)

        # compute the centre of the chip wrt the centre of the neighbouring chip in the stitched image
        neighbour_chip_label = existing_labels[neighbouring_origami_pos][DNA_ORIGAMI][0]
        point_P = chip_centre_to_neighbour_centre + np.mean(neighbour_chip_label, axis=0)

        # Compute translation between point P on the chip ready to stitch on the image and point O on the image
        translation_x = int(point_P[0] - point_O[0])
        translation_y = int(point_P[1] - point_O[1])

        # Determine the size of the canvas to enclose all information
        min_x = min(0, translation_x)
        min_y = min(0, translation_y)
        max_x = max(stitched_img.shape[1], translation_x + stitched_chip.shape[1])
        max_y = max(stitched_img.shape[0], translation_y + stitched_chip.shape[0])

        canvas_width = int(max_x - min_x)
        canvas_height = int(max_y - min_y)

        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Embed the image on the canvas
        canvas[-min_y: stitched_img.shape[0] - min_y, -min_x: stitched_img.shape[1] - min_x] = stitched_img
        # canvas[-min_y:image_A.shape[0] - min_y, -min_x:image_A.shape[1] - min_x][dilated_mask_r == 255] = image_A[
        #     dilated_mask_r == 255]

        # Embed the chip on the canvas along the row direction in the structure
        stitched_chip_label = stitched_label[DNA_ORIGAMI][0].astype(np.int32)

        mask = np.zeros_like(stitched_chip[:, :, 0], dtype=np.uint8)  # create a mask to only embed labeled region
        cv2.fillPoly(mask, [stitched_chip_label], 255)

        # Create a kernel for dilation due to imperfect manually labeled boxes
        # Define the dilation size (number of pixels to enlarge)
        dilation_size = 40
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel)

        # Embed of the new chip
        canvas[translation_y - min_y: translation_y + stitched_chip.shape[0] - min_y,
        translation_x - min_x: translation_x + stitched_chip.shape[1] - min_x][mask_dilated == 255] \
            = stitched_chip[mask_dilated == 255]

        # compute the stitched chip label in the stitched image
        new_stitched_chip_labels = defaultdict(list)
        for label_type, labels in stitched_label.items():
            for label in labels:
                new_label = label + np.array([translation_x, translation_y])
                new_stitched_chip_labels[label_type].append(new_label)

        # sanity check
        if stitch_pos in existing_labels.keys():
            raise Exception(f"Error: Save labels for the save chip at position {stitched_label} in the structure")
        existing_labels[stitch_pos] = new_stitched_chip_labels

        # find the extended bounding box enclosing the row structure
        rectangles = [label[DNA_ORIGAMI][0].astype(np.int32) for _, label in existing_labels.items()]
        all_corners = np.array([corner for rectangle in rectangles for corner in rectangle])
        rect = cv2.minAreaRect(all_corners)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # create a new mask to make a tight bounding canvas to row structure
        mask_canvas = np.zeros_like(canvas[:, :, 0], dtype=np.uint8)
        cv2.fillPoly(mask_canvas, [box.astype(np.int32)], 255)
        dilated_mask = cv2.dilate(mask_canvas, kernel)

        # find the bounding box of the dilated region
        y, x = np.where(dilated_mask == 255)
        min_x, min_y, max_x, max_y = np.min(x), np.min(y), np.max(x), np.max(y)

        # crop the region from the original image
        cropped_region = canvas[min_y:max_y + 1, min_x:max_x + 1]

        # plot
        if cls.debug:
            for pos, label_dict in existing_labels.items():
                if pos[0] == stitch_pos[0]:
                    Image.plot_labels(canvas, label_dict)

            pts = np.array([box]).reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(cropped_region, [pts], True, (255, 0, 0), 3)

            cv2.imwrite('combine_1.png', cropped_region)

        return canvas, new_stitched_chip_labels, box

    @staticmethod
    def __embed_component(component_img: np.ndarray,
                          background_img: np.ndarray,
                          component_labels: LabelsType,
                          position_recorder: np.ndarray) \
            -> Tuple[np.ndarray, LabelsType, np.ndarray] | Tuple[None, None, None]:
        updated_position_recorder = position_recorder.copy()

        # TODO: (stitch) position is computed by the label box of the chip
        new_centre_x, new_centre_y, updated_position_recorder = \
            Augmentor.__random_position(component_img,
                                        background_img.shape[0],
                                        updated_position_recorder)

        # Too hard to finish the task
        if new_centre_x is None:
            return None, None, None

        topLeft_corner_y = new_centre_y - component_img.shape[0] // 2
        topLeft_corner_x = new_centre_x - component_img.shape[1] // 2

        mask = cv2.cvtColor(component_img, cv2.COLOR_BGR2GRAY)
        # black regions in the component (due to the rotation of the component) -> black
        # non-black region -> white
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # if the component lies out of the background, generate another position
        # if Augmentor.__is_out(component_img.shape[: 2], background_img.shape[: 2], (new_centre_y, new_centre_x)):
        #     raise Exception(f"Error: Component falls out of the mosaic.")

        # invert this mask: black region -> white; non-black region -> black
        mask_inv = cv2.bitwise_not(mask)

        # Apply the inverted mask to the background itself
        # Black regions in the component -> region in the background passes
        # component image -> region in the background is blocked
        # replace the black region in the component with the one in the background
        bg_and_mask_inv = cv2.bitwise_and(background_img[
                                          topLeft_corner_y: topLeft_corner_y + component_img.shape[0],
                                          topLeft_corner_x: topLeft_corner_x + component_img.shape[1]
                                          ],
                                          background_img[
                                          topLeft_corner_y: topLeft_corner_y + component_img.shape[0],
                                          topLeft_corner_x: topLeft_corner_x + component_img.shape[1]
                                          ],
                                          mask=mask_inv)

        # Black region in the component -> blocked
        # Non-black region (component image) -> passes
        # fill in the background with the actual component image
        img_and_mask = cv2.bitwise_and(component_img, component_img, mask=mask)

        # Add two images: black regions in the component replaced by the background and leave the component
        result = cv2.add(bg_and_mask_inv, img_and_mask)

        # TODO: smooth the seam at boundaries
        # TODO: (stitch) later embedded chip cannot cover previous ones
        # Place the result back into the background
        background_img[
        topLeft_corner_y: topLeft_corner_y + component_img.shape[0],
        topLeft_corner_x: topLeft_corner_x + component_img.shape[1]] = result

        # convert labels in the component image reference system from TL to centre
        centre = component_img.shape[:2:-1] / 2
        component_labels_to_centre = Component.convert_TL_to_centre(centre, component_labels)

        # convert labels in the background image reference system from centre to TL
        component_labels = Component.convert_centre_to_TL((new_centre_x, new_centre_y),
                                                          component_labels_to_centre, False)

        return background_img, component_labels, updated_position_recorder

    @staticmethod
    def __random_position(component_img: np.ndarray,
                          background_size: int,
                          pos_recorder: np.ndarray) \
            -> Tuple[int, int, np.ndarray] | Tuple[None, None, None]:
        """

        :param component_img:
        :param background_size:
        :param pos_recorder:
        :return:
        """
        """
        First limitation:
        ----------------------------- 
        |                           |
        |    ...................    |
        |    .                 .    |
        |    ...................    |
        |                           |
        -----------------------------
        The component image cannot fall out of the background
        """
        img_height, img_width = component_img.shape[: 2]
        min_domain_h, max_domain_h = img_height / 2, background_size - (img_height / 2)
        min_domain_w, max_domain_w = img_width / 2, background_size - (img_width / 2)

        updated_position_recorder = pos_recorder.copy()

        # avoid to generate a centre falling into the existing chips
        counter = 0
        while True:
            counter += 1
            if counter >= Augmentor.patience:
                return None, None, None

            new_x, new_y = int(random.uniform(min_domain_w, max_domain_w)), \
                int(random.uniform(min_domain_h, max_domain_h))

            if updated_position_recorder[new_y, new_x] > 0:
                continue
            else:
                """
                Second limitation:
                +------------------------------+
                |                              |
                |                              |
                |               /\             |
                |              /  \            |
                |              \  /            |
                |               \/             |
                |                              |
                |        /\                    |
                |       /  \                   |
                |       \  /                   |
                |        \/                    |
                |                              |
                |                              |
                +------------------------------+
                The component image cannot overlap with existing ones
                """
                # extract the actual region occupied by the chip
                gray_scale_img = cv2.cvtColor(component_img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_scale_img, 0, 255, cv2.THRESH_BINARY)
                bool_mask = mask == 255
                compared_pos_recorder = np.zeros((background_size, background_size))

                topLeft_corner_y = new_y - img_height // 2
                topLeft_corner_x = new_x - img_width // 2

                # find the current occupied region in the augmented image
                compared_pos_recorder[topLeft_corner_y: topLeft_corner_y + img_height,
                topLeft_corner_x: topLeft_corner_x + img_width][bool_mask] += 255

                # temporary array to compare with the existing one
                _, updated_position_recorder_255 = cv2.threshold(updated_position_recorder, 0, 255, cv2.THRESH_BINARY)

                overlap = cv2.bitwise_and(updated_position_recorder_255, compared_pos_recorder)

                if np.any(overlap):
                    continue
                else:
                    pos_recorder[topLeft_corner_y: topLeft_corner_y + img_height,
                    topLeft_corner_x: topLeft_corner_x + img_width][bool_mask] += 1

                    # plt.imshow(pos_recorder, cmap='gray', vmin=0, vmax=1)
                    # plt.show()

                    return new_x, new_y, pos_recorder

    def __position_in_structure(self,
                                component_img: np.ndarray,
                                pos_recorder: np.ndarray,
                                pos_in_struc: List[Tuple[int, int]],
                                p2nm: float) -> Tuple[Tuple[int, int], np.ndarray]:
        pass

    # @staticmethod
    # def __find_corners(mask: np.ndarray) -> np.ndarray:
    #     """"""
    #     """
    #     cv2.RETR_TREE: retrieval mode creates a hierarchy of contours,
    #     cv2.CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments -> leaves only their end points
    #     """
    #     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     coordinates = contours[0].reshape(-1, 4, 2)  # (x ,y) <-> width, height
    #     # coordinates[:, [0, 1]] = coordinates[:, [1, 0]]  # (y, x)
    #
    #     return coordinates

    @staticmethod
    def __save_directory(mode: str, save_path: str):
        Augmentor.__path_exists_or_create(save_path)

        if mode == SIMPLE:
            Augmentor.__path_exists_or_create(save_path, "images")
            Augmentor.__path_exists_or_create(save_path, "labelTxt")
        elif mode == AUGMENTATION:
            Augmentor.__path_exists_or_create(save_path, "train/images")
            Augmentor.__path_exists_or_create(save_path, "train/labelTxt")
            Augmentor.__path_exists_or_create(save_path, "val/images")
            Augmentor.__path_exists_or_create(save_path, "val/labelTxt")
            Augmentor.__path_exists_or_create(save_path, "test/images")
            Augmentor.__path_exists_or_create(save_path, "test/labelTxt")

    @staticmethod
    def __id_to_image(category: str,
                      given_id: int,
                      name_img: Dict,
                      db: DatabaseManager) -> Union[Background, Component]:
        # TODO: (parallel) instead of querying the database all the time, fetch all required data once starting
        if category == BACKGROUND:
            col_name = "Background_name"
        else:
            col_name = "Sample"

        img_name = db.select_table(category).query_data(f"id = {given_id}", [col_name])

        try:
            image_name = img_name[0].split(".")[0]
            img = name_img[image_name]
        except Exception as e:
            raise KeyError(f"Cannot find image name {img_name} with a given error {e}")

        return img

    @staticmethod
    def __is_out(component_size: Tuple[int, int],
                 background_size: Tuple[int, int],
                 position: Tuple[int, int]) -> bool:
        """

        :param component_size:
        :param background_size:
        :param position: y, x
        :return:
        """
        half_height, half_width = [math.ceil(n) for n in np.divide(component_size, 2)]
        pos_y, pos_x = position
        height_domain, width_domain = background_size

        for h_op in [0, 1]:
            for w_op in [0, 1]:
                y = pos_y - half_height if h_op == 0 else pos_y + half_height
                x = pos_x - half_width if w_op == 0 else pos_x + half_width

                if 0 <= y <= height_domain and 0 <= x <= width_domain:
                    pass
                else:
                    return True

        return False

    @staticmethod
    def __is_larger(component_img: np.ndarray,
                    background_img: np.ndarray,
                    error_flag: bool = False) -> bool:
        if component_img.shape[0] >= background_img.shape[0] or component_img.shape[1] >= background_img.shape[1]:
            if error_flag:
                raise Exception(f"Error: Component image size is larger than the mosaic image size")
            else:
                return True
        else:
            return False

    @staticmethod
    def __path_exists_or_create(root_dir: str, directory: str = ""):
        save_path = os.path.join(root_dir, directory)

        if mkdir_if_not_exists(save_path):
            logger.warning(f"Directory {save_path} is not found")
            logger.info(f"Directory {save_path} is created")

    @staticmethod
    def __find_row_structure_label(stitched_row_labels: List[np.ndarray]):
        # all individual chips in one row structure with their corners
        rectangles = [label[DNA_ORIGAMI][0].astype(np.int32) for label in stitched_row_labels]
        all_corners = np.array([corner for rectangle in rectangles for corner in rectangle])

        # use the minimum area rectangle to enclose all the corners for chips
        rect = cv2.minAreaRect(all_corners)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        return box


if __name__ == "__main__":
    args = ArgumentParser.test_aug(GENERATE_EXTENDED_STRUCTURE)
    db = DatabaseManager("../data/DNA_augmentation", training_dataset_name=args.dataset_name)
    db.scan_and_update("../test_dataset", "../data", load_cache=True, cache_dir="test_cache")

    data_loader = DataLoader.initialise(img_path=args.img_path,
                                        dataset_path=args.dataset_path,
                                        save_path=args.save_path,
                                        cache_save_dir=args.cache_save_dir)

    # data_loader.load_cached_files(BACKGROUND, "../test_dataset/test_cache/background_2023_08_04_15:32.pkl")
    # data_loader.load_cached_files(CROPPED, "../test_dataset/test_cache/cropped_2023_08_19_00:39.pkl")
    #
    data_loader.load_backgrounds(0).load_cropped_components(labels="ordered_labels")

    # component
    # data_loader = DataLoader.initialise(args.img_path).load_raw_components()
    # task_assigner = TaskAssigner.component_task(args)
    # Augmentor.produce_components(data_loader, task_assigner, db)
    # db.close_connection()

    # backgrounds
    # task_assigner = TaskAssigner.background_task(args)
    # Augmentor.produce_backgrounds(data_loader, task_assigner, db)

    # augmentation
    # task_assigner = TaskAssigner.augmented_task(args, db)
    task_assigner = TaskAssigner.extended_structure_task(args, db)
    for task in task_assigner.augmentation_task_pipeline:
        print(task)
    Augmentor.produce_extended_structures(data_loader, task_assigner, db, args.debug)
    db.close_connection()
