import math
import os.path
import random
from collections import defaultdict
from typing import List, Dict, Union, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.ArgumentParser import ArgumentParser
from src.Background import Background
from src.Component import Component
from src.DataLoader import DataLoader
from src.DatabaseManager import DatabaseManager
from src.Image import Image
from src.TaskAssigner import TaskAssigner
from src.constant import BACKGROUND, COMPONENT, TRAINING, SIMPLE, AUGMENTATION, VALIDATION, \
    split_converter

from src.DNALogging import DNALogging
import logging

DNALogging.config_logging()
logger = logging.getLogger(__name__)


class Augmentor:

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
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        max_id = cls.__get_current_num(BACKGROUND, db)
        counter = max_id + 1 if max_id != 0 else max_id

        logger.info(">>> Start to produce background images")
        for texture, tasks in task_assigner.background_task_pipeline.items():
            logger.info(f">>> Producing total {len(tasks)} {texture} backgrounds")

            for one_task in tasks:
                concat_whole_image = cls.__mosaics_in_row(one_task[0: task_assigner.num_per_side],
                                                          data_loader.background_img[texture],
                                                          task_assigner.kernel_size)

                for start_img in range(task_assigner.num_per_side, task_assigner.num_mosaic_in_background,
                                       task_assigner.num_per_side):
                    concat_row_image = cls.__mosaics_in_row(one_task[start_img: start_img + task_assigner.num_per_side],
                                                            data_loader.background_img[texture],
                                                            task_assigner.kernel_size)

                    concat_whole_image = cls.smooth_seam(concat_whole_image, concat_row_image,
                                                         task_assigner.kernel_size, axis=0)

                save_name = f"fake_{texture}_background_{counter}.png"
                cv2.imwrite(os.path.join(save_path, save_name), concat_whole_image)
                db.select_table("background").insert_data(Background_name=save_name, Texture=texture)

                counter += 1

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

        for operation in task_in_row:
            augmented_images.append(Augmentor.operate(operation, random.choice(mosaics_pool)))

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
        if operation_type in ['v', 'h', 'n']:
            # flip
            return Augmentor.__flip(image.read(), operation_type)[0]
        elif -180 <= operation_type <= 180:
            # rotate
            return Augmentor.__rotate(image.read(), operation_type)[0]
        else:
            raise Exception(f"Error: Incorrect operation {operation_type} is given")

    @staticmethod
    def __flip(image: np.array,
               direction: str,
               label: np.array = None) -> Tuple[np.array, np.array]:
        flipped_label = None

        if direction == 'v':
            flipped_image = cv2.flip(image, 0)

            if label is not None:
                flipped_label = np.array([[pos[0], image.shape[0] - pos[1]] for pos in label])
        elif direction == 'h':
            flipped_image = cv2.flip(image, 1)

            if label is not None:
                flipped_label = np.array([[image.shape[1] - pos[0], pos[1]] for pos in label])
        else:
            flipped_image = image

            if label is not None:
                flipped_label = label

        return flipped_image, flipped_label

    @staticmethod
    def __rotate(image: np.array,
                 angle: int,
                 label: np.array = None) -> Union[Tuple[np.array], Tuple[np.array, None]]:
        # positive angle -> counter clock-wise
        # negative angle -> clock-wise
        rows, cols = image.shape[:2]
        center = (cols / 2, rows / 2)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, angle, scale)  # rotation matrix

        # Calculate the size of the new image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((rows * sin) + (cols * cos))
        new_h = int((rows * cos) + (cols * sin))

        # Adjust the rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated_label = None
        # if labels given
        if label is not None:
            rotation_matrix_2D = M[:, :2]  # 2 x 2
            _label = label.T  # 2 x 4
            rotated_label = np.dot(rotation_matrix_2D, _label)  # 2 x 4
            rotated_label = rotated_label.T + np.array([new_w / 2, new_h / 2])

        return cv2.warpAffine(image, M, (new_w, new_h)), rotated_label

    # NOTE: The following function is adapted from Tejas' previous work.
    # This function was originally written by Tejas Narayan.
    # Source: https://github.com/ic-dna-storage/tn21-ic-msc-project/blob/main/code/image_analysis/cv/models/dna_origami.py#L187C11-L187C11
    @staticmethod
    def produce_components(data_loader: DataLoader,
                           task_assigner: TaskAssigner,
                           db: DatabaseManager) -> Dict[str, List[np.array]]:
        # https://pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/
        find_chips_config = task_assigner.config["find_dna_chip"]
        cropped_origami = defaultdict(list)
        total_chips = 0
        index_in_name = Augmentor.__get_current_num(COMPONENT, db)

        save_path = os.path.join(task_assigner.save_path, "component", "images")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        logger.info(f">>> Start to crop DNA origami to produce component")
        for input_img in data_loader.raw_input_img:
            logger.info(f"Process image {input_img.img_name}")

            img = input_img.read()
            # img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, 1)
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(img_grey, (find_chips_config["blur_radius"], find_chips_config["blur_radius"]),
                                       0)

            otsu_threshold, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 5)

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
                # FIXME: added height, width
                db.select_table(COMPONENT).insert_data(Raw_image=input_img.img_name, Sample=img_name, Texture="TEXTURE")

                cropped_origami[input_img.img_name].append(found_chip)
                index_in_name += 1

        logger.info(f"All {total_chips} are saved in {save_path}")

        return cropped_origami

    # NOTE: The following function is adapted from Tejas' previous work.
    # This function was originally written by Tejas Narayan.
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
    def produce_augmented(cls,
                          data_loader: DataLoader,
                          task_assigner: TaskAssigner,
                          db: DatabaseManager,
                          debug: bool = False):
        background_size = data_loader.background_img["clean"][0].img_size[0]
        scaled_height = background_size * task_assigner.initial_scale

        if debug:
            for row in task_assigner.augmentation_task_pipeline:
                print(row)

        # Saving directory
        save_path_root = os.path.join(task_assigner.save_path, task_assigner.dataset_name)
        Augmentor.__save_directory(task_assigner.mode, save_path_root)

        logger.info(">>> Start to augment the dataset")
        for task in tqdm(task_assigner.augmentation_task_pipeline):
            # find the corresponding component from its id
            component = cls.__id_to_image(COMPONENT, task.component_id, data_loader.name_component, db)

            # scale into initial size if not yet
            if not component.initial_scale:
                # based on the height
                adjusted_scale = scaled_height / component.img_size[0]

                # initial scale
                component.resize_into(int(component.img_size[1] * adjusted_scale), int(scaled_height))
                component.update_resizing(adjusted_scale)

                component.initial_scale = True

                # Debug
                if debug:
                    component.draw_box("init")

            # Rescale the component image
            if task.required_scale in component.scaled_image:
                # have rescaled before
                component_img = component.scaled_image[task.required_scale]
                component_label = component.scaled_labels[task.required_scale]
            elif task.required_scale == 1.0:
                component_img = component.read()
                component_label = component.corners
            else:
                # new scale
                scaled_size = tuple(int(x) for x in np.dot(component.img_size[: 2], task.required_scale))[::-1]
                component_img, component_label = component.add_resizing_res(scaled_size, task.required_scale)

            # debug
            if debug:
                component.draw_box(f"resize_{task.required_scale:.2f}", component_img=component_img,
                                   component_label=component_label)

            # background
            background = cls.__id_to_image(BACKGROUND, task.background_id, data_loader.name_background, db)
            background_img = background.read()

            # check if component is larger than the background
            Augmentor.__is_larger(component_img, background_img, error_flag=True)

            # flip
            if task.flip == "n":
                # no flip
                pass
            elif task.required_scale == 1:
                # just flip
                if task.required_scale in component.flipped_image:
                    # original scale but have flipped before
                    component_img = component.flipped_image[task.flip]
                    component_label = component.flipped_label[task.flip]
                else:
                    # not flipped before
                    component_img, component_label = cls.__flip(component_img, task.flip, component_label)
                    component.add_flipping_res(task.flip, component_img, component_label)
            else:
                # rescaled and flip
                component_img, component_label = cls.__flip(component_img, task.flip, component_label)
            if debug:
                component.draw_box(f"flipped_{task.flip}", component_img=component_img, component_label=component_label)

            # rotate
            component_label_centre = Component.convert_TL_to_centre(component_img.shape[: 2], component_label)
            component_img, component_label = Augmentor.__rotate(component_img, task.rotation, component_label_centre)

            if debug:
                component.draw_box(f"rotated_{task.rotation}", component_img=component_img,
                                   component_label=component_label)

            # position
            task.position = Augmentor.__random_position(component_img.shape[: 2], background_size)

            # if the component lies out of the background, generate another position
            if Augmentor.__is_out(component_img.shape[: 2], background_img.shape[: 2], task.position):
                print(task)
                raise Exception(f"Error: Component falls out of the background.")

            # Embed the component on the background
            name_placeholder = f"{task.component_id}_{task.background_id}_{task.required_scale:.2f}_{task.flip}_" \
                               f"{task.rotation}"
            save_name = f"augmented_{name_placeholder}"

            final_img, final_label = Augmentor.__embed_component(component_img, component_label, background_img,
                                                                 task.position)

            # debug
            if debug:
                Background.draw_box(name_placeholder, final_img, final_label)

            # TODO: connect to the database

            # Save the image and its label based on the category it belongs to (training, validation, testing)
            category = ""
            if task_assigner.mode == AUGMENTATION:
                if task.split == TRAINING:
                    category = "train"
                elif task.split == VALIDATION:
                    category = "val"
                else:
                    category = "test"

            # images
            cv2.imwrite(os.path.join(save_path_root, category, "images", f"{save_name}.png"), final_img)

            # labels
            with open(os.path.join(save_path_root, category, "labelTxt", f"{save_name}.txt"), "w") as f:
                for coordinate in final_label:
                    x, y = coordinate
                    f.write(f"{x:.6f}" + " ")
                    f.write(f"{y:.6f}" + " ")

                f.write(task_assigner.label + " ")
                f.write(str(task_assigner.difficult))

            # update the database
            new_record = {
                "Image_name": f"{save_name}.png",
                "Component_id": task.component_id,
                "Background_id": task.background_id,
                "Component_scale": round(task.required_scale, 2),
                "Flip": task.flip,
                "Rotate": task.rotation,
                "LabelTxt": f"{save_name}.txt",
                "Category": split_converter[task.split]
            }

            db.select_table(task_assigner.dataset_name).insert_data(**new_record)

    @staticmethod
    def __save_directory(mode: str, save_path: str):
        Augmentor.__path_exist_or_create(save_path)

        if mode == SIMPLE:
            Augmentor.__path_exist_or_create(save_path, "images")
            Augmentor.__path_exist_or_create(save_path, "labelTxt")
        elif mode == AUGMENTATION:
            Augmentor.__path_exist_or_create(save_path, "train/images")
            Augmentor.__path_exist_or_create(save_path, "train/labelTxt")
            Augmentor.__path_exist_or_create(save_path, "val/images")
            Augmentor.__path_exist_or_create(save_path, "val/labelTxt")
            Augmentor.__path_exist_or_create(save_path, "test/images")
            Augmentor.__path_exist_or_create(save_path, "test/labelTxt")

    @staticmethod
    def __path_exist_or_create(root_dir: str, directory: str = ""):
        save_path = os.path.join(root_dir, directory)

        if not os.path.exists(save_path):
            logger.warning(f"Directory {save_path} is not found")
            logger.info(f"Directory {save_path} is created")
            os.makedirs(save_path)

    @staticmethod
    def __random_position(component_size: Tuple[int, int], background_size: int) -> Tuple[int, int]:
        img_height, img_width = component_size

        # half of the diagonal as the minimum distance from the centre of the component to the edge of the background
        half_diagonal = math.ceil(math.sqrt(img_height ** 2 + img_width ** 2) / 2)
        min_domain, max_domain = half_diagonal, background_size - half_diagonal

        return int(random.uniform(min_domain, max_domain)), int(random.uniform(min_domain, max_domain))

    @staticmethod
    def __is_larger(component_img: np.array,
                    background_img: np.array,
                    error_flag: bool = False) -> bool:
        if component_img.shape[0] >= background_img.shape[0] or component_img.shape[1] >= background_img.shape[1]:
            if error_flag:
                raise Exception(f"Error: Component image size is larger than the background image size")
            else:
                return True
        else:
            return False

    @staticmethod
    def __is_out(component_size: Tuple[int, int],
                 background_size: Tuple[int, int],
                 position: Tuple[int, int]) -> bool:
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
    def __id_to_image(category: str, given_id: int, name_img: Dict, db: DatabaseManager) -> Union[Background, Component]:
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
    def __embed_component(component_img: np.array,
                          component_label: np.array,
                          background_img: np.array,
                          position: Tuple[int, int]) -> np.array:
        new_centre_x, new_centre_y = position
        topLeft_corner_y = new_centre_y - component_img.shape[0] // 2
        topLeft_corner_x = new_centre_x - component_img.shape[1] // 2

        mask = cv2.cvtColor(component_img, cv2.COLOR_BGR2GRAY)

        # black regions in the component -> black
        # non-black region -> white
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # invert this mask: black region -> white; non-black region -> black
        mask_inv = cv2.bitwise_not(mask)

        # Apply the inverted mask to the background itself
        # Black regions in the component -> pass
        # Component -> blocked
        bg_and_mask_inv = cv2.bitwise_and(background_img[
                                          topLeft_corner_y: topLeft_corner_y + component_img.shape[0],
                                          topLeft_corner_x: topLeft_corner_x + component_img.shape[1]
                                          ],
                                          background_img[
                                          topLeft_corner_y: topLeft_corner_y + component_img.shape[0],
                                          topLeft_corner_x: topLeft_corner_x + component_img.shape[1]
                                          ],
                                          mask=mask_inv)
        # Black region -> block
        # Non-black region -> pass
        img_and_mask = cv2.bitwise_and(component_img, component_img, mask=mask)

        # Add two images: black regions in the component replaced by the background and leave the component
        result = cv2.add(bg_and_mask_inv, img_and_mask)

        # TODO: smooth the seam at boundaries
        # Place the result back into the background
        background_img[
        topLeft_corner_y: topLeft_corner_y + component_img.shape[0],
        topLeft_corner_x: topLeft_corner_x + component_img.shape[1]
        ] = result

        # label
        component_label_centre = Component.convert_TL_to_centre(component_img.shape[: 2], component_label)
        component_label = component_label_centre + np.array(position)

        return background_img, component_label

    @staticmethod
    def __chip_to_background(corners: List[Tuple[int, int]], ):
        pass


if __name__ == "__main__":
    args = ArgumentParser.test_args()
    db = DatabaseManager("../data/DNA_augmentation")
    db.scan_and_update(args.dataset_path)

    data_loader = DataLoader.initialise(args.img_path, args.dataset_path).load_backgrounds(0).load_components()

    # component
    # data_loader = DataLoader.initialise(args.img_path).load_raw_components()
    # task_assigner = TaskAssigner.component_task(args)
    # Augmentor.produce_components(data_loader, task_assigner, db)
    # db.close_connection()

    # backgrounds
    # task_assigner = TaskAssigner.background_task(args)
    # Augmentor.produce_backgrounds(data_loader, task_assigner, db)

    # augmentation
    task_assigner = TaskAssigner.augmented_task(args, db)
    Augmentor.produce_augmented(data_loader, task_assigner, db)

    db.close_connection()
