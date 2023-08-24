import glob
import math
import os
from collections import defaultdict
from typing import Union, Tuple, Dict, List, Callable

import cv2
import numpy as np

from src.constant import DNA_ORIGAMI
from src.typeHint import PointImageType, LabelsType, PointCoordinateType


def mkdir_if_not_exists(target_path: str) -> bool:
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        return True
    else:
        return False


def process_labels(param: np.ndarray | PointImageType,
                   labels: LabelsType,
                   operation_func: Callable,
                   cartesian: bool) -> LabelsType:
    processed_labels = defaultdict(list)

    for label_type in labels:
        for i in range(len(labels[label_type])):
            processed_labels[label_type].append(operation_func(param, labels[label_type][i], cartesian))

    return processed_labels


def ratio_to_number(ratio: List[float], num: int):
    return [int(ratio[0] / 10 * num), int(ratio[1] / 10 * num), int(ratio[-1] / 10 * num)]


def concatenate_txt(dir_1: str, dir_2: str, save_dir: str):
    mkdir_if_not_exists(save_dir)
    # Get the list of txt files in the new label folder
    new_label_txts = [f for f in os.listdir(dir_1) if f.endswith('.txt')]

    for txt in new_label_txts:
        # Check if the file also exists in the existing label files
        if txt in os.listdir(dir_2):
            with open(os.path.join(dir_1, txt), 'r') as f1, \
                    open(os.path.join(dir_2, txt), 'r') as f2, \
                    open(os.path.join(save_dir, txt), 'w') as fout:
                lines1 = f1.readlines()
                lines2 = f2.readlines()
                fout.writelines(lines1)
                fout.write("\n")
                fout.writelines(lines2)


def compute_m_c(point_1: np.ndarray, point_2: np.ndarray, cartesian: bool = False) -> Tuple[float, float]:
    """
    Point must be in the property Cartesian coordinate.
    :param cartesian:
    :param point_1:
    :param point_2:
    :return:
    """
    if not cartesian:
        transformer = np.array([1, -1])
        p1 = point_1 * transformer
        p2 = point_2 * transformer
    else:
        p1 = point_1
        p2 = point_2

    x_coords, y_coords = zip(*[p1, p2])
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]

    return m, c


def compute_angle_between_horizontal(m: float | None = None,
                                     labels: np.ndarray | None = None,
                                     cartesian: bool = False) -> float:
    """

    :param m:
    :param labels: location of the origami structure
    :return:
    """
    if m:
        pass
    elif labels is not None:
        # labels in the img reference
        m, _ = compute_m_c(labels[-1], labels[-2], cartesian=cartesian)
    else:
        raise ValueError("None correct parameter is provided to compute")

    return math.degrees(math.tan(m))


def compute_m_from_angle(angle: float) -> float:
    return math.tan(math.radians(angle))


# https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
def find_clockwise_order(points: np.ndarray) -> np.ndarray:
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2))
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def order_corners_in_DOTA_format(label_path: str, save_path: str, img_path: str | None = None):
    mkdir_if_not_exists(save_path)
    new_label_txts = [f for f in os.listdir(label_path) if f.endswith('.txt')]

    # Define the radius of the dots
    radius = 5

    for txt in new_label_txts:
        with (open(os.path.join(label_path, txt), 'r') as f1, open(os.path.join(save_path, txt), 'w') as fout):
            counter = 0

            for line in f1:
                values = line.strip().split(" ")
                ordered_coords = find_clockwise_order(
                    np.asarray([round(float(s), 6) for s in values[: -2]]).reshape((4, 2)))
                coords_list = ordered_coords.flatten().tolist()
                line = ' '.join(map(str, coords_list + values[-2:]))
                fout.writelines(line)
                fout.write("\n")

                if img_path and counter == 0:
                    img_name = txt[: -4] + ".png"
                    img = cv2.imread(os.path.join(img_path, img_name))

                    # Define the color for the red dots (in BGR format)
                    # red, green, blue, yellow
                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]

                    # Draw the red dots at the specified points
                    draw_point_arr = ordered_coords.astype(np.int32)

                    for point, color in zip(draw_point_arr, colors):
                        cv2.circle(img, point, radius, color, thickness=-1)

                    cv2.imwrite(os.path.join(save_path, img_name), img)

                counter += 1


def compute_new_centre_row(m: float,
                           c: float,
                           bottom_left_corner: PointCoordinateType,
                           bottom_right_corner: PointCoordinateType,
                           end_corner: PointCoordinateType,
                           gap_w: float) -> PointCoordinateType:
    """

    :param m:
    :param c:
    :param bottom_left_corner: wrt the chip centre and in cartisian coordinate
    :param bottom_right_corner:
    :param end_corner:
    :param gap_w:
    :return:
    """
    # Given the m and c of the bottom side in line equation for the chip at the start of the row in the structure
    # NOTE: this bottom side will not literally bottom due to the rotation over 90 degree. It is already defined
    # initially when the chip is loaded at the first time.

    # Given the bottom left corner (blc) of the new chip with respect to (wrt) the centre of the chip

    """
    Purpose:
        The given point is a bottom left corner with respect to its chip centre, and now the new centre of this chip
        with respect to the universal origin (X, Y) is needed to know for further stitching process.

    Method:
        bottom line equation: y = mx + c

        If the bottom left corner of the new chip lies on this line:
            blc_y + Y = m * (blc_x + X) + c     ------------------------------------------------------ (1)
        where we have two variables.

        With a given gap width, the distance from the bottom left corner to the bottom right corner of the neighbour
        stitched chip can be computed:
            || P_blc - P_brc || = w     -------------------------------------------------------------- (2)

        Define a = blc_x - brc_x, b = blc_y - brc_y, substitute these and (1) into (2):
            Ax**2 + Bx + C = 0      ------------------------------------------------------------------ (3)
        where
            A = 1 + m
            B = 2 * [a + m * (b + m * blc_x + c - blc_y)]
            C = (m * blc_x + c - blc_y)(m * blc_x + c - blc_y + 2 * b) + a**2 + b**2 - w**2

        Solve (3) to obtain x_1 and x_2, and reject the value lying within the stitched chip by:
            min(end_corner) <= x <= max(end_corner)
    """
    blc_x, blc_y = bottom_left_corner
    brc_x, brc_y = bottom_right_corner

    a = blc_x - brc_x
    b = blc_y - brc_y

    common_term = m * blc_x + c - blc_y

    A = 1 + m**2
    B = 2 * (a + m * (b + common_term))
    C = common_term * (common_term + 2 * b) + a**2 + b**2 - gap_w**2

    discriminant = B ** 2 - 4 * A * C

    if discriminant < 0:
        raise Exception("No real solutions")

    x_1 = (-B + math.sqrt(discriminant)) / (2 * A)
    x_2 = (-B - math.sqrt(discriminant)) / (2 * A)

    xl, yl = blc_x + x_2, m * (blc_x + x_2) + c
    # print("x_2 new bottom left corner: ", xl, yl)
    # print("distance: ", np.linalg.norm(np.array(bottom_right_corner) - (xl, yl)))

    xll, yll = blc_x + x_1, m * (blc_x + x_1) + c
    # print("x_1 new bottom left corner: ", xll, yll)
    # print("distance: ", np.linalg.norm(np.array(bottom_right_corner) - (xll, yll)))

    # print("X:", x_1, x_2)

    min_end_x = min(end_corner[0], bottom_right_corner[0])
    max_end_x = max(end_corner[0], bottom_right_corner[0])

    # print(min_end_x, max_end_x)

    X, Y = None, None

    for x in [x_1, x_2]:
        new_blc_x = blc_x + x

        if min_end_x <= new_blc_x <= max_end_x:
            continue
        else:
            X = x
            Y = m * (blc_x + X) + c - blc_y
            break

    if not X or not Y:
        raise Exception("Error: Position of the new stitched chip is computed incorrectly.")

    # print(compute_m_c(np.array([new_blc_x, Y + blc_y]), np.array(bottom_right_corner)))

    return round(X), round(Y)


if __name__ == "__main__":

    pass
