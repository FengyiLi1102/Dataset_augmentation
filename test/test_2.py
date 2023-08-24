from collections import defaultdict
from Augmentor import *
from utils import *
import cv2
import numpy as np
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
from shapely.geometry import Polygon


def draw(img, label, name, points=None):
    imgc = img.copy()
    label_place_holder = label

    Image.plot_labels(imgc, label_place_holder)

    if points:
        for point in points:
            point = [int(p) for p in point]
            cv2.circle(imgc, point, 10, (0, 0, 255), thickness=-1)
    cv2.imwrite(f"test_{name}.png", imgc)


def find_cc(label):
    chip_centre_x, chip_centre_y = label.mean(axis=0)

    return chip_centre_x, chip_centre_y


degree = 0

chip_1_name = "cropped_regular_component_7"
chip_2_name = "cropped_irregular_component_36"

chip = cv2.imread(f"../test_dataset/cropped/images/{chip_1_name}.png")
chip_c = cv2.imread(f"../test_dataset/cropped/images/{chip_2_name}.png")

labels = defaultdict(list)
with open(f"../test_dataset/cropped/ordered_labels/{chip_1_name}.txt", "r") as f:
    for line in f:
        values = line.strip().split(" ")[: -1]

        # (x, y) in (width, height)
        # Two objects: DNA-origami and active-site
        labels[values[-1]].append(np.array(values[: -1], dtype=float).reshape(-1, 2))

labels_c = defaultdict(list)
with open(f"../test_dataset/cropped/ordered_labels/{chip_2_name}.txt", "r") as f:
    for line in f:
        values = line.strip().split(" ")[: -1]

        # (x, y) in (width, height)
        # Two objects: DNA-origami and active-site
        labels_c[values[-1]].append(np.array(values[: -1], dtype=float).reshape(-1, 2))

img_centre = chip.shape[1] / 2, chip.shape[0] / 2
img_centre_c = chip_c.shape[1] / 2, chip_c.shape[0] / 2

labels_to_centre = Component.convert_TL_to_centre(img_centre, labels)
i_r = compute_angle_between_horizontal(labels=labels[DNA_ORIGAMI][0])  # self convert to CC
print("initial rotation for stitch:", i_r)

img, labels_new, M = Augmentor._rotate(chip, degree - i_r, labels=labels_to_centre)
# draw(img, labels_new, "rotated_c1")
labels_new_to_centre = Component.convert_TL_to_centre(img_centre, labels_new)
chip_rotation = compute_angle_between_horizontal(labels=labels_new_to_centre[DNA_ORIGAMI][0])
print("final: rotation:", chip_rotation)

labels_c_centre = Component.convert_TL_to_centre(img_centre_c, labels_c)
fr = degree - compute_angle_between_horizontal(labels=labels_c_centre[DNA_ORIGAMI][0])
print("rotation for chip:", fr)
img_c, labels_new_c, M_c = Augmentor._rotate(chip_c, fr, labels=labels_c_centre)
img_new_c_centre = img_c.shape[1] / 2, img_c.shape[0] / 2
labels_new_c_centre = Component.convert_TL_to_centre(img_new_c_centre, labels_new_c)
c_r = compute_angle_between_horizontal(labels=labels_new_c_centre[DNA_ORIGAMI][0])

print("final rotation of chip:", c_r)

chip_centre = find_cc(labels_new[DNA_ORIGAMI][0])  # to TL
print(chip_centre)
draw(img, labels_new, "rotated_c1", points=[chip_centre])
labels_to_chip_centre_d = Component.convert_TL_to_centre(chip_centre, labels_new, cartesian=True)
print(labels_to_chip_centre_d[DNA_ORIGAMI][0])
# build the coordinate with the chip centre
length = np.linalg.norm(labels_to_chip_centre_d[DNA_ORIGAMI][0][-1] - labels_to_chip_centre_d[DNA_ORIGAMI][0][-2])
m, c = compute_m_c(labels_to_chip_centre_d[DNA_ORIGAMI][0][-1], labels_to_chip_centre_d[DNA_ORIGAMI][0][-2], cartesian=True)
# print("m: ", m)
# print("c: ", c)

chip_c_centre = find_cc(labels_new_c[DNA_ORIGAMI][0])
draw(img_c, labels_new_c, "test_c", points=[chip_c_centre])
labels_c_to_chip_centre_d = Component.convert_TL_to_centre(chip_c_centre, labels_new_c, cartesian=True)
# print(labels_c_to_chip_centre_d[DNA_ORIGAMI][0])

gap = length / 70 * 0.5
print(gap)

stitched_labels = dict()
# stitched_labels[(0, 0)] = defaultdict(list)
# for key, values in labels_new.items():
#     for label in values:
#         nn = label
#         nn[:, 1] -=
#         stitched_labels[(0, 0)][key].append(label)
i = 0


def plot(image_A, image_B, point_P, point_O, chip_c_labels, chip_labels, stitched_labels, i):
    # Calculate translation between point P and point O
    translation_x = int(point_P[0] - point_O[0])
    translation_y = int(point_P[1] - point_O[1])

    # Determine the size of the canvas
    min_x = min(0, translation_x)
    min_y = min(0, translation_y)
    max_x = max(image_A.shape[1], translation_x + image_B.shape[1])
    max_y = max(image_A.shape[0], translation_y + image_B.shape[0])

    canvas_width = int(max_x - min_x)
    canvas_height = int(max_y - min_y)

    # Create the canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    # Define the dilation size (number of pixels to enlarge)
    dilation_size = 20

    # Create a kernel for dilation
    kernel = np.ones((dilation_size, dilation_size), np.uint8)

    # canvas[-min_y:image_A.shape[0] - min_y, -min_x:image_A.shape[1] - min_x][dilated_mask_r == 255] = image_A[
    #     dilated_mask_r == 255]
    canvas[-min_y:image_A.shape[0] - min_y, -min_x:image_A.shape[1] - min_x] = image_A
    cv2.imwrite("tt.png", canvas)

    chip_c_label = chip_c_labels[DNA_ORIGAMI][0].astype(np.int32)
    mask_T = np.zeros_like(image_B[:, :, 0], dtype=np.uint8)
    cv2.fillPoly(mask_T, [chip_c_label], 255)
    mask_T_dilated = cv2.dilate(mask_T, kernel)
    canvas[translation_y - min_y: translation_y + image_B.shape[0] - min_y,
    translation_x - min_x: translation_x + image_B.shape[1] - min_x][mask_T_dilated == 255] = image_B[
        mask_T_dilated == 255]

    label_B = defaultdict(list)
    for key, value in chip_c_labels.items():
        for label in value:
            nn = label + np.array([translation_x, translation_y])
            # nn[:, 1] -= translation_y
            label_B[key].append(nn)
    stitched_labels[(0, i + 1)] = label_B

    if i == 0:
        stitched_labels[(0, 0)] = chip_labels

    rectangles = []
    for key, label in stitched_labels.items():
        nn = label[DNA_ORIGAMI][0] * np.array([1, -1])
        nn = nn.astype(np.int32)
        rectangles.append(nn)

    rectangles = [label[DNA_ORIGAMI][0].astype(np.int32) for _, label in stitched_labels.items()]
    all_corners = np.array([corner for rectangle in rectangles for corner in rectangle])
    rect = cv2.minAreaRect(all_corners)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    mask = np.zeros_like(canvas[:, :, 0], dtype=np.uint8)

    # Fill the region defined by the corners with white color
    cv2.fillPoly(mask, [box.astype(np.int32)], 255)

    dilated_mask = cv2.dilate(mask, kernel)
    # Find the bounding box of the dilated region
    y, x = np.where(dilated_mask == 255)
    min_x, min_y, max_x, max_y = np.min(x), np.min(y), np.max(x), np.max(y)

    # Crop the region from the original image
    cropped_region = canvas[min_y:max_y + 1, min_x:max_x + 1]

    pts = np.array([box]).reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(cropped_region, [pts], True, (0, 0, 255), 2)


    cv2.imwrite(f'combine_{i}.png', cropped_region)

    return label_B, stitched_labels


def crop(imga, label, dilation=30):
    img = imga.copy()
    canvas = np.zeros_like(img, dtype=np.uint8)
    chip_label = label[DNA_ORIGAMI][0].astype(np.int32)
    mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
    cv2.fillPoly(mask, [chip_label], 255)
    kernel = np.ones((dilation, dilation), dtype=np.uint8)
    dilated_mask_r = cv2.dilate(mask, kernel)

    # Find the tight bounding box
    ys, xs = np.where(dilated_mask_r == 255)
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)

    # Crop the canvas to the bounding box
    canvas[dilated_mask_r == 255] = img[dilated_mask_r == 255]
    cropped_canvas = canvas[min_y:max_y+1, min_x:max_x+1]

    # Update the label coordinates
    new_label_c = label.copy()
    for key, value in new_label_c.items():
        for la in value:
            la -= np.array([min_x, min_y])

    # pts = np.array(new_label).reshape((-1, 1, 2)).astype(np.int32)
    # cv2.polylines(cropped_canvas, [pts], True, (0, 0, 255), 2)
    # Image.plot_labels(cropped_canvas, new_label_c)
    # cv2.imwrite("crop.png", cropped_canvas)

    return cropped_canvas, new_label_c


img, new_label = crop(img, labels_new)

nlcx, nlcy = np.mean(new_label[DNA_ORIGAMI][0], axis=0)
new_label_d = Component.from_TL_to_centre((nlcx, nlcy), new_label[DNA_ORIGAMI][0], cartesian=True)

x, y = compute_new_centre_row(m,
                              c,
                              labels_c_to_chip_centre_d[DNA_ORIGAMI][0][-1],
                              new_label_d[-2],
                              new_label_d[-1],
                              gap
                              )
print("new position: ", x, y)
xl, yl = labels_c_to_chip_centre_d[DNA_ORIGAMI][0][-1] + np.asarray((x, y))
print("new bottom left corner: ", xl, yl)
print("gap distance: ", np.linalg.norm(labels_to_chip_centre_d[DNA_ORIGAMI][0][-2] - (xl, yl)))

labels_T, stitched_labels = plot(img, img_c, (x + nlcx, -y + nlcy), chip_c_centre,
                                 labels_new_c,
                                 new_label, stitched_labels, 0)  # a large box to label the structure to TL

new_c_x, new_c_y = np.mean(labels_T[DNA_ORIGAMI][0], axis=0)
label_T_to_chip_centre_d = Component.from_TL_to_centre((new_c_x, new_c_y), labels_T[DNA_ORIGAMI][0], cartesian=True)
m_new, c_new = compute_m_c(label_T_to_chip_centre_d[-1], label_T_to_chip_centre_d[-2])
# x, y = compute_new_centre_row(m,
#                               c,
#                               labels_c_to_chip_centre_d[DNA_ORIGAMI][0][-1],
#                               labels_to_chip_centre_d[DNA_ORIGAMI][0][-2],
#                               labels_to_chip_centre_d[DNA_ORIGAMI][0][-1],
#                               gap)

new_x, new_y = compute_new_centre_row(m_new,
                                      c_new,
                                      labels_c_to_chip_centre_d[DNA_ORIGAMI][0][-1],
                                      label_T_to_chip_centre_d[-2],
                                      label_T_to_chip_centre_d[-1],
                                      gap)
print(new_x, new_y)

simg = cv2.imread(f"combine_{i}.png")

# labels_T = plot(img, img_c, (x + chip_centre[0], -y + chip_centre[1]), chip_c_centre,
#                 labels_new_c[DNA_ORIGAMI][0],
#                 labels_new[DNA_ORIGAMI][0])

labels_T, stitched_labels = plot(simg, img_c, (new_x + new_c_x, -new_y + new_c_y), chip_c_centre,
                                 labels_new_c,
                                 labels_T, stitched_labels, 1)

print(len(stitched_labels))

col_1 = cv2.imread("combine_1.png")
col_2 = col_1.copy()
col_3 = col_1.copy()

col_imgs = [col_1, col_2, col_3]


def align(structure_row_img, gap = 10):
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

    cv2.imwrite("whole.png", canvas)

align(col_imgs, 10)
