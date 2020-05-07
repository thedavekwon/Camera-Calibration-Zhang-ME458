import glob
import cv2
import numpy as np

from constants import DATA_PATH, PATTERN_SIZE, SQUARE_SIZE, CORNER_PATH


def get_images():
    images = sorted(glob.glob(DATA_PATH + "/*.jpg"))
    for image in images:
        yield image.split("/")[-1].split(".")[0], cv2.imread(image, 0)  # greyscale


def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey()


def get_correspondence():
    W_def = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), dtype=np.float64)  # World Coordinate
    W_def[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2) * SQUARE_SIZE
    correspondences = []
    cnt = 0
    for image_name, image in get_images():
        retval, corners = cv2.findChessboardCorners(image, patternSize=PATTERN_SIZE)
        # detected
        if retval:
            corners = corners.reshape(-1, 2)  # image coordinmate
            ec = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(ec, PATTERN_SIZE, corners, retval)
            cv2.imwrite(CORNER_PATH + "/" + image_name + "_corners.jpg", ec)
            if corners.shape[0] == W_def.shape[0]:
                correspondences.append([corners.astype(np.int), W_def[:, :-1].astype(np.int)])
            cnt += 1
    return correspondences


if __name__ == "__main__":
    get_correspondence()
