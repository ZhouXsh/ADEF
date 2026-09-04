from pathlib import Path

import cv2
import dlib
import numpy as np
from skimage import transform as tf

HERE = Path(__file__).resolve().parent
PREDICTOR_CANDIDATES = [
    HERE / "shape_predictor_68_face_landmarks.dat",
    HERE.parent / "checkpoints" / "shape_predictor_68_face_landmarks.dat",
]
PREDICTOR_PATH = next((p for p in PREDICTOR_CANDIDATES if p.is_file()), PREDICTOR_CANDIDATES[0])
BASE_68 = HERE / "base_68.npy"
BASE_68_CLOSE = HERE / "base_68_close.npy"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(PREDICTOR_PATH))


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def crop_image(image, resize_size=163, crop_ratio=0.64):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None, None, False
    for rect in rects:
        shape = shape_to_np(predictor(gray, rect))
        x, y, w, h = rect_to_bb(rect)
        center_x = x + int(0.5 * w)
        center_y = y + int(0.5 * h)
        r = int(crop_ratio * h)
        new_x = center_x - r
        new_y = center_y - r
        roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]
        try:
            roi = cv2.resize(roi, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
        except Exception:
            return None, None, False
        scale = resize_size / (2 * r)
        shape = (shape - np.array([new_x, new_y])) * scale
        return roi, shape, True
    return None, None, False


def crop_and_align(image):
    roi, landmark, ret = crop_image(image)
    if not ret:
        return False, False
    template_path = BASE_68 if np.sum(landmark[37:39, 1] - landmark[40:42, 1]) < -9 else BASE_68_CLOSE
    template = np.load(template_path)
    pts2 = np.float32(template[27:45, :])
    pts1 = np.float32(landmark[27:45, :])
    tform = tf.SimilarityTransform()
    tform.estimate(pts2, pts1)
    dst = tf.warp(roi, tform, output_shape=(163, 163))
    dst = np.array(dst * 255, dtype=np.uint8)
    dst = dst[1:129, 1:129, :]
    return dst, True
