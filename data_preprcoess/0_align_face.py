"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import sys
import shutil
import logging
from pathlib import Path

sys.path.append('..')

from utils import id_utils, io_utils

import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.ndimage
import dlib
import argparse
from tqdm import tqdm


def crop_img(img, det):
    min_x = det.left()
    max_x = det.right()
    min_y = det.top()
    max_y = det.bottom()

    return img[min_y:max_y, min_x:max_x]


def resize_img(img, min_out_size=512):
    h, w = img.size
    min_in_size = min([h, w])
    if min_in_size < min_out_size:
        factor = min_out_size / min_in_size
        img = img.resize((int(factor * h), int(factor * w)))
    else:
        factor = 1
    return img, factor


def get_landmark_of_right_person(filepath, predictor, person_identifier, min_id_size=0):
    """
    If person_identifier is None, return first keypoints found
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    orig_img = PIL.Image.open(filepath).convert('RGB')
    img, resize_factor = resize_img(orig_img)
    img = np.array(img)

    # img = dlib.load_rgb_image(str(filepath))
    dets = detector(img, 1)

    shape = None
    for k, d in enumerate(dets):
        cropped_img = crop_img(img, d)
        if max(cropped_img.shape[:2]) < min_id_size:
            logging.info(f'{filepath.name}: Face too small: {cropped_img.shape[:2]}, not using for anything.')
            continue

        if person_identifier is None or person_identifier.verify_id(cropped_img, filepath.name):
            logging.info(f'Found the person in {filepath.name}')
            shape = predictor(img, d)
            break

    t = list(shape.parts()) if shape is not None else []
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)

    lm_orig_coo = np.int64(np.round(lm / resize_factor))
    return np.array(orig_img), lm_orig_coo


def process_single_image(filepath, lnds_predictor, person_identifier=None,
                         output_size=1024, transform_size=4096, min_size=512,
                         enable_padding=True, min_id_size=0):
    img, lm = get_landmark_of_right_person(filepath, lnds_predictor, person_identifier, min_id_size)
    if len(lm) == 0:
        logging.info(f'{filepath.name}: Not using, landmarks not found')
        return None

    img = PIL.Image.fromarray(img)

    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    if max(img.size) < min_size:
        logging.info(f'{filepath.name}: Aligned face too small, not using for dataset. Face size: {img.size}')
        return None

    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                        PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    logging.info(f'done with {filepath.name}')
    # Return aligned image.
    return img


def parse_args(raw_args=None):
    parser = argparse.ArgumentParser(description="Filter ID and Align")
    parser.add_argument("--images_dir", type=Path, required=True,
                        help="The directory of the images to be aligned")
    parser.add_argument("--save_dir", type=Path, required=True,
                        help="The directory to save the aligned images")
    parser.add_argument("--trash_dir", type=Path, required=True,
                        help="The directory to save unused images")

    parser.add_argument("--landmarks_model", type=Path, required=True)
    parser.add_argument("--id_model", type=Path)

    parser.add_argument("--output_size", type=int, default=1024)

    parser.add_argument("--min_size", type=int, default=500)
    parser.add_argument("--min_id_size", type=int, default=200)
    parser.add_argument("--id_bank_size", type=int, default=20)
    parser.add_argument("--id_threshold", type=float, default=0.55)

    parser.add_argument("--test", action='store_true')

    args = parser.parse_args(raw_args)
    return args


def process_args(args):
    args.save_dir.mkdir(exist_ok=True, parents=True)
    args.trash_dir.mkdir(exist_ok=True, parents=True)

    if args.test:
        args.min_size = 0
        args.min_id_size = 0
        args.id_bank_size = np.inf
        args.id_threshold = 0

    return args


def main(raw_args=None):
    args = parse_args(raw_args)
    args = process_args(args)

    images_paths = io_utils.get_images_in_dir(args.images_dir)
    lnds_predictor = dlib.shape_predictor(str(args.landmarks_model))

    person_identifier = None
    if args.id_model:
        person_identifier = id_utils.PersonIdentifier(args.id_model, args.id_bank_size, args.id_threshold)

    for image_path in tqdm(images_paths):
        file_name = image_path.name
        try:
            aligned_img = process_single_image(image_path, lnds_predictor, person_identifier,
                                               output_size=args.output_size, min_size=args.min_size,
                                               min_id_size=args.min_id_size)
            if aligned_img is None:
                raise Exception('Alignment returned None')
            output_file = args.save_dir.joinpath(file_name)
            aligned_img.save(output_file)
        except Exception as e:
            output_file = args.trash_dir.joinpath(file_name)
            shutil.copy2(image_path, output_file)
            logging.info(f'Failed aligning {file_name}, because {e}')


if __name__ == "__main__":
    main()
