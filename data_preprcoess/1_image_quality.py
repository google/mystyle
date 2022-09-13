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
import argparse
import warnings
import functools
import multiprocessing
from pathlib import Path

sys.path.append('..')
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import io_utils

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import imquality.brisque as brisque


def decide_on_img(queue, save_dir, trash_path, filters):
    while True:
        msg = queue.get()
        if msg == 'END':
            break

        f = msg
        try:
            img = Image.open(f)
        except:
            print(f'Failed opening {f.name}')
            shutil.copy2(f, trash_path)
            continue

        for filter in filters:
            if filter(img) is False:
                shutil.copy2(f, trash_path)
                print(f'trashed {f}')
                break
        else:
            shutil.copy2(f, save_dir)
            print(f'saved {f}')


def gray_image_filter(img: Image.Image, gray_threshold: float = 10):
    x = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2HSV)
    if x[:, :, 0].mean() < gray_threshold and x[:, :, 1].mean() < gray_threshold:
        return False
    return True


def quality_filter(img: Image.Image, quality_threshold: float = 45):
    quality = brisque.score(img)
    return quality < quality_threshold


def main(raw_args=None):
    args = parse_args(raw_args)
    args = process_args(args)

    filters = [functools.partial(quality_filter, quality_threshold=args.quality_threshold), gray_image_filter]

    queue = multiprocessing.Queue(maxsize=args.queue_size)
    process = [None] * args.num_workers

    for i in range(args.num_workers):
        p = multiprocessing.Process(target=decide_on_img, args=(queue, args.save_dir, args.trash_dir, filters))
        p.daemon = True
        p.start()
        process[i] = p

    print('Putting images in queue')
    for f in tqdm(io_utils.get_images_in_dir(args.images_dir)):
        queue.put(f)

    for _ in process:
        queue.put('END')

    for p in process:
        print(f'Joining on {p}')
        p.join()


def parse_args(raw_args=None):
    parser = argparse.ArgumentParser(description="Intrinsic Filter")

    parser.add_argument("--images_dir", type=Path, required=True,
                        help="The directory of input images")
    parser.add_argument("--save_dir", type=Path, required=True,
                        help="The directory to save good images")
    parser.add_argument("--trash_dir", type=Path, required=True,
                        help="The directory to save unused images")

    parser.add_argument("--quality_threshold", type=float, default=45)

    parser.add_argument("--queue_size", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=20)

    args = parser.parse_args(raw_args)
    return args


def process_args(args):
    args.save_dir.mkdir(exist_ok=True, parents=True)
    args.trash_dir.mkdir(exist_ok=True, parents=True)

    return args


if __name__ == '__main__':
    main()
