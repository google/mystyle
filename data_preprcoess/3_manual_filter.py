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

import shutil
import argparse
from pathlib import Path

from utils import io_utils

import cv2


def main(in_path, trash_path):
    for f in io_utils.get_images_in_dir(in_path):
        img = cv2.imread(str(f))
        win_name = 'Accept with Y, reject with any other key'
        cv2.imshow(win_name, img)
        cv2.moveWindow(win_name, 0, 0)
        key = cv2.waitKey()
        cv2.destroyWindow(win_name)

        if key == ord('y'):
            continue

        dst_path = trash_path.joinpath(f.name)
        shutil.move(f, dst_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Manual Filter')

    parser.add_argument('--input_path', required=True, type=Path)
    parser.add_argument('--trash_path', required=True, type=Path)

    args = parser.parse_args()
    args.trash_path.mkdir(exist_ok=True)
    main(args.input_path, args.trash_path)
