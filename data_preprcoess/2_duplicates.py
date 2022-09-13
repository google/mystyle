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

import argparse
import sys
from pathlib import Path
import shutil

sys.path.append('..')

from utils import id_utils, io_utils

import matplotlib
import numpy as np
from tqdm import tqdm
from PIL import Image

matplotlib.use('Agg')


def find_id_dups(id_model: Path, images_dir: Path, threshold: float, images_dir_2: Path):
    person_identifier = id_utils.PersonIdentifier(id_model, None, None)  # Used only as a wrapper for feature extraction
    feats_dict = {}
    dups_dict = {}

    files_list = io_utils.get_images_in_dir(images_dir)
    if images_dir_2:
        files_list.extend(io_utils.get_images_in_dir(images_dir))

    for fp in tqdm(files_list):
        img = Image.open(fp)
        img = np.array(img)
        # feats.append(person_identifier.get_feature(img))
        feat = person_identifier.get_feature(img)
        for other_name, other_feat in feats_dict.items():
            try:
                sim = person_identifier.compute_similarity(feat, other_feat)
                if sim > threshold:
                    this_dup_list = dups_dict.setdefault(other_name, [])
                    this_dup_list.append((fp, sim))
                    dups_dict[other_name] = this_dup_list
            except Exception as e:
                print(f'Failed for {other_name}')

        feats_dict[fp] = feat

    return dups_dict


def clear_dups(id_model, images_dir: Path, trash_dir: Path, threshold: float, images_dir_2: Path):
    duplicates = find_id_dups(id_model, images_dir, threshold, images_dir_2)

    for k, v in duplicates.items():
        src_path = k

        # If this file doesn't exist - it was identified as dup of someone else.
        if not src_path.exists():
            continue
        if len(list(v)) == 0:
            continue

        trash_dup_path = trash_dir.joinpath(k.stem)
        trash_dup_path.mkdir(exist_ok=True)

        for dup in v:
            if type(dup) == tuple:
                dup = dup[0]

            dst_path = trash_dup_path.joinpath(dup.name)
            dup_path = dup
            try:
                print(f'Moving {dup} as dup of {k}')
                shutil.move(dup_path, dst_path)
            except Exception as e:
                pass


def parse_args(raw_args=None):
    parser = argparse.ArgumentParser(description="Extrinsic Filter")
    parser.add_argument('--images_dir', type=Path, required=True)
    parser.add_argument('--images_dir_2', type=Path)

    parser.add_argument('--id_model', type=Path, required=True)
    parser.add_argument('--trash_dir', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.92)
    args = parser.parse_args(raw_args)
    return args


def process_args(args):
    args.trash_dir.mkdir(exist_ok=True)
    return args


if __name__ == '__main__':
    args = parse_args()
    args = process_args(args)
    clear_dups(args.id_model, args.images_dir, args.trash_dir, args.threshold, args.images_dir_2)
