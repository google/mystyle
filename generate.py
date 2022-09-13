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

import os
import sys
from pathlib import Path
from argparse import ArgumentParser

from utils import latent_space_ops, io_utils

from torchvision.utils import save_image
import torch
import numpy as np

torch.manual_seed(2)
np.random.seed(2)

# required for pickle to magically find torch_utils for loading official FFHQ checkpoint
sys.path.append('third_party/stylegan2_ada_pytorch')


def synthesize(anchors_path, generator_path, output_path, num_points_to_sample, num_anchor_points):
    anchors = io_utils.load_latents(anchors_path)
    latents = latent_space_ops.sample_from_P0(anchors, num_points_to_sample, num_anchor_points).to('cuda')
    generator = io_utils.load_net(generator_path).to('cuda')

    batch_size = 4
    i = 0
    while i < latents.shape[0]:
        lats = latents[i: i + batch_size]
        imgs = generator(lats.squeeze(1), noise_mode='const', force_fp32=True)

        for j in range(min(batch_size, num_points_to_sample - i)):
            save_image(imgs[j], output_path.joinpath('images', f'{i + j}.jpg'), nrow=1, normalize=True, range=(-1, 1))
            io_utils.save_latents(lats[j], output_path.joinpath('latents', f'{i + j}.pt'))

        del imgs
        i += batch_size


def parse_args(raw_args):
    parser = ArgumentParser()
    parser.add_argument('--anchors_path', required=True, type=Path)
    parser.add_argument('--generator_path', required=True, type=Path)
    parser.add_argument('--output_path', required=True, type=Path)

    parser.add_argument('--device', default='0')

    parser.add_argument('--num_points_to_sample', type=int, default=30)
    parser.add_argument('--num_anchors_for_sampling', type=int, default=3)

    args = parser.parse_args(raw_args)
    return args


def process_args(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    args.output_path.mkdir(exist_ok=True, parents=True)
    args.output_path.joinpath('latents').mkdir(exist_ok=True, parents=True)
    args.output_path.joinpath('images').mkdir(exist_ok=True, parents=True)

    return args


def main(raw_args=None):
    # TODO(2): support W synthesis
    args = parse_args(raw_args)
    args = process_args(args)

    synthesize(args.anchors_path, args.generator_path, args.output_path,
               args.num_points_to_sample, args.num_anchors_for_sampling)


if __name__ == '__main__':
    with torch.no_grad():
        main()
