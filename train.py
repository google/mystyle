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

# required for pickle to magically find torch_utils for loading official FFHQ checkpoint
sys.path.append('third_party/stylegan2_ada_pytorch')

import infer_anchors
from reconstruct.tune_net import Tuner
from utils import io_utils
from utils.data_utils import PersonalizedDataset

import torch


def get_data(args):
    if args.anchor_dir is not None:
        dataset = PersonalizedDataset(args.images_dir, args.anchor_dir)
    else:
        print('Path to anchors was not given, inferring them on the fly...')
        dataset = infer_anchors.invert(args.images_dir, args.encoder_checkpoint,
                                       args.output_dir, args.verbose)
    return dataset


def generate_from_anchors(generator, dataset, output_path):
    generator.eval()
    for sample in dataset:
        img = generator(sample.w_code.cuda(), noise_mode='const', force_fp32=True)
        io_utils.save_images(img, output_path.joinpath(sample.name))


def parse_args(raw_args=None):
    parser = ArgumentParser('Train arguments script')

    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--verbose', type=io_utils.str2bool, default="True")

    parser.add_argument('--images_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--generator_path', type=Path, required=True)

    parser.add_argument('--anchor_dir', type=Path)
    parser.add_argument('--encoder_checkpoint', type=Path)

    args = parser.parse_args(raw_args)
    return args


def process_args(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    args.device = torch.device(f'cuda')

    if not args.images_dir.exists():
        raise ValueError(f'Image directory {args.images_dir} does not exist')
    if not args.generator_path.exists():
        raise ValueError(f'Domain checkpoint {args.generator_path} does not exist')

    args.output_dir.mkdir(exist_ok=True, parents=True)
    args.debug_output_dir = None
    if args.verbose:
        args.debug_output_dir = args.output_dir.joinpath('debug')
        args.debug_output_dir.mkdir(exist_ok=True, parents=True)

    return args


def main(raw_args=None):
    args = parse_args(raw_args)
    args = process_args(args)

    dataset = get_data(args)

    generator = io_utils.load_net(args.generator_path)

    if args.verbose:
        generate_from_anchors(generator, dataset, args.debug_output_dir.joinpath('before'))

    network_tuner = Tuner(args.device, generator, args.debug_output_dir.joinpath('during') if args.verbose else None)
    network_tuner.reconstruct(dataset)

    torch.save(network_tuner.generator, args.output_dir.joinpath(f'mystyle_model.pt'))

    if args.verbose:
        generate_from_anchors(generator, dataset, args.debug_output_dir.joinpath('after'))


if __name__ == '__main__':
    main()
