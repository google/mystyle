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
from argparse import ArgumentParser
from pathlib import Path

import inversion_encoder
from utils import io_utils, data_utils

import torch


def parse_args(raw_args=None):
    parser = ArgumentParser('Infer anchors with W Encoder')

    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--verbose', type=io_utils.str2bool, default="True")

    parser.add_argument('--images_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--encoder_checkpoint', type=Path, required=True)

    args, _ = parser.parse_known_args(raw_args)
    return args


def process_args(args):
    if not args.images_dir.exists():
        raise ValueError(f'Image directory {args.images_dir} does not exist')

    args.output_dir.joinpath('latents').mkdir(exist_ok=True, parents=True)

    if args.verbose:
        args.output_dir.joinpath('inversions').mkdir(exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    return args


def invert(images_dir, encoder_checkpoint, output_dir, verbose=False):
    dataset = data_utils.PersonalizedDataset(images_dir, latent_dir=None)
    encoder = inversion_encoder.InversionEncoder(encoder_checkpoint, torch.device(f'cuda'))

    with torch.no_grad():
        dataset = encoder.invert(dataset)

    dataset.save_latents(output_dir)

    if verbose:
        dataset.save_recons(output_dir)

    return dataset


def main(raw_args=None):
    args = parse_args(raw_args)
    args = process_args(args)
    invert(args.images_dir, args.encoder_checkpoint, args.output_dir, args.verbose)


if __name__ == '__main__':
    main()
