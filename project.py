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
from argparse import ArgumentParser
from pathlib import Path

sys.path.append('third_party/stylegan2_ada_pytorch')

from reconstruct.alpha_projector import AlphaProjector
from utils import io_utils, image_ops
from utils.data_utils import PersonalizedDataset
from utils.io_utils import load_latents
from utils.image_ops import Degradation

import torch
import numpy as np

torch.manual_seed(2)
np.random.seed(2)


def project(args):
    dataset = PersonalizedDataset(args.images_dir,
                                  mask_dir=args.mask_dir)

    anchors = load_latents(args.anchor_dir, to_w=True)
    generator = io_utils.load_net(args.generator_path)

    if args.mask_dir is not None:
        deg_func = Degradation.hole
    elif args.sr_factor is not None:
        deg_func = lambda x, **kwargs: Degradation.downsample(x, args.sr_factor)
    else:
        deg_func = lambda x, **kwargs: x

    alpha_projector = AlphaProjector(args.device, generator,
                                     args.debug_output_dir, args.is_wplus,
                                     anchors, deg_func,
                                     beta=args.beta)

    for sample in dataset:
        sample.img = deg_func(sample.img, mask=sample.mask)
        sample.save_input(args.output_dir)

        sample = alpha_projector.reconstruct(sample)

        if sample.mask is not None:
            blended_recon = image_ops.blend(sample.img, sample.recon_img, sample.mask)
            sample.set(recon_img=blended_recon)

        sample.save_latent(args.output_dir)
        sample.save_recon(args.output_dir)

        # TODO(3): SR blending, face segmenetation model?


def parse_args(raw_args=None):
    parser = ArgumentParser('Projection arguments')

    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--verbose', type=io_utils.str2bool, default="True")

    parser.add_argument('--images_dir', type=io_utils.existing_path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--generator_path', type=io_utils.existing_path, required=True)
    parser.add_argument('--anchor_dir', type=io_utils.existing_path, required=True)

    parser.add_argument('--beta', type=io_utils.float_or_none, default=0.03,
                        help='Controls the maximal allowed dilation of the personalized space.'
                             'Pass None to not restrict dilation.')
    parser.add_argument('--is_wplus', type=io_utils.str2bool, default="True")

    parser.add_argument('--mask_dir', type=io_utils.existing_path)
    parser.add_argument('--sr_factor', type=float)

    # TODO(2): support W/W+ projection

    args = parser.parse_args(raw_args)
    return args


def process_args(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    args.device = torch.device(f'cuda')

    args.output_dir.mkdir(exist_ok=True, parents=True)
    args.debug_output_dir = None
    if args.verbose:
        args.debug_output_dir = args.output_dir.joinpath('debug')
        args.debug_output_dir.mkdir(exist_ok=True, parents=True)

    if args.sr_factor is not None and args.sr_factor > 1:
        args.sr_factor = 1 / args.sr_factor

    return args


def main(raw_args=None):
    args = parse_args(raw_args)
    args = process_args(args)
    project(args)


if __name__ == '__main__':
    main()
