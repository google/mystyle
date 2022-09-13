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
from enum import Enum
from argparse import ArgumentParser

sys.path.append('third_party/stylegan2_ada_pytorch')

from utils import io_utils
from utils.io_utils import load_latents, load_net, load_single_latent
from utils.latent_space_ops import to_wplus, project_latents_to_spanned_subspace, ConvexProjector, to_column_vector, \
    to_row_vector

import numpy as np
import torch
import torch.nn.functional as F


class EdgePolicy(Enum):
    STOP = 0
    CONTINUE = 1
    PROJECT = 2

    @classmethod
    def argtype(cls, s: str) -> Enum:
        return cls[s.upper()]

    def __str__(self):
        return self.name


def get_edited_w(convex_projector, alpha, gamma, anchors, edit_mag, num_edits, edge_policy, beta=0.02):
    """
    latent codes are in W+ row format - i.e [1,num_ws,w_dim]
    """
    thetas = torch.FloatTensor(np.union1d(
        np.linspace(-edit_mag, 0, num_edits // 2),
        np.linspace(0, edit_mag, num_edits // 2),
    )).reshape((-1, 1)).cuda()

    edited_ws = []
    used_thetas = []

    for theta in thetas:
        edit_alpha = alpha + theta * gamma
        edit_w = edit_alpha @ anchors
        min_dilation = torch.abs(torch.min(edit_alpha))

        if min_dilation > beta:
            if edge_policy == EdgePolicy.STOP:
                continue  # Skip it
            elif edge_policy == EdgePolicy.PROJECT:
                edit_w, _ = convex_projector.solve(edit_w.squeeze(0), beta)
                edit_w = torch.from_numpy(edit_w).unsqueeze(0).cuda()

        used_thetas.append(theta)
        edited_ws.append(edit_w)

    edited_ws = torch.cat(edited_ws, dim=0) if len(edited_ws) > 0 else None
    return used_thetas, edited_ws


def edit(alphas_dir, edit_direction, generator, anchors, output_path, edit_mag, num_edits, edge_policy, beta):
    edit_direction = to_column_vector(edit_direction)
    gamma, proj_standard = project_latents_to_spanned_subspace(edit_direction, anchors)
    convex_projector = ConvexProjector(anchors, num_layers=generator.num_ws)

    print(f'Cosine similarity of projected direction to original one is'
          f' {F.cosine_similarity(proj_standard, edit_direction, dim=0).item():.3f}')

    batch_size = 16

    for alpha_file in alphas_dir.iterdir():
        all_images = []

        name = alpha_file.stem
        curr_out = output_path.joinpath(name)
        curr_out.mkdir(exist_ok=True)

        alpha = load_single_latent(alpha_file)
        alpha = to_wplus(alpha, num_layers=generator.num_ws)
        gamma = to_wplus(gamma, num_layers=generator.num_ws)

        thetas, w_codes = get_edited_w(convex_projector, alpha, gamma, anchors, edit_mag,
                                       num_edits, edge_policy, beta)

        for chunk in w_codes.split(batch_size):
            all_images.append(generator(chunk, noise_mode='const', force_fp32=True).cpu())

        all_images = torch.cat(all_images, dim=0)
        for i, img in enumerate(all_images):
            io_utils.save_images(img, curr_out.joinpath(f'idx_{i:03d}_mag_{thetas[i].item()}'.replace('.', 'd')))


def parse_args(raw_args=None):
    parser = ArgumentParser('Editing arguments')

    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--verbose', type=io_utils.str2bool, default="True")

    parser.add_argument('--editing_direction_path', type=io_utils.existing_path, required=True)
    parser.add_argument('--output_dir', type=io_utils.create_path, required=True)
    parser.add_argument('--anchor_dir', type=io_utils.existing_path, required=True)
    parser.add_argument('--generator_path', type=io_utils.existing_path, required=True)

    parser.add_argument('--alphas_dir', type=io_utils.existing_path, required=True)

    parser.add_argument('--edit_mag', type=float, default=2)
    parser.add_argument('--num_edits', type=int, default=11)

    parser.add_argument('--edge_policy', type=EdgePolicy.argtype, choices=EdgePolicy, default=EdgePolicy.CONTINUE,
                        help='What to do when editing leaves P_beta')

    parser.add_argument('--beta', type=float, default=0.03)

    args = parser.parse_args(raw_args)
    return args


def process_args(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    args.device = torch.device(f'cuda')
    return args


def main(raw_args=None):
    args = parse_args(raw_args)
    args = process_args(args)

    anchors = load_latents(args.anchor_dir, to_w=True)

    generator = load_net(args.generator_path)
    edit_direction = load_single_latent(args.editing_direction_path)

    with torch.no_grad():
        generator.eval()
        edit(args.alphas_dir, edit_direction, generator, anchors, args.output_dir,
             args.edit_mag, args.num_edits, args.edge_policy, args.beta)


if __name__ == '__main__':
    main()
