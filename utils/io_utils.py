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

import pickle
from pathlib import Path

from third_party.stylegan2_ada_pytorch import dnnlib
from utils import latent_space_ops

import numpy as np
import torch
import torchvision
from PIL import Image

IMAGE_SUFFIX = ['.jpg', '.png', '.svg', '.webp', '.jpeg']


def str2bool(s):
    if s.lower() in ['false', 'f', 'no']:
        return False
    if s.lower() in ['true', 't', 'yes']:
        return True

    raise ValueError(f"Don't know how to convert {s} to bool")


def float_or_none(s):
    if s.lower() == 'none':
        return None
    else:
        return float(s)


def existing_path(s):
    p = Path(s)
    if not p.exists():
        raise ValueError(f'Input path {s} does not exist but is expected to')

    return p


def create_path(s):
    p = Path(s)
    p.mkdir(exist_ok=True, parents=True)
    return p


def load_single_latent(latent_path: Path):
    suffix = latent_path.suffix
    if suffix == '.pt':
        x = torch.load(latent_path)
    elif suffix == '.npy':
        x = torch.FloatTensor(np.load(latent_path))
    elif suffix == '.pickle':
        with open(latent_path, 'rb') as fp:
            x = pickle.load(fp)
    else:
        raise NotImplemented()

    return x.to('cuda')


def load_latents(latents_dir: Path, to_w=False):
    latents = []
    for f in latents_dir.iterdir():
        if not (f.is_file() or f.suffix == '.pt'):
            continue

        latents.append(torch.load(f))

    latents = torch.stack(latents, dim=0)

    if to_w:
        latents = latent_space_ops.wplus_to_w(latents)

    return latents


def load_net(file_path: Path):
    try:
        with dnnlib.util.open_url(str(file_path)) as f:
            G = pickle.load(f)['G_ema'].synthesis
    except Exception as e:
        G = torch.load(file_path)

    return G.cuda()


def save_images(frames: torch.FloatTensor, output_path: Path):
    parent_dir = output_path.parent
    parent_dir.mkdir(exist_ok=True, parents=True)

    torchvision.utils.save_image(
        frames,
        output_path.with_suffix('.jpg'),
        nrow=frames.shape[0],
        normalize=True,
        range=(-1, 1)
    )


def save_latents(latent: torch.FloatTensor, output_path: Path):
    if latent is None:
        return

    parent_dir = output_path.parent
    parent_dir.mkdir(exist_ok=True, parents=True)
    torch.save(latent, output_path)


def load_mask(mask_path: Path):
    mask_img = Image.open(mask_path).convert('L')
    mask = np.array(mask_img)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0

    mask = torch.FloatTensor(mask)
    mask = torch.unsqueeze(mask, dim=0) / 255
    return mask


def get_images_in_dir(input_dir: Path):
    global IMAGE_SUFFIX
    image_fps = [fp for fp in input_dir.iterdir() if fp.suffix in IMAGE_SUFFIX]
    return image_fps
