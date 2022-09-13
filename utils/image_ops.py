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

import cv2
import torch
import numpy as np
import torch.nn.functional as F


def to_np(x):
    return x.cpu().detach().double().numpy()


def convert_pixel_range(img, src, dst):
    if src != dst:
        src, dst = np.float32(src), np.float32(dst)
        img = np.clip(img, src[0], src[1])
        scale = (dst[1] - dst[0]) / (src[1] - src[0])
        bias = dst[0] - src[0] * scale
        img = img * scale + bias
    return img


def to_opencv_image(img, in_range=None):
    if in_range is None:
        in_range = [-1, 1]
    img = convert_pixel_range(img.squeeze(), in_range, [0, 255])
    img = np.uint8(np.round(
        to_np(torch.permute(img, (1, 2, 0)))
    ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def to_torch_image(img, out_range=None):
    if out_range is None:
        out_range = [-1, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    img = convert_pixel_range(img, [0, 255], out_range)
    return img


def blend(src_img: torch.Tensor, dst_img: torch.Tensor, mask: torch.Tensor):
    src_img = to_opencv_image(src_img)
    dst_img = to_opencv_image(dst_img)

    mask = 1 - mask  # Invert mask, take hidden region from src instead of hiding it
    mask_np = np.uint8(255 * mask.squeeze().cpu())

    kernel = np.ones((20, 20), np.uint8)
    mask_np = cv2.dilate(mask_np, kernel, iterations=1)

    br = cv2.boundingRect(mask_np)  # bounding rect (x,y,width,height)
    center = (br[0] + br[2] // 2, br[1] + br[3] // 2)
    out = cv2.seamlessClone(dst_img, src_img, mask_np, center, cv2.NORMAL_CLONE)

    out = to_torch_image(out)
    return out


class Degradation:
    @staticmethod
    def hole(img, mask):
        deg_img = img * mask.to(img.device)
        return deg_img

    @staticmethod
    def downsample(img, factor):
        deg_img = F.interpolate(img, scale_factor=factor, mode='area', recompute_scale_factor=False)
        return deg_img
