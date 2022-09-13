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

import abc
import math

import hyperparams

import lpips
import torch.nn.functional as F


class BaseReconstructor(abc.ABC):
    def __init__(self, device, generator, num_steps, debug_out_path, l2_weight):
        self.device = device
        self.generator = generator
        self.l2_weight = l2_weight
        self.debug_out_path = debug_out_path

        self.visualize_freq = hyperparams.visualize_freq
        self.num_steps = num_steps
        self.optimizer = None

        self.lpips = lpips.LPIPS(net='alex').to(self.device).eval()
        self.set_optimization()

    @abc.abstractmethod
    def set_optimization(self):
        raise NotImplementedError()

    def reconstruction_loss(self, synth, target):
        """
        Compute reconstruction loss from input, output pair.
        Children prepare Inputs and Outputs according to degradation
        """

        dist = self.lpips(synth, target)
        l2_dist = (synth - target).square().mean()

        loss = dist + self.l2_weight * l2_dist

        return loss

    def reconstruct(self, dataset):
        """
        Iteratively optimize something for image reconstruction, not abstract
        """
        pass

    @staticmethod
    def get_perceptual_features(img, net):
        target = img

        if target.shape[2] > 256:
            target_small = F.interpolate(target, size=(256, 256), mode='area')
        else:
            target_small = target

        return net(target_small, resize_images=False, return_lpips=True)

    def need_visualize(self, step):
        return self.debug_out_path and step % math.ceil(self.visualize_freq * self.num_steps) == 0
