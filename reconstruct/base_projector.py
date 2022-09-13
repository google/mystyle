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

from reconstruct.base_reconstructor import BaseReconstructor
from utils import io_utils

from tqdm import tqdm


class BaseProjector(BaseReconstructor):
    def __init__(self, device, generator, num_steps, debug_out_path, l2_weight, degradation_func=None):
        super().__init__(device, generator, num_steps, debug_out_path, l2_weight)
        self.degradation_func = degradation_func

    @abc.abstractmethod
    def get_latent(self):
        """
        get final W/W+ latent
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def regularization_loss(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_optimized(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, sample):
        raise NotImplementedError()

    def reconstruction_loss(self, synth, target, **kwargs):
        degraded_synth = self.degradation_func(synth, **kwargs)
        # degraded_target = self.degradation_func(target, **kwargs)
        return super(BaseProjector, self).reconstruction_loss(degraded_synth, target)

    def reconstruct(self, sample):
        self.set_optimization()

        # TODO(3): implement early stopping
        for step in tqdm(range(self.num_steps)):
            to_visualize = self.need_visualize(step)

            latent_opt = self.get_latent()
            target = sample.img.cuda()

            synth = self.generator(latent_opt, noise_mode='const', force_fp32=True)

            recon_loss = self.reconstruction_loss(synth, target, mask=sample.mask)
            regularization_loss = self.regularization_loss()
            loss = recon_loss + regularization_loss

            if to_visualize:
                io_utils.save_images(synth.detach().cpu(),
                                     self.debug_out_path.joinpath(f'{sample.name}_step_{step}'))
                print(f'step {step + 1:>4d}/{self.num_steps}: loss {float(loss.item()):<5.2f}')

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        self.save(sample)
        return sample
