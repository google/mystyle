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

import numpy as np
import cvxpy as cp
import torch


def to_np(x):
    return x.cpu().detach().double().numpy()


def wplus_to_w(latents: torch.FloatTensor):
    """
    latents is expected to have shape N,1,18,512
    """
    _, counts = torch.unique(latents, dim=2, return_counts=True)
    if len(counts) != 1:
        raise ValueError('input latent is not a W code, conversion from W+ is undefined')

    return latents[:, 0, 0, :]


def to_wplus(latents: torch.FloatTensor, num_layers=18):
    if latents.dim() == 1:
        latents.unsqueeze_(dim=0)
    if latents.dim() == 2:
        return to_row_vector(latents).unsqueeze(dim=1).repeat(1, num_layers, 1)

    return latents


def project_latents_to_spanned_subspace(latent: torch.FloatTensor, basis: torch.FloatTensor):
    proj_barycentric = (basis @ basis.t()).inverse() @ basis @ latent
    proj_standard = basis.t() @ proj_barycentric

    return proj_barycentric, proj_standard


def to_column_vector(latent: torch.FloatTensor):
    if latent.dim() == 1:
        return latent.unsqueeze(1)
    if latent.dim() == 2:
        if latent.shape[1] != 1 and latent.shape[0] == 1:
            return latent.t()
        else:
            return latent
    else:
        raise ValueError('Conversion to column vector is undefined')


def to_row_vector(latent: torch.FloatTensor):
    return to_column_vector(latent).t()


def sample_from_P0(anchors: torch.FloatTensor, num_points_to_sample: int, num_anchors_for_sample: int = 3):
    num_anchors = anchors.shape[0]

    scalars = torch.zeros(num_points_to_sample, num_anchors).to(anchors.device)
    tmp_scalars = torch.rand(num_points_to_sample, num_anchors_for_sample).to(anchors.device)
    tmp_scalars = (tmp_scalars / torch.sum(tmp_scalars, axis=-1, keepdims=True))

    indices = np.zeros((num_points_to_sample, num_anchors_for_sample), dtype=int)

    for i in range(num_points_to_sample):
        indices[i, :] = np.random.choice(anchors.shape[0], size=num_anchors_for_sample, replace=False)
        scalars[i, indices[i]] = tmp_scalars[i]

    if anchors.dim() == 4:
        points = torch.einsum('pN, Nolk ->  polk', scalars, anchors)
    elif anchors.dim() == 2:
        points = scalars @ anchors
    else:
        raise ValueError('Anchors are in unknown format')

    return points


class ConvexProjector:
    def __init__(self, anchors, num_layers):
        self.anchors = to_np(anchors)  # (num_anchors, w_dim)
        self.num_anchors = self.anchors.shape[0]
        self.w_dim = self.anchors.shape[1]
        self.alpha = cp.Variable((num_layers, self.num_anchors))

        self.w = cp.Parameter((num_layers, self.w_dim))
        self.min_neg = cp.Parameter(1, )

        obj = cp.Minimize(0.5 * cp.sum_squares(self.alpha @ self.anchors - self.w))
        cons = [cp.sum(self.alpha, axis=1) == 1., self.min_neg <= self.alpha]
        # cons = [cp.sum(self.alpha) == 1.]
        self.prob = cp.Problem(obj, cons)

    def solve(self, w, beta):
        self.w.value = to_np(w)
        self.min_neg.value = -1 * np.ones(1, ) * beta
        self.prob.solve()

        print(f'Projection to P_beta took {self.prob.solver_stats.solve_time:.2f} seconds')

        solution_w = self.alpha.value @ self.anchors
        return solution_w, self.prob.value  # solution, error
