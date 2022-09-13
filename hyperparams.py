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

# Projection hyper-params
l2_weight = 1
sum_1_reg_weight = 10
deltas_reg_weight = 10
deltas_layers_to_reg = 12
projection_lr = 5e-3
projection_steps = 1000

# Tuning hyper-params
tune_lr = 3e-4
tune_steps = 1000

# Shared
visualize_freq = 0.1

