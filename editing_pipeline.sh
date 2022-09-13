# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


train_image_dir="XXX"
test_image_dir="XXX"

train_output_dir="XXX"
inversion_output_dir="XXX"
test_tuning_output_dir="XXX"

domain_generator_path="XXX"

# First, train a Personalized generator using MyStyle

python train.py \
    --images_dir "$train_image_dir" \
    --output_dir "$train_output_dir" \
    --generator_path "$domain_generator_path" \
    --encoder_checkpoint /path/to/pretrained/inversion/encoder \
    --device 1

# Now invert test images into P+ space.
# Consider inverting to P if further test-time tuning is performed.

python project.py \
    --images_dir "$test_image_dir" \
    --output_dir "$inversion_output_dir" \
    --anchor_dir "$train_output_dir/w" \
    --generator_path "$train_output_dir/mystyle_model.pt" \
    --beta 0.03 \
    --device 1
#    --is_wplus "False"

# (Optional & Recommended) To improve reconstruction, further tune the generator on test images

python train.py \
    --images_dir "$test_image_dir" \
    --output_dir "$test_tuning_output_dir" \
    --generator_path "$train_output_dir/mystyle_model.pt" \
    --anchor_dir "$inversion_output_dir/w" \
    --device 1

# Finally, edit

python edit.py \
    --alphas_dir "$inversion_output_dir/alpha" \
    --output_dir /path/to/save/results \
    --anchor_dir "$train_output_dir/w" \
    --generator_path "$test_tuning_output_dir/mystyle_model.pt" \
    --editing_direction_path /path/to/an/editing/direction \
    --edge_policy continue \
    --device 1



