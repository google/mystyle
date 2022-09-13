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

import tkinter as tk
from tkinter import messagebox

import torch
from third_party.arcface.arcface import Backbone

import cv2
import numpy as np


class PersonIdentifier:
    def __init__(self, model_path, num_examples, threshold):
        super().__init__()
        if model_path is None:
            raise ValueError('PersonIdentifier class cannot be init without FR network')

        self.features_dict = {}
        self.num_examples = num_examples
        self.threshold = threshold
        self.first_manual_run = True

        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se').cuda()
        self.facenet.load_state_dict(torch.load(model_path))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112)).cuda()
        self.facenet.eval()
        for module in [self.facenet, self.face_pool]:
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def transform_image(img):
        if not torch.is_tensor(img):
            img = torch.from_numpy(img)
        x = torch.unsqueeze(img, 0)
        x = x / 127.5 - 1
        x = torch.clamp(x, -1, 1)
        if x.shape[-1] == 3:
            x = torch.permute(x, [0, 3, 1, 2])
        return x

    def get_feature(self, img):
        # img = img[:, :, 35:223, 32:220]  # Crop interesting region
        img = self.transform_image(img).cuda()
        x = self.face_pool(img)
        x_feats = self.facenet(x)
        return x_feats

    @staticmethod
    def compute_similarity(feat1, feat2):
        d = torch.cosine_similarity(feat1, feat2)
        return d

    def auto_is_same_person(self, img):
        # bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow('image', bgr_img)
        # key = cv2.waitKey()
        # cv2.destroyWindow('image')
        curr_feature = self.get_feature(img)
        similarities = []
        for img_name, feature in self.features_dict.items():
            similarity = self.compute_similarity(curr_feature, feature)
            if similarity > self.threshold:
                return True
            similarities.append(similarity.item())

        if np.count_nonzero(np.array(similarities) > self.threshold * 0.8) > len(self.features_dict) / 3:
            # If no score is above the threshold but at least third are close to it, also approve
            return True

        return False

    @staticmethod
    def message_box():
        root = tk.Tk().withdraw()
        messagebox.showinfo('Annotation Instructions',
                            f'To create a personalized dataset, we first have to know which person!\n'
                            'We therefore need a few exemplar images. '
                            'When an image pops up, press "y" to approve it or any other key to decline.')

    def manual_is_same_person(self, img):
        if self.first_manual_run:
            self.message_box()
            self.first_manual_run = False

        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        window_name = f'Press "y" to approve or any other key to decline'
        cv2.imshow(window_name, bgr_img)
        key = cv2.waitKey()
        cv2.destroyWindow(window_name)

        return key == ord('y')

    def verify_id(self, img, img_name):
        if len(self.features_dict.keys()) < self.num_examples:
            same_person = self.manual_is_same_person(img)
            add_to_examples = same_person and True
        else:
            same_person = self.auto_is_same_person(img)
            add_to_examples = False

        if add_to_examples:
            self.features_dict[img_name] = self.get_feature(img)

        return same_person
