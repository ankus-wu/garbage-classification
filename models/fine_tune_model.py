# -*- coding: utf-8 -*-
# file: fine_tune_model.py
# author: JinTian
# time: 10/05/2017 9:54 AM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
from torchvision import models
from torch import nn
import torch.nn.functional as F
import torch

IMAGE_SIZE = 224

USE_GPU = torch.cuda.is_available()
MODEL_SAVE_FILE = 'ants_and_bees.pth'

def fine_tune_model():
    model_ft = models.resnet18(pretrained=True)
    num_features = model_ft.fc.in_features
    # fine tune we change original fc layer into classes num of our own
    model_ft.fc = nn.Linear(num_features, 6)

    if USE_GPU:
        model_ft = model_ft.cuda()
    return model_ft


class ClassifyNet(torch.nn.Module):
    def __init__(self):
        super(ClassifyNet, self).__init__()
        # restnet18
        self.restnet_ft = models.resnet18(pretrained=True)
        num_features = self.restnet_ft.fc.in_features
        self.restnet_ft.fc = nn.Linear(num_features, 6)

        # vgg
        #self.vgg_ft = models.vgg11_bn(pretrained=True)
        #num_ftrs = self.vgg_ft.classifier[6].in_features
        #self.vgg_ft.classifier[6] = nn.Linear(num_ftrs,6)

        self.dense_layer = torch.nn.Linear(6 + 6, 6)
 
    def forward(self, x):
        x1 = self.restnet_ft(x)
        #x2 = self.vgg_ft(x)
        # x3 = self.dense_layer(torch.cat([x1, x2], 1))
        # out = F.softmax(x1)
        return x1
