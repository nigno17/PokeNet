#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Nino Cauli
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import models


class PokeNet(torch.nn.Module):
    def __init__(self, D_latent = 512, D_actions = 12):
        
        super(PokeNet, self).__init__()
        self.D_features = 256 * 6 * 6
        
        alexNet = models.alexnet(pretrained=True)
        self.features = alexNet.features
        self.latent1 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
        )
        self.latent2 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(D_latent * 2, D_latent),
            nn.ReLU(inplace=True),
            nn.Linear(D_latent, D_actions),
            nn.Softmax(),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(D_actions + D_latent, D_latent),
            nn.ReLU(inplace=True),
            nn.Linear(D_latent, D_latent),
            nn.ReLU(inplace=True),
        )

    def forward(self, img1, img2, input_actions):

        features1 = self.features(img1)
        features1 = features1.view(features1.size(0), self.D_features)
        features2 = self.features(img2)
        features2 = features2.view(features2.size(0), self.D_features)
        
        lat1 = self.latent1(features1)
        lat2 = self.latent1(features2)
        
        join_features = torch.cat((lat1, lat2), 1)
        output_actions = self.inverse_model(join_features)
        
        join_act_feat = torch.cat((lat1, input_actions), 1)
        output_latent = self.forward_model(join_act_feat)
        
        return output_actions, output_latent, lat2
    
class PokeNet1C(torch.nn.Module):
    def __init__(self, D_latent = 512, D_actions = 12):
        
        super(PokeNet1C, self).__init__()
        self.D_features = 256 * 3 * 3
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.latent1 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
        )
        self.latent2 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(D_latent * 2, D_latent),
            nn.ReLU(inplace=True),
            nn.Linear(D_latent, D_actions),
            nn.Softmax(),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(D_actions + D_latent, D_latent),
            nn.ReLU(inplace=True),
            nn.Linear(D_latent, D_latent),
            nn.ReLU(inplace=True),
        )

    def forward(self, img1, img2, input_actions):

        features1 = self.features(img1)
        features1 = features1.view(features1.size(0), self.D_features)
        features2 = self.features(img2)
        features2 = features2.view(features2.size(0), self.D_features)
        
        lat1 = self.latent1(features1)
        lat2 = self.latent1(features2)
        
        join_features = torch.cat((lat1, lat2), 1)
        output_actions = self.inverse_model(join_features)
        
        join_act_feat = torch.cat((lat1, input_actions), 1)
        output_latent = self.forward_model(join_act_feat)
        
        return output_actions, output_latent, lat2
