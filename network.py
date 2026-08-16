import torch.nn as nn
import torchvision
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class ResNet_cross_entropy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        backbone = getattr(torchvision.models, f"resnet{args.model_depth}")(weights=None)
        #
        fc_inputs = backbone.fc.in_features
        #
        self.model_extractor = nn.Sequential(*list(backbone.children())[:-1])
        #
        self.Flatten = nn.Flatten(start_dim=1)
        #
        self.model_linear =  nn.Sequential(nn.Linear(fc_inputs, 1))
        #
        self.model_classifier = nn.Sequential(nn.Linear(fc_inputs, args.age_groups))

        
    # g is the same shape of y
    def forward(self, x):
        #"output of model dim is 2G"
        z = self.model_extractor(x)
        #
        z = self.Flatten(z)
        #
        y_hat = self.model_linear(z)
        #
        logits = self.model_classifier(z)
        # the ouput dim of the embed is : 512
        #
        return y_hat, z, logits