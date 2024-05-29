import time
import torch.nn as nn

class BasicModel(nn.Module):
    def __init__(self, configs, device):
        super(BasicModel, self).__init__()
        self.args = configs
        self.device = device

    def forward(self, *args, **kwargs):
        pass