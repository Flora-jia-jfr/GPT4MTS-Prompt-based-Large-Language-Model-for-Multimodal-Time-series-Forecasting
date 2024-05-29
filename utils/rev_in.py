import torch
import torch.nn as nn

class RevIn(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: Number of input features
        :param eps: Stability add-on value
        :param affine: If True, RevIN has learnable affine parameters
        """
        super(RevIn, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last.to(x.device)
        else:
            x = x - self.mean.to(x.device)
        x = x / self.stdev.to(x.device)
        if self.affine:
            x = x * self.affine_weight.to(x.device)
            x = x + self.affine_bias.to(x.device)
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias.to(x.device)
            x = x / (self.affine_weight.to(x.device) + self.eps * self.eps)
        x = x * self.stdev.to(x.device)
        if self.subtract_last:
            x = x + self.last.to(x.device)
        else:
            x = x + self.mean.to(x.device)
        return x
