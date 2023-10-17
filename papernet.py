import torch
import torch.nn as nn


class PaperNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2)
        )
        self._init_weights()

    def random(self, size):
        while True:
            w = (torch.FloatTensor(*size).uniform_(-1, 1) * 10).int().float() / 10
            if (w != 0).all():
                return w

    def _init_weights(self):
        self.layers[0].weight.data = self.random(size=(2, 2))
        self.layers[0].bias.data = self.random(size=(2,))

        self.layers[2].weight.data = self.random(size=(2, 2))
        self.layers[2].bias.data = self.random(size=(2,))

        self.layers[4].weight.data = self.random(size=(2, 2))
        self.layers[4].bias.data = self.random(size=(2,))

    @torch.no_grad()
    def forward(self, x):
        return self.layers(x)

    def print_w_b(self):
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                print(layer)
                print('\t[+] w:', layer.weight.data.flatten())
                print('\t[+] b:', layer.bias.data.flatten())
                print()


