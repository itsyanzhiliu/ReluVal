import torch.nn as nn
import torch

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layers = nn.Sequential(*[
            nn.Linear(2, 2),
            # nn.ReLU(),
            nn.Linear(2, 1),
        ])
        self._init_weights()

    def _init_weights(self):
        # 1st linear layer
        self.layers[0].weight.data = torch.tensor([
            [2., 3.],
            [1., 1.],
        ])
        self.layers[0].bias.data.fill_(0)
        
        # 2nd linear layer (change index if needed)
        self.layers[-1].weight.data = torch.tensor([
            [1., -1.],
        ])
        self.layers[-1].bias.data.fill_(0)

    @torch.no_grad()
    def forward(self, x):
        for layer in self.layers:
            # print('input:', x)
            x = layer(x)
            # print('output:', x)
        return x


if __name__ == "__main__":
    net = SimpleNet()
    x = torch.tensor([[2., 1.]])
    y = net(x)
    print(y)