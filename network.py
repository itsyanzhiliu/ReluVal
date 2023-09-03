import torch.nn as nn
import torch

class SimpleNet(nn.Module):
    def __init__(self, weights1, bias1, weights2, bias2):
        super(SimpleNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )
        self._init_weights(weights1, bias1, weights2, bias2)

    def _init_weights(self, weights1, bias1, weights2, bias2):
        # 1st linear layer
        self.layers[0].weight.data = weights1.clone().detach()
        self.layers[0].bias.data = bias1.clone().detach()
        
        # 2nd linear layer
        self.layers[-1].weight.data = weights2.clone().detach()
        self.layers[-1].bias.data = bias2.clone().detach()

    @torch.no_grad()
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    # Define the weights and biases for the two linear layers using PyTorch tensors
    weights1 = torch.tensor([[2., 3.], [1., 1.]], dtype=torch.float32)
    bias1 = torch.tensor([0., 0.], dtype=torch.float32)
    weights2 = torch.tensor([[1., -1.]], dtype=torch.float32)
    bias2 = torch.tensor([0.], dtype=torch.float32)

    # Create the network with the specified weights and biases
    net = SimpleNet(weights1, bias1, weights2, bias2)

    # Define input tensor
    x = torch.tensor([[2., 1.]])

    # Forward pass
    y = net(x)
    print("Output:", y)
