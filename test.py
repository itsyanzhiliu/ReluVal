from network import SimpleNet
from reluval import forward
import torch

if __name__ == "__main__":
    net = SimpleNet()
    lower = torch.tensor([[4., 1.]])
    upper = torch.tensor([[6., 5.]])
    print(forward(net, lower, upper))
