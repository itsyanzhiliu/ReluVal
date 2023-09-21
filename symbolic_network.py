import torch.nn as nn
import torch
import sympy as sp


class EquationNet(nn.Module):
    def __init__(self):
        super(EquationNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )

        self.x = sp.symbols('x')
        self.y = sp.symbols('y')

    def forward(self, coefficients):
        # Ensure that coefficients have shape (1, 2)
        coefficients = coefficients.view(1, -1)

        # Define the symbolic equation
        a, b = coefficients[0, 0], coefficients[0, 1]
        equation = a * self.x + b * self.y

        # Return the symbolic equation as a string
        return str(equation)


if __name__ == "__main__":
    coefficients = torch.tensor([[2.0, 1.0]], dtype=torch.float32)

    net = EquationNet()

    symbolic_equation = net(coefficients)

    print("Symbolic Equation:", symbolic_equation)
