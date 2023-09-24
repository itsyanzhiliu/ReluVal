import torch
import torch.nn as nn
import sympy as sp


class EquationNet(nn.Module):
    def __init__(self):
        super(EquationNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )
        self._init_weights()
        self.x = sp.symbols('x')
        self.y = sp.symbols('y')

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

    def forward(self, coefficients):
        # Ensure that coefficients have shape (1, 2)
        coefficients = coefficients.view(1, -1)

        # Define the symbolic equation
        a, b = coefficients[0, 0], coefficients[0, 1]
        equation = a * self.x + b * self.y

        # Return the symbolic equation as a string
        return str(equation)


def _pos(x):
    return torch.clamp(x, 0, torch.inf)


def _neg(x):
    return torch.clamp(x, -torch.inf, 0)


def _evaluate(eq_lower, eq_upper, input_lower, input_upper):
    input_lower = input_lower.view(-1, 1)
    input_upper = input_upper.view(-1, 1)
    o_l_l = _pos(eq_upper[:-1]) * input_lower + _neg(eq_lower[:-1]) * input_upper
    o_u_u = _pos(eq_upper[:-1]) * input_upper + _neg(eq_lower[:-1]) * input_lower
    o_l_l = o_l_l[:, 0]
    o_u_u = o_u_u[:, 0]
    return o_l_l, o_u_u



def relu_transform(eq_lower, eq_upper, input_lower, input_upper, input_bounds=None):
    # evaluate output ranges
    output_eq_lower = eq_lower.clone()
    output_eq_upper = eq_upper.clone()

    if input_bounds is not None:
        o_l_l, o_u_u = input_bounds
    else:
        o_l_l, o_u_u = _evaluate(eq_lower, eq_upper, input_lower, input_upper)

    grad_mask = torch.zeros(o_l_l.size(0))

    for i, (ll, uu) in enumerate(zip(o_l_l, o_u_u)):
        if uu <= 0:
            grad_mask[i] = 0
            output_eq_lower[:, i] = 0
            output_eq_upper[:, i] = 0
        elif ll >= 0:
            grad_mask[i] = 2
        else:
            grad_mask[i] = 1
            output_eq_lower[:, i] = 0
            output_eq_upper[:-1, i] = 0
            output_eq_upper[-1, i] = uu
    return (output_eq_lower, output_eq_upper), grad_mask


def linear_transform(layer, eq_lower, eq_upper):
    pos_weight, neg_weight = _pos(layer.weight), _neg(layer.weight)
    out_eq_upper = eq_upper @ pos_weight.T + eq_lower @ neg_weight.T
    out_eq_lower = eq_lower @ pos_weight.T + eq_upper @ neg_weight.T
    if layer.bias is not None:
        out_eq_lower[-1] += layer.bias
        out_eq_upper[-1] += layer.bias
    return out_eq_lower, out_eq_upper

def tensor_to_equation(coefficients):
    """
    Converts a PyTorch tensor into a symbolic equation string with variable names 'x' and 'y'.

    Args:
        coefficients (torch.Tensor): The tensor containing coefficients.

    Returns:
        str: The symbolic equation string.
    """
    variable_names = ['x', 'y']
    equation_str = " + ".join([f"{coeff:.0f}*{var}" for coeff, var in zip(coefficients, variable_names)])
    return equation_str

@torch.no_grad()
def forward(net, lower, upper, return_grad_mask=False):
    input_features = lower.numel()

    # initialize lower and upper equation
    eq_lower = torch.cat([torch.eye(input_features), torch.zeros(1, input_features)], dim=0)
    eq_upper = eq_lower.clone()

    o_l_l = lower.clone()
    o_u_u = upper.clone()
    grad_mask = {}

    for layer_id, layer in enumerate(net.layers):
        if isinstance(layer, nn.Linear):
            eq_lower, eq_upper = linear_transform(layer, eq_lower, eq_upper)
        elif isinstance(layer, nn.ReLU):
            (eq_lower, eq_upper), grad_mask_l = relu_transform(eq_lower, eq_upper,
                                                               lower, upper,
                                                               input_bounds=(o_l_l, o_u_u))
            grad_mask[layer_id] = grad_mask_l
        else:
            raise NotImplementedError
        o_l_l, o_u_u = _evaluate(eq_lower, eq_upper, lower, upper)

    if return_grad_mask:
        return (o_l_l, o_u_u), grad_mask
    return tensor_to_equation(o_l_l), tensor_to_equation(o_u_u)


if __name__ == "__main__":
    # coefficients = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    # Create the network
    net = EquationNet()

    # coefficients
    lower = torch.tensor([[1., 1.]])
    upper = torch.tensor([[1., 1.]])

    print(forward(net, lower, upper))
