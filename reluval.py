import torch.nn as nn
import torch

def _pos(x):
    return torch.clamp(x, 0, torch.inf)


def _neg(x):
    return torch.clamp(x, -torch.inf, 0)


def _evaluate(eq_lower, eq_upper,
              input_lower, input_upper):
    input_lower = input_lower.view(-1, 1)
    input_upper = input_upper.view(-1, 1)
    o_l_l = _pos(eq_upper[:-1]) * input_lower + _neg(eq_lower[:-1]) * input_upper
    o_u_u = _pos(eq_upper[:-1]) * input_upper + _neg(eq_lower[:-1]) * input_lower
    o_l_l = o_l_l.sum(0) + eq_lower[-1]
    o_u_u = o_u_u.sum(0) + eq_upper[-1]
    return o_l_l, o_u_u


def relu_transform(eq_lower, eq_upper,
                   input_lower, input_upper,
                   input_bounds=None):
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


@torch.no_grad()
def forward(net, lower, upper, return_grad_mask=False):
    input_features = lower.numel()

    # initialize lower and upper equation
    eq_lower = torch.concat([torch.eye(input_features), torch.zeros(1, input_features)], dim=0)
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
    return o_l_l, o_u_u

