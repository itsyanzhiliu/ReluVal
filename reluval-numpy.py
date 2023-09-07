import numpy as np

def _pos(x):
    return np.clip(x, 0, np.inf)

def _neg(x):
    return np.clip(x, -np.inf, 0)

def _evaluate(eq_lower, eq_upper, input_lower, input_upper):
    input_lower = input_lower.reshape(-1, 1)
    input_upper = input_upper.reshape(-1, 1)
    o_l_l = _pos(eq_upper[:-1]) * input_lower + _neg(eq_lower[:-1]) * input_upper
    o_u_u = _pos(eq_upper[:-1]) * input_upper + _neg(eq_lower[:-1]) * input_lower
    o_l_l = np.sum(o_l_l, axis=0) + eq_lower[-1]
    o_u_u = np.sum(o_u_u, axis=0) + eq_upper[-1]
    return o_l_l, o_u_u

def relu_transform(eq_lower, eq_upper, input_lower, input_upper, input_bounds=None):
    output_eq_lower = eq_lower.copy()
    output_eq_upper = eq_upper.copy()

    if input_bounds is not None:
        o_l_l, o_u_u = input_bounds
    else:
        o_l_l, o_u_u = _evaluate(eq_lower, eq_upper, input_lower, input_upper)

    grad_mask = np.zeros(o_l_l.shape[0])

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

def linear_transform(weights, bias, eq_lower, eq_upper):
    pos_weight, neg_weight = _pos(weights), _neg(weights)
    out_eq_upper = np.dot(eq_upper, pos_weight.T) + np.dot(eq_lower, neg_weight.T)
    out_eq_lower = np.dot(eq_lower, pos_weight.T) + np.dot(eq_upper, neg_weight.T)
    if bias is not None:
        out_eq_lower[-1] += bias
        out_eq_upper[-1] += bias
    return out_eq_lower, out_eq_upper

def forward(net, lower, upper, return_grad_mask=False):
    input_features = lower.size

    eq_lower = np.concatenate((np.eye(input_features), np.zeros((1, input_features))), axis=0)
    eq_upper = eq_lower.copy()

    o_l_l = lower.copy()
    o_u_u = upper.copy()
    grad_mask = {}

    for layer_id, layer in enumerate(net['layers']):
        if isinstance(layer, tuple):  # Linear layer with weights and bias
            weights, bias = layer
            eq_lower, eq_upper = linear_transform(weights, bias, eq_lower, eq_upper)
        elif isinstance(layer, str) and layer == 'relu':
            (eq_lower, eq_upper), grad_mask_l = relu_transform(eq_lower, eq_upper, lower, upper, input_bounds=(o_l_l, o_u_u))
            grad_mask[layer_id] = grad_mask_l
        else:
            raise NotImplementedError
        o_l_l, o_u_u = _evaluate(eq_lower, eq_upper, lower, upper)

    if return_grad_mask:
        return (o_l_l, o_u_u), grad_mask
    return o_l_l, o_u_u

# Example usage:
if __name__ == "__main__":
    # Define the weights and biases for the two linear layers
    weights1 = np.array([[2., 3.], [1., 1.]])
    bias1 = np.array([1., 2.])  # Bias for the first linear layer
    weights2 = np.array([[1., -1.]])
    bias2 = np.array([0.])  # Bias for the second linear layer

    # Create the network as a dictionary of layers
    net = {'layers': [(weights1, bias1), 'relu', (weights2, bias2)]}

    # Define input bounds
    input_lower = np.array([4., 1.])
    input_upper = np.array([6., 5.])

    # Forward pass
    output_lower, output_upper = forward(net, input_lower, input_upper)
    print("Output Lower Bound:", output_lower)
    print("Output Upper Bound:", output_upper)
