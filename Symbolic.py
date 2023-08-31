import numpy as np


def symbolic_interval_analysis(network, input_interval):
    eq = np.array([input_interval[1], input_interval[0]])  # Initialize eq = (equp, eqlow)
    num_layers = len(network)
    R = np.zeros((num_layers, max(layer.shape[1] for layer in network)), dtype=object)  # Cache for mask matrix

    for layer_idx, layer in enumerate(network):
        weights = layer
        eq = np.dot(eq, weights)

        if layer_idx != num_layers - 1:
            for i in range(len(layer)):
                if np.less_equal(eq[i], 0).all():
                    R[layer_idx][i] = np.array([0, 0])  # d(relu(x)) = [0, 0]
                    eq[i] = 0
                elif np.greater_equal(eq[i], 0).all():
                    R[layer_idx][i] = np.array([1, 1])  # d(relu(x)) = [1, 1]
                else:
                    R[layer_idx][i] = np.array([0, 1])  # d(relu(x)) = [0, 1]
                    eq[i] = 0

    output_interval = {"lower": eq[0], "upper": eq[1]}
    return R, output_interval


if __name__ == '__main__':
    # Example usage
    network = [
        np.array([[2.0, 1.0], [3.0, 1.0]]),  # weights for layer 1
        np.array([[1.0], [-1.0]])  # weights for layer 2
    ]
    input_interval = np.array([[4, 6], [1, 5]])

    R_matrix, output_interval = symbolic_interval_analysis(network, input_interval)
    print("R matrix:", R_matrix)
    print("Output interval:", output_interval)
