import numpy as np

def backward_propagation_gradient_interval(network, R_matrix):
    num_layers = len(network)
    gup = [np.array([R_matrix[num_layers - 1][i][0]]) for i in range(len(network[num_layers - 1]))]
    glow = [np.array([R_matrix[num_layers - 1][i][1]]) for i in range(len(network[num_layers - 1]))]

    for layer_idx in range(num_layers - 2, -1, -1):
        layer = network[layer_idx]
        weights = layer
        g_layer = [np.zeros_like(layer[i], dtype=object) for i in range(len(layer))]

        for node in range(len(layer)):
            g_layer[node] = np.array([gup[node], glow[node]])  # Create an interval containing gup and glow
            g_layer[node] = np.multiply(R_matrix[layer_idx][node], g_layer[node])  # Apply interval Hadamard product
            g_layer[node] = np.multiply(weights[node], g_layer[node])  # Apply interval matrix multiplication

            gup[node] = g_layer[node][0]  # Update gup for the next layer
            glow[node] = g_layer[node][1]  # Update glow for the next layer

    return gup, glow


if __name__ == '__main__':
    network = [
        np.array([[2.0, 1.0], [3.0, 1.0]]),  # weights for layer 1
        np.array([[1.0], [-1.0]])  # weights for layer 2
    ]

    # Create a placeholder R_matrix with the same structure as the network
    R_matrix = [
        [np.zeros_like(layer, dtype=object) for layer in network],  # R matrix for layer 1
        [np.zeros_like(layer, dtype=object) for layer in network]  # R matrix for layer 2
    ]

    updated_gup, updated_glow = backward_propagation_gradient_interval(network, R_matrix)
    print("Updated gup:", updated_gup)
    print("Updated glow:", updated_glow)

