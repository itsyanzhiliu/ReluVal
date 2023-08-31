import numpy as np

def symbolic_interval_analysis(network, input_interval):
    eq_upper = [input_interval[0][1], input_interval[1][1]]  # Upper bounds of the input interval
    eq_lower = [input_interval[0][0], input_interval[1][0]]  # Lower bounds of the input interval

    num_layers = len(network)
    R = np.zeros((num_layers, max(layer.shape[1] for layer in network)), dtype=object)  # Cache for mask matrix

    for layer_idx, layer in enumerate(network):
        weights = layer
        eq_upper = np.dot(eq_upper, weights)  # Update upper bounds
        eq_lower = np.dot(eq_lower, weights)  # Update lower bounds

        if layer_idx != num_layers - 1:
            for i in range(len(layer)):
                if np.less_equal(eq_upper[i], 0).all():
                    R[layer_idx][i] = np.array([0, 0])  # d(relu(x)) = [0, 0]
                    eq_upper[i] = 0
                elif np.greater_equal(eq_lower[i], 0).all():
                    R[layer_idx][i] = np.array([1, 1])  # d(relu(x)) = [1, 1]
                else:
                    R[layer_idx][i] = np.array([0, 1])  # d(relu(x)) = [0, 1]
                    eq_upper[i] = 0

    output_interval = {"lower": eq_lower, "upper": eq_upper}
    return R, output_interval


if __name__ == "__main__":
    # Define the neural network architecture
    network = [
        np.array([[2.0, 1.0], [3.0, 1.0]]),  # weights for layer 1
        np.array([[1.0], [-1.0]])  # weights for layer 2
    ]

    # Define the input interval as [lower_bounds, upper_bounds]
    input_interval = [np.array([4, 6]), np.array([1, 5])]

    # Calculate the symbolic interval analysis
    R_matrix, output_interval = symbolic_interval_analysis(network, input_interval)


    # Print the results
    print("R matrix:", R_matrix)
    print("Output interval:", output_interval)
