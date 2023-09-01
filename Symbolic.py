import numpy as np
import sympy as sp

def concrete_symbolic_interval_analysis(network, input_interval):
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

    output_interval = [eq_lower, eq_upper]
    return R, output_interval


def symbolic_interval_analysis(network, input_symbols):
    x, y = input_symbols  # Symbolic variables for x and y

    eq_upper = [x, y]  # Upper bounds of the input interval
    eq_lower = [x, y]  # Lower bounds of the input interval

    num_layers = len(network)
    R = np.empty((num_layers, max(layer.shape[1] for layer in network)), dtype=object)  # Cache for mask matrix

    for layer_idx, layer in enumerate(network):
        weights = layer
        eq_upper = [sp.expand(sum(eq_upper[i] * weights[i][j] for i in range(len(layer)))) for j in range(weights.shape[1])]
        eq_lower = [sp.expand(sum(eq_lower[i] * weights[i][j] for i in range(len(layer)))) for j in range(weights.shape[1])]

        if layer_idx != num_layers - 1:
            for i in range(len(layer)):
                if eq_upper[i].is_number and eq_upper[i] <= 0:
                    R[layer_idx][i] = "d(relu(x)) = [0, 0]"
                    eq_upper[i] = 0
                elif eq_lower[i].is_number and eq_lower[i] >= 0:
                    R[layer_idx][i] = "d(relu(x)) = [1, 1]"
                else:
                    R[layer_idx][i] = "d(relu(x)) = [0, 1]"
                    eq_upper[i] = 0

    output_equation = {"lower": eq_lower, "upper": eq_upper}
    return R, output_equation

if __name__ == "__main__":
    # Define the neural network architecture
    network = [
        np.array([[2.0, 1.0], [3.0, 1.0]]),  # weights for layer 1
        np.array([[1.0], [-1.0]])  # weights for layer 2
    ]

    # Define the input interval as [lower_bounds, upper_bounds]
    input_concrete_interval = [[4, 6],[1, 5]]
    input_interval = [sp.symbols('x'), sp.symbols('y')]

    # Calculate the concrete symbolic interval analysis
    R_concrete_matrix, output_concrete_interval = concrete_symbolic_interval_analysis(network, input_concrete_interval)
    # Calculate symbolic interval equations
    R_matrix, output_interval = symbolic_interval_analysis(network, input_interval)

    input_symbols = [sp.symbols('x'), sp.symbols('y')]



    print("R matrix:", R_matrix)
    print("Interval equations:", output_interval)

    print('-----------------')

    print("Concrete R matrix:", R_concrete_matrix)
    print("Output concrete interval:", output_concrete_interval)


