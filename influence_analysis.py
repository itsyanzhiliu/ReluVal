import numpy as np

def choose_most_influential_feature(network, input_interval, gradient_interval):
    num_features = input_interval.shape[0]
    largest = -float('inf')
    split_feature = None

    for i in range(num_features):
        # Calculate the range of each input interval
        r = input_interval[i][1] - input_interval[i][0]

        # Calculate the influence from each input to output
        e = np.multiply(gradient_interval[i][0], r)

        if np.all(e > largest):
            largest = e
            split_feature = i

    return split_feature

if __name__ == '__main__':
    # Example usage
    network = [
        np.array([[2.0, 1.0], [3.0, 1.0]]),  # weights for layer 1
        np.array([[1.0], [-1.0]])  # weights for layer 2
    ]
    input_interval = np.array([[4, 6], [1, 5]])
    gradient_interval = np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])  # Example gradient interval

    most_influential_feature = choose_most_influential_feature(network, input_interval, gradient_interval)
    print("Most influential feature to split:", most_influential_feature)
