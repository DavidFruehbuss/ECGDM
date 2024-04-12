import math
import numpy as np

def sinusoidal_node_features_for_graph(node_indices, num_features=10):
    """
    Compute sinusoidal node features for positional encoding of graph nodes.

    Args:
        node_indices (list): List of node indices in the graph.
        num_features (int): Number of features in the encoding.

    Returns:
        numpy.ndarray: Sinusoidal node features for the given node indices.
    """
    # Initialize an array to store the sinusoidal node features for all nodes
    sinusoidal_features = np.zeros((len(node_indices), num_features))
    
    # Compute the frequencies for the sinusoidal functions
    frequencies = np.array([1 / (10000 ** (2 * (i // 2) / num_features)) for i in range(num_features)])
    
    # Compute the sinusoidal encoding for each node index in the list
    for j, node_index in enumerate(node_indices):
        for i in range(num_features):
            if i % 2 == 0:
                sinusoidal_features[j, i] = math.sin(node_index * frequencies[i])
            else:
                sinusoidal_features[j, i] = math.cos(node_index * frequencies[i])
    
    return sinusoidal_features

# Example usage:
# graph_nodes = [1, 2, 3, 4, 5]  # List of node indices in the graph
# num_features = 16  # Number of features in the encoding
# sinusoidal_encoding = sinusoidal_node_features_for_graph(graph_nodes, num_features)
# print("Sinusoidal node features for the graph nodes:", sinusoidal_encoding)
