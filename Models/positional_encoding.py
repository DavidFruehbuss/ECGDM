import torch
import torch.nn as nn
import math

class GraphPositionalEncoding(nn.Module):
    def __init__(self, num_atoms, max_graph_nodes=5000):
        super(GraphPositionalEncoding, self).__init__()
        self.d_model = num_atoms
        self.max_graph_nodes = max_graph_nodes

    def forward(self, h, chain_pos, idx, sizes, max_size=9):
        """
        Adds positional encoding to the input tensor for graphs.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_nodes_per_graph, d_model).
            idx (torch.Tensor): Index Tensor indicating to which graph each node belongs.
                                 Shape: (batch_size * num_nodes_per_graph).
            sizes (torch.Tensor): Size of each graph (molecue and protein threated as seperate graphs)

        Returns:
            torch.Tensor: Output tensor with positional encoding added.
        """
        device = h.device

        # Generate positions tensor
        chain_pos = chain_pos

        # Scale positions within each graph based on the mask
        positions = positions.masked_fill(mask.unsqueeze(-1) == 0, 0)

        # Compute sinusoidal positional encodings
        angles = torch.arange(0, self.d_model, 2).float() * math.pi / self.d_model
        angles = torch.pow(10000, -angles)
        encodings = torch.einsum("bi,d->bid", positions.float(), angles)

        # Apply sine to even indices and cosine to odd indices
        encodings = torch.stack([torch.sin(encodings), torch.cos(encodings)], dim=-1)
        encodings = encodings.view(batch_size, max_nodes_per_graph, -1)

        # Add positional encodings to the input tensor
        return x + encodings

# Example usage
d_model = 512  # Dimensionality of the model
max_graph_nodes = 100  # Maximum number of nodes per graph
pos_encoder = GraphPositionalEncoding(d_model, max_graph_nodes)

# Example input tensor
batch_size = 2
nodes_per_graph = [5, 4]  # Number of nodes for each graph in the batch
input_tensors = [torch.randn(nodes, d_model) for nodes in nodes_per_graph]
max_nodes = max(nodes_per_graph)

# Pad input tensors to have the same number of nodes per graph
input_tensors = [torch.cat([t, torch.zeros(max_nodes - t.size(0), d_model)]) for t in input_tensors]

# Create mask tensor indicating the presence of nodes in each graph
mask = torch.zeros(batch_size, max_nodes)
for i, nodes in enumerate(nodes_per_graph):
    mask[i, :nodes] = 1

# Adding positional encoding to the input tensors
output_tensors = pos_encoder(torch.stack(input_tensors), mask)

print(output_tensors.size())  # Output shape: (batch_size, max_nodes_per_graph, d_model)
