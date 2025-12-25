import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj=None):
        support = self.linear(x)
        if adj is not None:
            output = torch.matmul(adj, support)
        else:
            output = support
        return output

class SpatialGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(SpatialGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GraphConvolution(hidden_dim, hidden_dim))
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        for layer in self.layers:
            x = self.activation(layer(x, adj))
        return x
