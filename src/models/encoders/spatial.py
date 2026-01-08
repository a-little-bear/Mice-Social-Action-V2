import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj=None):
        support = self.linear(x)
        if adj is not None:
            if adj.dtype != support.dtype:
                adj = adj.to(dtype=support.dtype)
            
            if adj.dim() == support.dim(): 
                output = torch.matmul(adj, support)
            elif adj.dim() == 2 and support.dim() == 3:
                output = torch.matmul(adj, support)
            else:
                output = torch.matmul(adj, support)
        else:
            output = support
        return output

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # Scaling factor for attention stability
        self.scale = out_features ** -0.5

    def forward(self, h, adj=None):
        Wh = torch.matmul(h, self.W) 
        B, N, _ = Wh.size()
        
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) 
        
        # Apply scaling BEFORE softmax to prevent gradient explosion/NaNs
        e = e * self.scale

        if adj is not None:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(B, -1, -1)
            # Use a large negative number for masked attention
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
        else:
            attention = e
            
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh) 

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        B, N, E = Wh.size()
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix.view(B, N, N, 2 * E)

class SpatialEncoder(nn.Module):
    """
    Unified Spatial Encoder supporting spatialGNN, st_gcn, and gat.
    """
    def __init__(self, config):
        super().__init__()
        self.type = config.get('type', 'spatialGNN')
        self.input_dim = config.get('input_dim', 32)
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.num_nodes = config.get('num_nodes', 7) 
        
        # Mapping input to hidden space and nodes
        self.input_projector = nn.Linear(self.input_dim, self.num_nodes * self.hidden_dim)
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if self.type == 'spatialGNN':
            for i in range(self.num_layers):
                self.layers.append(GraphConvolution(self.hidden_dim, self.hidden_dim))
                self.norms.append(nn.LayerNorm(self.hidden_dim))
            
        elif self.type == 'st_gcn':
            for i in range(self.num_layers):
                self.layers.append(nn.ModuleDict({
                    'gcn': nn.Linear(self.hidden_dim, self.hidden_dim),
                    'tcn': nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4)
                }))
                self.norms.append(nn.LayerNorm(self.hidden_dim))
            
        elif self.type == 'gat':
            for i in range(self.num_layers):
                self.layers.append(GraphAttentionLayer(
                    self.hidden_dim, 
                    self.hidden_dim, 
                    dropout=self.dropout, 
                    alpha=0.2, 
                    concat=False
                ))
                self.norms.append(nn.LayerNorm(self.hidden_dim))
        else:
            raise ValueError(f"Unknown spatial encoder type: {self.type}")

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, adj=None):
        """
        x: (Batch, Time, Features)
        adj: (Nodes, Nodes) or None
        """
        B, T, F_total = x.shape
        
        # Initial projection to node feature space
        # (B, T, F) -> (B*T, N, H)
        x = self.input_projector(x.view(B*T, F_total))
        x = x.view(B*T, self.num_nodes, self.hidden_dim)

        if self.type == 'spatialGNN':
            for gcn, norm in zip(self.layers, self.norms):
                res = x
                x = gcn(x, adj)
                x = F.relu(x)
                x = norm(x)
                x = x + res # Residual connection
            
        elif self.type == 'st_gcn':
            if adj is None:
                adj = torch.eye(self.num_nodes).to(x.device).unsqueeze(0).expand(B*T, -1, -1)
            elif adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(B*T, -1, -1)

            for layer, norm in zip(self.layers, self.norms):
                res = x
                # Spatial part
                support = layer['gcn'](x)
                x = torch.bmm(adj, support)
                x = F.relu(x)
                
                # Temporal projection across window
                x_reshaped = x.view(B, T, self.num_nodes, self.hidden_dim).permute(0, 2, 3, 1) # (B, N, H, T)
                B_idx, N_idx, H_idx, T_idx = x_reshaped.shape
                x_reshaped = x_reshaped.reshape(B_idx * N_idx, H_idx, T_idx) # (B*N, H, T)
                x_reshaped = layer['tcn'](x_reshaped)
                
                x = x_reshaped.view(B, self.num_nodes, self.hidden_dim, T).permute(0, 3, 1, 2) # (B, T, N, H)
                x = x.reshape(B * T, self.num_nodes, self.hidden_dim)
                
                x = norm(x)
                x = x + res
                x = self.dropout_layer(x)

        elif self.type == 'gat':
            if adj is None:
                # Full connectivity if no adj provided
                adj = torch.ones(B*T, self.num_nodes, self.num_nodes).to(x.device)
            elif adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(B*T, -1, -1)
                
            for gat, norm in zip(self.layers, self.norms):
                res = x
                x = gat(x, adj)
                x = norm(x)
                x = x + res # Residual
                x = self.dropout_layer(x)

        # Global average pooling over nodes to get (B*T, H)
        x_out = x.mean(dim=1) 
        return x_out.view(B, T, -1) 
