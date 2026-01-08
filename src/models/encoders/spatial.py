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
            # Check for dimensions mismatch if adj is (N,N) but x is (B*T, F)
            # This class is generic, we handle reshaping in SpatialEncoder
            if adj.dtype != support.dtype:
                adj = adj.to(dtype=support.dtype)
            
            # If shapes are compatible for matmul
            if adj.dim() == support.dim(): 
                output = torch.matmul(adj, support)
            elif adj.dim() == 2 and support.dim() == 3:
                # adj: (N,N), support: (B, N, F)
                output = torch.matmul(adj, support)
            else:
                # Fallback to simple multiplication if broadcasting works, or ignore adj
                # For flattened spatialGNN input (B*T, F), we can't easily validly use (N,N) adj 
                # unless we unflatten internally.
                # Assuming SpatialEncoder handles passing compatible inputs.
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
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj=None):
        Wh = torch.matmul(h, self.W) 
        B, N, _ = Wh.size()
        
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) 

        zero_vec = -9e15*torch.ones_like(e)
        
        if adj is not None:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(B, -1, -1)
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
        self.num_nodes = config.get('num_nodes', 1) 
        # Note: num_nodes must be set correctly in config for st_gcn/gat
        
        if self.type == 'spatialGNN':
            self.layers = nn.ModuleList()
            self.layers.append(GraphConvolution(self.input_dim, self.hidden_dim))
            for _ in range(self.num_layers - 1):
                self.layers.append(GraphConvolution(self.hidden_dim, self.hidden_dim))
            self.activation = nn.ReLU()
            
        elif self.type == 'st_gcn':
            self.node_dim = self.input_dim // self.num_nodes
            self.gcn = nn.Linear(self.node_dim, self.hidden_dim) 
            self.tcn = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4)
            self.dropout_layer = nn.Dropout(self.dropout)
            
        elif self.type == 'gat':
            self.node_dim = self.input_dim // self.num_nodes
            self.gat_layer = GraphAttentionLayer(
                self.node_dim, 
                self.hidden_dim, 
                dropout=self.dropout, 
                alpha=0.2, 
                concat=True
            )
            self.dropout_layer = nn.Dropout(self.dropout)
            
        else:
            raise ValueError(f"Unknown spatial encoder type: {self.type}")

    def forward(self, x, adj=None):
        """
        x: (Batch, Time, Features)
        adj: (Nodes, Nodes) or None
        """
        B, T, F_total = x.shape
        
        if self.type == 'spatialGNN':
            # spatialGNN treats the whole frame features as one or passes adj if dimensions match
            # To handle ablation "use_adj=True" vs "False" for spatialGNN,
            # we need to decide what 'spatialGNN' means with adj.
            # Original code was MLP style. 
            # If use_adj is True, we probably want GCN behavior on nodes.
            # But graph convolution usually requires (Nodes, Features).
            # If x is flattened, we assume it's just MLP if adj is None.
            # If adj is present, we must reshape.
            
            if adj is not None:
                # Assuming spatialGNN with adj means GCN on nodes
                # Reshape to (B*T, N, C)
                if self.num_nodes > 1 and F_total % self.num_nodes == 0:
                    node_dim = F_total // self.num_nodes
                    x_reshaped = x.view(B*T, self.num_nodes, node_dim)
                    
                    # Apply GCN per layer
                    # Note: Our GraphConvolution above is generic linear.
                    # We need explicit GCN logic: D^-0.5 A D^-0.5 X W
                    # The GraphConvolution class in original spatial.py was basically Linear(x) + matmul(adj, support).
                    # This works for (B, N, F) input and (N, N) adj.
                    
                    curr_x = x_reshaped
                    for layer in self.layers:
                        # layer(curr_x, adj) will do matmul(adj, linear(curr_x))
                        curr_x = self.activation(layer(curr_x, adj))
                    
                    # Flatten back
                    return curr_x.view(B, T, -1)
                else:
                    # Fallback to MLP if dimensions don't make sense for graph
                    x_flat = x.view(B*T, F_total)
                    for layer in self.layers:
                        x_flat = self.activation(layer(x_flat, None))
                    return x_flat.view(B, T, -1)
            else:
                # MLP behavior (original spatialGNN)
                x_flat = x.view(B*T, F_total)
                for layer in self.layers:
                    x_flat = self.activation(layer(x_flat, None))
                return x_flat.view(B, T, -1)

        elif self.type == 'st_gcn':
            N = self.num_nodes
            if F_total % N != 0:
                 raise ValueError(f"Input features {F_total} not divisible by num_nodes {N}")
            C = F_total // N
            
            x_reshaped = x.view(B*T, N, C)
            
            if adj is None:
                # Fallback adj if not provided but st_gcn needs graph structure?
                # Or just identity
                adj = torch.eye(N).to(x.device).unsqueeze(0).expand(B*T, -1, -1)
            elif adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(B*T, -1, -1)
            
            support = self.gcn(x_reshaped) 
            x_out = torch.bmm(adj, support) 
            x_out = F.relu(x_out)
            
            # (B*T, N, H) -> (B, T, N, H)
            x_out = x_out.view(B, T, N, -1)
            
            # ST-GCN usually aggregates over nodes for temporal modeling?
            # TopologyEncoder implementation was: mean(dim=2) -> (B, -1, T) -> TCN
            
            # Permute for proper TCN input (B, Channels, Time)
            # Pooling nodes:
            x_out = x_out.mean(dim=2) # (B, T, H)
            x_out = x_out.permute(0, 2, 1) # (B, H, T)
            
            x_out = self.tcn(x_out)
            x_out = self.dropout_layer(x_out)
            
            return x_out.transpose(1, 2) # (B, T, H)

        elif self.type == 'gat':
            N = self.num_nodes
            if F_total % N != 0:
                 raise ValueError(f"Input features {F_total} not divisible by num_nodes {N}")
            C = F_total // N
            
            x_reshaped = x.view(B*T, N, C)
            
            if adj is None:
                adj = torch.ones(B*T, N, N).to(x.device) # Full attention if no adj
            elif adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(B*T, -1, -1)
                
            x_out = self.gat_layer(x_reshaped, adj) 
            x_out = self.dropout_layer(x_out)
            
            # Pooling
            x_out = x_out.mean(dim=1) # (B*T, H)
            
            return x_out.view(B, T, -1)

        return x