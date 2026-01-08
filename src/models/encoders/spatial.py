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
        
        # New: Robust feature to node mapping
        self.must_project = (self.input_dim % self.num_nodes != 0) or (self.type in ['st_gcn', 'gat'])
        
        if self.must_project:
            self.input_projector = nn.Linear(self.input_dim, self.num_nodes * self.hidden_dim)
            self.node_feat_dim = self.hidden_dim
        else:
            self.node_feat_dim = self.input_dim // self.num_nodes

        if self.type == 'spatialGNN':
            self.layers = nn.ModuleList()
            # If we projected, in_features is node_feat_dim
            in_dim = self.node_feat_dim if self.must_project else self.input_dim
            self.layers.append(GraphConvolution(in_dim, self.hidden_dim))
            for _ in range(self.num_layers - 1):
                self.layers.append(GraphConvolution(self.hidden_dim, self.hidden_dim))
            self.activation = nn.ReLU()
            
        elif self.type == 'st_gcn':
            self.gcn_layer = nn.Linear(self.node_feat_dim, self.hidden_dim) 
            self.tcn = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4)
            self.dropout_layer = nn.Dropout(self.dropout)
            
        elif self.type == 'gat':
            self.gat_instance = GraphAttentionLayer(
                self.node_feat_dim, 
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
        
        # 1. Uniformly handle feature to node mapping
        if self.must_project:
            # Handle case where F_total changed since init (e.g. different features enabled)
            if F_total != self.input_dim:
                # Re-initialize projector if needed (fallback)
                self.input_projector = nn.Linear(F_total, self.num_nodes * self.hidden_dim).to(x.device)
                self.input_dim = F_total
            
            x_reshaped = self.input_projector(x.view(B*T, F_total))
            x_reshaped = x_reshaped.view(B*T, self.num_nodes, self.hidden_dim)
            node_feat_dim = self.hidden_dim
        else:
            node_feat_dim = F_total // self.num_nodes
            x_reshaped = x.view(B*T, self.num_nodes, node_feat_dim)

        # 2. Process with specific graph architecture
        if self.type == 'spatialGNN':
            curr_x = x_reshaped
            for layer in self.layers:
                curr_x = self.activation(layer(curr_x, adj))
            return curr_x.view(B, T, -1)

        elif self.type == 'st_gcn':
            if adj is None:
                adj = torch.eye(self.num_nodes).to(x.device).unsqueeze(0).expand(B*T, -1, -1)
            elif adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(B*T, -1, -1)
            
            support = self.gcn_layer(x_reshaped) 
            x_out = torch.bmm(adj, support) 
            x_out = F.relu(x_out)
            
            x_out = x_out.view(B, T, self.num_nodes, -1)
            x_out = x_out.mean(dim=2) # (B, T, H)
            x_out = x_out.permute(0, 2, 1) # (B, H, T)
            
            x_out = self.tcn(x_out)
            x_out = self.dropout_layer(x_out)
            
            return x_out.transpose(1, 2) # (B, T, H)

        elif self.type == 'gat':
            if adj is None:
                adj = torch.ones(B*T, self.num_nodes, self.num_nodes).to(x.device)
            elif adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(B*T, -1, -1)
                
            x_out = self.gat_instance(x_reshaped, adj) 
            x_out = self.dropout_layer(x_out)
            x_out = x_out.mean(dim=1) 
            
            return x_out.view(B, T, -1)

        return x