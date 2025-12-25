import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, h, adj):
        # h: [batch_size, N, in_features]
        # adj: [batch_size, N, N]
        
        Wh = torch.matmul(h, self.W) # [B, N, out_features]
        B, N, _ = Wh.size()
        
        # Prepare for attention mechanism
        # a_input: [B, N, N, 2*out_features]
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) # [B, N, N]

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh) # [B, N, out_features]

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

class TopologyEncoder(nn.Module):
    """
    Module B: Stream 1 - Topology Stream
    Supports: ST-GCN, GAT
    """
    def __init__(self, config):
        super().__init__()
        self.type = config['type']
        input_dim = config.get('input_dim', 32)
        hidden_dim = config['hidden_dim']
        dropout = config.get('dropout', 0.1)
        
        if self.type == 'st_gcn':
            # Simplified ST-GCN: GCN + Temporal Conv
            self.gcn = nn.Linear(input_dim, hidden_dim) # Placeholder for GCN weight
            self.tcn = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4)
            self.dropout = nn.Dropout(dropout)
            
        elif self.type == 'gat':
            self.gat_layer = GraphAttentionLayer(input_dim, hidden_dim, dropout=dropout, alpha=0.2, concat=True)
            self.dropout = nn.Dropout(dropout)
            
        else:
            raise ValueError(f"Unknown topology encoder type: {self.type}")

    def forward(self, x, adj=None):
        # x: [B, T, N, F] or [B, T, F_flat] depending on upstream
        # Assuming x comes in as [B, T, N, F] for topology processing
        
        if x.dim() == 3:
            # If flattened [B, T, F_flat], try to reshape if we know N
            # For now, assume x is [B, T, F] and we treat F as nodes? No.
            # If input is flattened, we can't easily do graph ops without reshaping.
            # Assuming the caller handles reshaping or x is already [B, T, N, F]
            pass

        if self.type == 'st_gcn':
            # x: [B, T, N, F]
            B, T, N, F = x.shape
            x = x.view(B*T, N, F)
            
            # Simple GCN: AXW
            if adj is None:
                adj = torch.ones(N, N).to(x.device) # Fully connected if no adj
                adj = adj.unsqueeze(0).expand(B*T, -1, -1)
            
            # Support = XW
            support = self.gcn(x) # [B*T, N, H]
            # Output = A * Support
            x = torch.bmm(adj, support) # [B*T, N, H]
            x = F.relu(x)
            
            # Reshape for TCN: [B, H*N, T] or [B*N, H, T]
            # Let's pool nodes for temporal processing or keep them?
            # Usually ST-GCN keeps nodes.
            x = x.view(B, T, N, -1).permute(0, 3, 2, 1).contiguous() # [B, H, N, T]
            x = x.view(B, -1, T) # Flatten nodes into channels for TCN: [B, H*N, T]
            
            # TCN
            # Note: This changes channel dim to hidden_dim. 
            # We need to be careful about dimensions.
            # Let's apply TCN per node? Or global?
            # For simplicity in this architecture, let's average nodes after GCN
            x = x.view(B, -1, N, T).mean(dim=2) # [B, H, T]
            x = self.tcn(x)
            x = self.dropout(x)
            return x.transpose(1, 2) # [B, T, H]

        elif self.type == 'gat':
            # x: [B, T, N, F]
            B, T, N, F = x.shape
            x = x.view(B*T, N, F)
            
            if adj is None:
                adj = torch.ones(B*T, N, N).to(x.device)
            elif adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(B*T, -1, -1)
                
            x = self.gat_layer(x, adj) # [B*T, N, H]
            x = self.dropout(x)
            
            # Pooling nodes
            x = x.view(B, T, N, -1).mean(dim=2) # [B, T, H]
            return x

        return x
