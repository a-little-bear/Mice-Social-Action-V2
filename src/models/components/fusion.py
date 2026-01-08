import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    """
    Refined Context-Aware Channel Attention.
    Instead of meaningless temporal softmax over a single context vector,
    this uses the context to modulate the importance of different feature channels.
    """
    def __init__(self, temporal_dim, context_dim, hidden_dim):
        super().__init__()
        self.context_to_weight = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, temporal_dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(temporal_dim, temporal_dim)
        self.norm = nn.LayerNorm(temporal_dim)

    def forward(self, temporal_features, context_features):
        # temporal_features: (B, T, D_t)
        # context_features: (B, D_c)
        
        # Generate channel-wise attention weights from context
        # (B, D_t)
        weights = self.context_to_weight(context_features)
        
        # Apply modulation
        # (B, T, D_t) * (B, 1, D_t)
        out = temporal_features * weights.unsqueeze(1)
        
        out = self.proj(out)
        return self.norm(out + temporal_features) # Residual for stability

class GatedFusion(nn.Module):
    def __init__(self, temporal_dim, context_dim, hidden_dim):
        super().__init__()
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(temporal_dim + context_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, temporal_features, context_features):
        # temporal_features: (B, T, D_t)
        # context_features: (B, D_c)
        
        B, T, _ = temporal_features.shape
        context_expanded = context_features.unsqueeze(1).expand(-1, T, -1) # (B, T, D_c)
        
        # Project to same dimension
        h_temporal = self.temporal_proj(temporal_features)
        h_context = self.context_proj(context_expanded)
        
        # Compute gate
        combined_raw = torch.cat([temporal_features, context_expanded], dim=-1)
        z = self.gate(combined_raw)
        
        # Gated fusion: z * temporal + (1-z) * context
        out = z * h_temporal + (1 - z) * h_context
        
        return self.norm(out)
