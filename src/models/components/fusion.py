import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    """
    Refined Context-Aware Channel Attention (Squeeze-and-Excitation style).
    Uses the context to modulate the importance of different feature channels.
    """
    def __init__(self, temporal_dim, context_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.context_to_weight = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, temporal_dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(temporal_dim, temporal_dim)
        self.norm = nn.LayerNorm(temporal_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, temporal_features, context_features):
        # temporal_features: (B, T, D_t)
        # context_features: (B, D_c)
        
        # (B, D_t)
        weights = self.context_to_weight(context_features)
        
        # Channel-wise modulation
        # (B, T, D_t) * (B, 1, D_t)
        modulated = temporal_features * weights.unsqueeze(1)
        
        out = self.proj(modulated)
        out = self.dropout(out)
        
        # Residual connection ensures the model can at least maintain backbone features
        return self.norm(out + temporal_features)

class GatedFusion(nn.Module):
    """
    Learned Gated Fusion between temporal features and context.
    Allows the model to dynamically decide how much lab/subject info to incorporate.
    """
    def __init__(self, temporal_dim, context_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        
        # Shortcut for residual connection if dimensions change
        self.shortcut = nn.Linear(temporal_dim, hidden_dim) if temporal_dim != hidden_dim else nn.Identity()
        
        self.gate = nn.Sequential(
            nn.Linear(temporal_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, temporal_features, context_features):
        # temporal_features: (B, T, D_t)
        # context_features: (B, D_c)
        
        B, T, _ = temporal_features.shape
        context_expanded = context_features.unsqueeze(1).expand(-1, T, -1) # (B, T, D_c)
        
        # Project both to fusion space
        h_temporal = self.temporal_proj(temporal_features)
        h_context = self.context_proj(context_expanded)
        
        # Compute dynamic gate (per channel, per time step)
        combined_raw = torch.cat([temporal_features, context_expanded], dim=-1)
        z = self.gate(combined_raw)
        
        # Gated blending: z * temporal + (1-z) * context
        blended = z * h_temporal + (1 - z) * h_context
        
        # Apply dropout to the blended features
        out = self.dropout(blended)
        
        # Add residual connection from input temporal features
        res = self.shortcut(temporal_features)
        
        return self.norm(out + res)
