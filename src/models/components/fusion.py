import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    """
    Improved FiLM-style (Feature-wise Linear Modulation) Fusion.
    Instead of just scaling (SE-style), this learns both Scale (Gamma) and Shift (Beta).
    This prevents gradient starvation when Scale is near 0 and allows better domain adaptation.
    """
    def __init__(self, temporal_dim, context_dim, hidden_dim, dropout=0.1):
        super().__init__()
        # We project context to 2 * temporal_dim to get both Gamma and Beta
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True), # LeakyReLU better than ReLU for gradient flow
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * temporal_dim)
        )
        self.proj = nn.Linear(temporal_dim, temporal_dim)
        self.norm = nn.LayerNorm(temporal_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize Gamma to 0 (which means scale=1 after calc) and Beta to 0
        # This ensures training starts as if Fusion is Identity, preventing shock.
        nn.init.zeros_(self.context_net[-1].weight)
        nn.init.zeros_(self.context_net[-1].bias)

    def forward(self, temporal_features, context_features):
        # temporal_features: (B, T, D_t)
        # context_features: (B, D_c)
        
        # (B, 2*D_t)
        params = self.context_net(context_features)
        
        # Split into Gamma (Scale) and Beta (Shift)
        gamma, beta = torch.chunk(params, 2, dim=1)
        
        # Gamma: range (-1, 1) + 1.0 -> (0, 2). Centered at 1.0.
        # This prevents the "multiplying by near-zero" problem of Sigmoid.
        scale_factor = 1.0 + torch.tanh(gamma)
        shift_factor = beta
        
        # FiLM: Scale * Feat + Shift
        # (B, 1, D) * (B, T, D) + (B, 1, D)
        modulated = (scale_factor.unsqueeze(1) * temporal_features) + shift_factor.unsqueeze(1)
        
        # Additional projection to mix channels
        out = self.proj(modulated)
        out = self.dropout(out)
        
        # Residual connection
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
