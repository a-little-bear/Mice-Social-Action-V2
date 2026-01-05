import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, temporal_dim, context_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(temporal_dim, hidden_dim)
        self.key = nn.Linear(context_dim, hidden_dim)
        self.value = nn.Linear(context_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        
        self.output_proj = nn.Linear(temporal_dim + hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, temporal_features, context_features):
        # temporal_features: (B, T, D_t)
        # context_features: (B, D_c) -> (B, 1, D_c)
        
        B, T, _ = temporal_features.shape
        context_expanded = context_features.unsqueeze(1) # (B, 1, D_c)
        
        # Q: from temporal features (B, T, H)
        # K, V: from context features (B, 1, H)
        q = self.query(temporal_features)
        k = self.key(context_expanded)
        v = self.value(context_expanded)
        
        # Attention scores: (B, T, 1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Context context: (B, T, H)
        context_out = torch.matmul(attn_weights, v)
        
        # Concatenate and project
        combined = torch.cat([temporal_features, context_out], dim=-1)
        out = self.output_proj(combined)
        
        return self.norm(out)

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
