import torch
import torch.nn as nn
from .encoders.temporal import TemporalEncoder
from .encoders.spatial import SpatialGNN
from .components.lca import LabContextAdapter

class HHSTFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.spatial_encoder = None
        if config['spatial_encoder']['enabled']:
            self.spatial_encoder = SpatialGNN(
                input_dim=config['temporal_backbone']['input_dim'],
                hidden_dim=config['spatial_encoder']['hidden_dim']
            )
            
        temporal_config = config['temporal_backbone'].copy()
        if self.spatial_encoder:
            temporal_config['input_dim'] = config['spatial_encoder']['hidden_dim']

        self.temporal_encoder = TemporalEncoder(
            config=temporal_config
        )
        
        self.context_adapter = None
        if config['context_adapter']['enabled']:
            self.context_adapter = LabContextAdapter(
                config=config['context_adapter']
            )
            
        fusion_dim = config['temporal_backbone']['hidden_dim']
        if self.context_adapter:
            fusion_dim += config['context_adapter']['embedding_dim']
            
        self.classifier = nn.Linear(fusion_dim, config['classifier']['num_classes'])
        
        self.two_stage_head = None
        if config.get('two_stage', {}).get('enabled', False):
            self.two_stage_head = nn.Linear(fusion_dim, 1)

    def forward(self, x, lab_ids=None, subject_ids=None):
        if self.spatial_encoder:
            B, T, F = x.shape
            # adj = torch.eye(F).to(x.device).unsqueeze(0).expand(B*T, -1, -1)
            x_flat = x.view(B*T, F)
            x_spatial = self.spatial_encoder(x_flat, adj=None)
            x = x_spatial.view(B, T, -1)
            
        features = self.temporal_encoder(x)
        
        if self.context_adapter and lab_ids is not None:
            context = self.context_adapter(lab_ids)
            context = context.unsqueeze(1).expand(-1, features.shape[1], -1)
            features = torch.cat([features, context], dim=-1)
            
        logits = self.classifier(features)
        
        if self.two_stage_head:
            detection_logits = self.two_stage_head(features)
            return logits, detection_logits
            
        return logits
