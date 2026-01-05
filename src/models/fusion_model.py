import torch
import torch.nn as nn
from .encoders.temporal import TemporalEncoder
from .encoders.spatial import SpatialGNN
from .components.lca import LabContextAdapter
from .components.fusion import AttentionFusion, GatedFusion
from src.data.features import FeatureGenerator

class HHSTFModel(nn.Module):
    def __init__(self, config, feature_generator=None):
        super().__init__()
        self.config = config
        self.feature_generator = feature_generator
        
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
            
        temporal_dim = config['temporal_backbone']['hidden_dim']
        fusion_type = config['fusion'].get('type', 'concat')
        fusion_hidden_dim = config['fusion'].get('hidden_dim', 1024)
        
        self.fusion_module = None
        final_dim = temporal_dim
        
        if self.context_adapter:
            context_dim = config['context_adapter']['embedding_dim']
            
            if fusion_type == 'attention':
                self.fusion_module = AttentionFusion(temporal_dim, context_dim, fusion_hidden_dim)
                final_dim = fusion_hidden_dim
            elif fusion_type == 'gated':
                self.fusion_module = GatedFusion(temporal_dim, context_dim, fusion_hidden_dim)
                final_dim = fusion_hidden_dim
            else: # concat
                final_dim += context_dim
            
        self.classifier = nn.Linear(final_dim, config['classifier']['num_classes'])
        
        self.two_stage_head = None
        if config.get('two_stage', {}).get('enabled', False):
            self.two_stage_head = nn.Linear(final_dim, 1)

    def forward(self, x, lab_ids=None, subject_ids=None):
        if self.feature_generator:
            x = self.feature_generator(x)
            
        if self.spatial_encoder:
            B, T, F = x.shape
            x_flat = x.view(B*T, F)
            x_spatial = self.spatial_encoder(x_flat, adj=None)
            x = x_spatial.view(B, T, -1)
            
        features = self.temporal_encoder(x)
        
        if self.context_adapter and lab_ids is not None:
            context = self.context_adapter(lab_ids)
            
            if self.fusion_module:
                features = self.fusion_module(features, context)
            else:
                # Default concat behavior
                context_expanded = context.unsqueeze(1).expand(-1, features.shape[1], -1)
                features = torch.cat([features, context_expanded], dim=-1)
            
        logits = self.classifier(features)
        
        if self.two_stage_head:
            detection_logits = self.two_stage_head(features)
            return logits, detection_logits
            
        return logits
