import torch
import torch.nn as nn
from .encoders.temporal import TemporalEncoder
from .encoders.spatial import SpatialEncoder
from .components.lca import LabContextAdapter
from .components.fusion import AttentionFusion, GatedFusion
from src.data.features import FeatureGenerator
from src.data.skeleton import get_mouse_skeleton_adj

class HHSTFModel(nn.Module):
    def __init__(self, config, feature_generator=None):
        super().__init__()
        self.config = config
        self.feature_generator = feature_generator
        
        self.spatial_encoder = None
        
        # We need adj buffer even before spatial_encoder if we want to register it properly
        self.register_buffer('adj', None) 
        
        if config['spatial_encoder']['enabled']:
            # Use SpatialEncoder with full config
            self.spatial_encoder = SpatialEncoder(config['spatial_encoder'])
            
            # Load adjacency matrix if enabled
            if config['spatial_encoder'].get('use_adj', False):
                num_nodes = config['spatial_encoder'].get('num_nodes', None)
                adj = self._get_adjacency_matrix(num_nodes)
                if adj is not None:
                    self.register_buffer('adj', adj) # Register as buffer to move with model
            
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

    def _get_adjacency_matrix(self, num_nodes=None):
        if num_nodes:
            return get_mouse_skeleton_adj(num_nodes, strategy='normalized')
        return None

    def forward(self, x, lab_ids=None, subject_ids=None):
        if self.feature_generator:
            x = self.feature_generator(x)
            
        if self.spatial_encoder:
            # spatial_encoder now handles reshaping internally if necessary
            # We just pass x and adj
            # adj is automatically on the correct device if registered as buffer
            x = self.spatial_encoder(x, adj=self.adj)
            
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
