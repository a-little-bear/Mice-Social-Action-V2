import torch
import torch.nn as nn

class LabContextAdapter(nn.Module):
    """
    Module III: Lab-Context Adapter (LCA)
    Injects Lab and Subject embeddings into the model.
    """
    def __init__(self, config):
        super().__init__()
        self.lab_embedding = nn.Embedding(config.get('num_labs', 30), config['embedding_dim'])
        # Subject embedding might be tricky if subjects are not unique across labs or if we don't have a global map.
        # For now, we'll assume a simple hashing or small number of subjects per batch.
        self.subject_embedding = nn.Embedding(config.get('num_subjects', 100), config['embedding_dim'])
        self.output_dim = config['embedding_dim'] * 2
        
        # Hardcoded mapping for MABe 2022 labs
        self.lab_map = {
            'CalMS21_task1': 0, 'CalMS21_task2': 1, 'CalMS21_supplemental': 2, 
            'MABe22_keypoints': 3, 'MABe22_movies': 4, 'AdaptableSnail': 5, 
            'BoisterousParrot': 6, 'CautiousGiraffe': 7, 'DeliriousFly': 8, 
            'ElegantMink': 9, 'GroovyShrew': 10, 'InvincibleJellyfish': 11, 
            'JovialSwallow': 12, 'LyricalHare': 13, 'NiftyGoldfinch': 14, 
            'PleasantMeerkat': 15, 'ReflectiveManatee': 16, 'SparklingTapir': 17, 
            'TranquilPanther': 18, 'UppityFerret': 19, 'CRIM13': 20,
            'Unknown': 21
        }

    def forward(self, lab_ids, subject_ids=None):
        # lab_ids: Tensor of ints
        # subject_ids: Tensor of ints
        
        lab_emb = self.lab_embedding(lab_ids)
        
        if subject_ids is not None:
            sub_emb = self.subject_embedding(subject_ids)
            return torch.cat([lab_emb, sub_emb], dim=-1)
            
        return lab_emb
