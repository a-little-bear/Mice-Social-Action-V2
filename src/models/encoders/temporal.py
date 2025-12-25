import torch
import torch.nn as nn

class MultiScaleCNN(nn.Module):
    """
    Multi-scale CNN from 2nd place solution.
    Kernels: [3, 5, 7, 9]
    """
    def __init__(self, input_dim, hidden_dim, kernels=[3, 5, 7, 9]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ) for k in kernels
        ])
        self.fusion = nn.Conv1d(hidden_dim * len(kernels), hidden_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, C, T]
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)
        out = self.fusion(out)
        return out

class SqueezeFormerBlock(nn.Module):
    """
    SqueezeFormer Block implementation.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim), # Depthwise
            nn.GELU(),
            nn.Conv1d(dim, dim, 1) # Pointwise
        )
    
    def forward(self, x):
        # x: [B, T, C]
        # Attention
        res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + res
        
        # Conv
        res = x
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = x + res
        return x

class WaveNetBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.filter_conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation//2)
        self.gate_conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation//2)
        self.skip_conv = nn.Conv1d(channels, channels, 1)
        self.res_conv = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        x = filter_out * gate_out
        s = self.skip_conv(x)
        r = self.res_conv(x)
        return r, s

class WaveNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.start_conv = nn.Conv1d(input_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList([
            WaveNetBlock(hidden_dim, 3, 2**i) for i in range(num_layers)
        ])
        self.end_conv = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, x):
        # x: [B, C, T]
        x = self.start_conv(x)
        skip_connections = []
        for block in self.blocks:
            x, s = block(x)
            skip_connections.append(s)
        x = torch.sum(torch.stack(skip_connections), dim=0)
        x = self.end_conv(x)
        return x

class TemporalEncoder(nn.Module):
    """
    Module B: Stream 2 - Temporal Stream
    Supports: 1D-CNN, Transformer, WaveNet, Squeezeformer, MultiScaleCNN
    """
    def __init__(self, config):
        super().__init__()
        self.type = config['type']
        input_dim = config.get('input_dim', 32) # Example default
        hidden_dim = config['hidden_dim']
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        if self.type == '1d_cnn':
            self.model = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            )
        elif self.type == 'multi_scale_cnn':
            self.model = MultiScaleCNN(hidden_dim, hidden_dim)
        elif self.type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=config.get('transformer_heads', 4), batch_first=True)
            self.model = nn.TransformerEncoder(encoder_layer, num_layers=config.get('transformer_layers', 4))
        elif self.type == 'squeezeformer':
            # Stack multiple SqueezeFormer blocks
            self.model = nn.Sequential(*[
                SqueezeFormerBlock(hidden_dim, config.get('transformer_heads', 4)) 
                for _ in range(config.get('transformer_layers', 4))
            ])
        elif self.type == 'bi_lstm':
            self.model = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=config.get('lstm_layers', 2),
                batch_first=True,
                bidirectional=True
            )
            # Output dim will be hidden_dim * 2
        elif self.type == 'wavenet':
            self.model = WaveNet(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown temporal encoder type: {self.type}")

    def forward(self, x):
        # x: [B, T, C]
        x = self.input_proj(x)
        
        if self.type in ['1d_cnn', 'multi_scale_cnn', 'wavenet']:
            x = x.transpose(1, 2) # [B, C, T]
            x = self.model(x)
            x = x.transpose(1, 2) # [B, T, C]
        elif self.type == 'bi_lstm':
            x, _ = self.model(x)
        else:
            x = self.model(x)
            
        return x
