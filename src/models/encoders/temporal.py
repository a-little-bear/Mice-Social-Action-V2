import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class FastAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = dropout

    def forward(self, q, k, v, is_causal=False):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal)

class MultiScaleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernels=[3, 5, 9, 15, 31]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.GELU()
            ) for k in kernels
        ])
        self.fusion = nn.Sequential(
            nn.Conv1d(len(kernels) * (hidden_dim // 2), hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.residual = nn.Conv1d(input_dim, hidden_dim, kernel_size=1) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)
        out = self.fusion(out)
        return out + res

class SqueezeFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        # Pre-Norm architecture for better stability
        self.norm1 = nn.LayerNorm(dim)
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.fast_attn = FastAttention(dropout=dropout)
        
        # Dual-FFN Style Convolution / Depthwise Separable
        self.norm2 = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.GELU(),
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim), # Depthwise
            nn.GELU(),
            nn.Conv1d(dim, dim, 1), # Pointwise
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        
        # Attention with residual
        res = x
        x = self.norm1(x)
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        x = self.fast_attn(q, k, v)
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.dropout(self.out_proj(x))
        x = x + res
        
        # Convolution with residual
        res = x
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = x + res
        return x

class WaveNetBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.filter_conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation//2)
        self.gate_conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation//2)
        self.skip_conv = nn.Conv1d(channels, channels, 1)
        self.res_conv = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        x = filter_out * gate_out
        x = self.dropout(x)
        
        s = self.skip_conv(x)
        r = self.res_conv(x)
        return r + res, s

class WaveNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=8, dropout=0.1):
        super().__init__()
        self.start_conv = nn.Conv1d(input_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList([
            WaveNetBlock(hidden_dim, 3, 2**(i % 8), dropout=dropout) for i in range(num_layers)
        ])
        self.end_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 1)
        )

    def forward(self, x):
        x = self.start_conv(x)
        skip_connections = []
        for block in self.blocks:
            x, s = block(x)
            skip_connections.append(s)
        
        x = torch.sum(torch.stack(skip_connections), dim=0)
        x = self.end_conv(x)
        return x

class TemporalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.type = config['type']
        input_dim = config.get('input_dim', 32) 
        hidden_dim = config['hidden_dim']
        dropout = config.get('dropout', 0.1)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = None
        
        if self.type == '1d_cnn':
            self.model = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            )
        elif self.type == 'multi_scale_cnn':
            self.model = MultiScaleCNN(hidden_dim, hidden_dim)
        elif self.type == 'transformer':
            self.pos_encoding = PositionalEncoding(hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=config.get('transformer_heads', 8), 
                dim_feedforward=hidden_dim*4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.model = nn.TransformerEncoder(encoder_layer, num_layers=config.get('num_layers', 6))
        elif self.type == 'squeezeformer':
            self.pos_encoding = PositionalEncoding(hidden_dim)
            self.model = nn.Sequential(*[
                SqueezeFormerBlock(hidden_dim, config.get('transformer_heads', 8), dropout=dropout) 
                for _ in range(config.get('num_layers', 6))
            ])
        elif self.type == 'bi_lstm':
            # Note: Bi-LSTM output is 2 * hidden_dim
            self.model = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2, # Halve to maintain hidden_dim at output
                num_layers=config.get('num_layers', 2),
                batch_first=True,
                dropout=dropout if config.get('num_layers', 1) > 1 else 0,
                bidirectional=True
            )
        elif self.type == 'wavenet':
            self.model = WaveNet(hidden_dim, hidden_dim, num_layers=config.get('num_layers', 12), dropout=dropout)
        else:
            raise ValueError(f"Unknown temporal encoder type: {self.type}")

    def forward(self, x):
        x = self.input_proj(x)
        
        if self.pos_encoding:
            x = self.pos_encoding(x)
            
        if self.type in ['1d_cnn', 'multi_scale_cnn', 'wavenet']:
            x = x.transpose(1, 2) 
            x = self.model(x)
            x = x.transpose(1, 2) 
        elif self.type == 'bi_lstm':
            x, _ = self.model(x)
        else:
            x = self.model(x)
            
        return x
