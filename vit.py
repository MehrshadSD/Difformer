import torch
from torch import nn, pi
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.dim = dim
        self.inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dim_cond, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            cond_block = nn.Linear(dim_cond, dim, bias = False)
            nn.init.zeros_(cond_block.weight)
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout),
                cond_block
            ]))
            
    def forward(self, x, cond=None):
        for attn, ff, cond_block in self.layers:
            cx = 0
            if cond is not None:
                cx = cond_block(cond)
            x = attn(x) + x
            x = (ff(x) + cx) * (cx + 1.) + x

        return self.norm(x)

def divisible_by(num, den):
    return (num % den) == 0

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        # x = rearrange(x, 'b s 1 -> b s 1')
        freqs = x * rearrange(self.weights, 'd -> 1 1 d') * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class DenoiseViT(nn.Module):
    def __init__(self, *,  
                 dim_cond,
                 dim_input,
                 dim, 
                 depth, 
                 heads, 
                 mlp_dim, 
                 dim_head = 64, 
                 dropout = 0., 
                 emb_dropout = 0.):
        super().__init__()

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(dim_input),
            nn.Linear(dim_input, dim),
            nn.LayerNorm(dim),
        )

        self.to_time_emb = nn.Sequential(
            LearnedSinusoidalPosEmb(dim_cond),
            nn.Linear(dim_cond + 1, dim),
        )
        
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dim_cond, dropout)

        self.to_denoise = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim_input)
        )

    def forward(
        self, 
        noised,
        times,
        cond 
    ):
        x = self.to_patch_embedding(noised)

        time_emb = self.to_time_emb(times)
        x += time_emb

        x = self.dropout(x)
        x = self.transformer(x, cond = cond)

        return self.to_denoise(x)