from __future__ import annotations
from typing import Literal, Tuple, Optional, List
import torch
from torch import nn, Tensor

# tools
ACTIVATIONS = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
}

def fc_block(in_dim: int,
             hidden_dim: int,
             out_dim: int,
             layers: int,
             dropout: float,
             activation: Literal['relu', 'silu'] = 'relu') -> nn.Sequential:
    """
    a full forward 
    layers ≥ 2
    """
    if layers < 2:
        raise ValueError('layers must be at least 2')
    act = ACTIVATIONS[activation]
    seq: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), act(), nn.Dropout(dropout)]
    for _ in range(layers - 2):
        seq += [nn.Linear(hidden_dim, hidden_dim), act(), nn.Dropout(dropout)]
    seq += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*seq)


class GaussianSmearing(nn.Module):
    """Gaussian encoding"""
    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.register_buffer('offset', offset)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * dist.pow(2))


class _BaseAtomEncoder(nn.Module):
    """
 categorical embedding

    """
    def __init__(self,
                 emb_dim: int,
                 categorical_dims: List[int],
                 scalar_dim: int,
                 lm_dim: Optional[int] = None):
        super().__init__()
        self.emb_dim = emb_dim
        self.scalar_dim = scalar_dim
        self.lm_dim = lm_dim or 0

        # categorical embeddings
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(d, emb_dim) for d in categorical_dims
        )

        # scalar projector
        if scalar_dim:
            self.scalar_proj = nn.Linear(scalar_dim, emb_dim)
        else:
            self.register_parameter('scalar_proj', None)  # type: ignore

        # lm projector
        if self.lm_dim:
            self.lm_proj = nn.Linear(emb_dim + self.lm_dim, emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, cat_N + scalar_dim (+ lm_dim)]
        """
        cat_N = len(self.cat_embeddings)
        expected = cat_N + self.scalar_dim + self.lm_dim
        if x.size(1) != expected:
            raise ValueError(f'Expecting {expected} feature columns, got {x.size(1)}')

        # categorical sum
        cat_part = x[:, :cat_N].long()
        emb_list = [emb(cat_part[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        out = torch.stack(emb_list, dim=0).sum(0)  # [B, emb_dim]

        # scalar
        if self.scalar_dim:
            scalar_part = x[:, cat_N: cat_N + self.scalar_dim]
            out = out + self.scalar_proj(scalar_part)  # type: ignore

        # lm
        if self.lm_dim:
            lm_part = x[:, -self.lm_dim:]
            out = self.lm_proj(torch.cat([out, lm_part], dim=1))

        return out


class AtomEncoder(_BaseAtomEncoder):
    """
    新接口：feature_dims = (categorical_dims, scalar_dim)
    """
    def __init__(self,
                 emb_dim: int,
                 feature_dims: Tuple[List[int], int],
                 sigma_embed_dim: int,
                 lm_embedding_dim: int = 0):
        cat_dims, scalar = feature_dims
        super().__init__(emb_dim, cat_dims, scalar + sigma_embed_dim, lm_embedding_dim)


