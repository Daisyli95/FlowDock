"""
bfn_utils.py
Continuous-time Bayesian Flow Network (BFN) schedule and helper utilities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


# ------------------------------------------------------------------
# 1. Core BFN schedule
# ------------------------------------------------------------------
class BFNScheduler(nn.Module):
    """
    Implements the continuous-time BFN schedule:
        α(t) = 1 - exp(-β t)              (precision)
        σ(t) = exp(-β t / 2)              (std-dev)
    with β > 0 a hyper-parameter (default 4.0).
    """

    def __init__(self, beta: float = 4.0):
        super().__init__()
        self.beta = float(beta)

    # --------------------------------------------------------------
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """α(t) = 1 - exp(-β t)"""
        return 1.0 - torch.exp(-self.beta * t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """σ(t) = √(1 - α(t)) = exp(-β t / 2)"""
        return torch.exp(-0.5 * self.beta * t)

    # --------------------------------------------------------------
    def corrupt(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Corrupt clean data x0 → y(t) = α(t) x0 + σ(t) ε, ε∼N(0, I).
        Args
            x0 : (*shape) clean samples
            t  : (*batch) time in [0, 1]
        Returns
            y  : (*shape) noisy samples
        """
        eps = torch.randn_like(x0)
        a = self.alpha(t).reshape(-1, *([1] * (x0.ndim - 1)))
        s = self.sigma(t).reshape(-1, *([1] * (x0.ndim - 1)))
        return a * x0 + s * eps

    # --------------------------------------------------------------
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time embedding fed to the network = α(t) (scalar).
        If you need a vectorial embedding, use `sinusoidal_embedding`
        afterwards.
        """
        return self.alpha(t).clamp(min=1e-5)   # avoid 0 at t=0


# ------------------------------------------------------------------
# 2. Time embeddings (unchanged from original code)
# ------------------------------------------------------------------
def sinusoidal_embedding(timesteps: torch.Tensor,
                         embedding_dim: int,
                         max_positions: int = 10000) -> torch.Tensor:
    """
    Classic sinusoidal embedding used in transformers / diffusion.
    timesteps: (*B,) tensor of scalars (e.g. α(t))
    """
    assert timesteps.ndim == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero-pad
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embedding for time.
    """
    def __init__(self, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def get_timestep_embedding(embedding_type: str,
                           embedding_dim: int,
                           embedding_scale: float = 10000) -> nn.Module:
    if embedding_type == 'sinusoidal':
        return lambda x: sinusoidal_embedding(embedding_scale * x, embedding_dim)
    elif embedding_type == 'fourier':
        return GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


# ------------------------------------------------------------------
# 3. Sampling (ODE solver)
# ------------------------------------------------------------------
@torch.no_grad()
def sample_bfn(model,
               shape: Tuple[int, ...],
               scheduler: BFNScheduler,
               steps: int = 50,
               device: str = "cpu",
               eps: float = 1e-3) -> torch.Tensor:
    """
    Generate samples by integrating the BFN probability-flow ODE
        dy/dt = -½ β y + ½ β α(t) v_θ(y(t), t)
    from t=1 → t=0 using simple Euler.

    Args
        model : callable (y, t) → v
        shape : (*shape) of samples to generate
        scheduler : BFNScheduler instance
        steps     : number of Euler steps
        device    : device
        eps       : small offset to avoid t=0 exactly
    Returns
        x0 : (*shape) generated samples
    """
    dt = -1.0 / steps
    t = torch.linspace(1.0, 0.0, steps + 1, device=device)

    y = torch.randn(shape, device=device)  # y(1) ~ N(0, I)

    for k in range(steps):
        tk = t[k].expand(shape[0])
        v = model(y, scheduler(tk))  # predicted ε
        a = scheduler.alpha(tk).reshape(-1, *([1] * (y.ndim - 1)))
        beta = scheduler.beta
        y = y + dt * (-0.5 * beta * y + 0.5 * beta * a * v)

    # Final denoising at t=0
    return y / scheduler.alpha(torch.tensor(eps, device=device))


# ------------------------------------------------------------------
# 4. One global scheduler instance (convenient)
# ------------------------------------------------------------------
bfn_scheduler = BFNScheduler(beta=4.0)   # import this everywhere
