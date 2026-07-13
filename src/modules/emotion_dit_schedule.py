from __future__ import annotations

import torch
import torch.nn as nn


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        num_steps: int,
        mode: str = "linear",
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        s: float = 0.008,
    ) -> None:
        super().__init__()
        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, num_steps)
        elif mode == "quadratic":
            betas = torch.linspace(beta_1**0.5, beta_T**0.5, num_steps) ** 2
        elif mode == "sigmoid":
            betas = torch.sigmoid(torch.linspace(-5, 5, num_steps))
            betas = betas * (beta_T - beta_1) + beta_1
        elif mode == "cosine":
            x = torch.linspace(0, num_steps, num_steps + 1)
            alpha_bars = torch.cos(
                ((x / num_steps) + s) / (1 + s) * torch.pi * 0.5
            ) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
            betas = torch.clamp(betas, 0.0001, 0.999)
        else:
            raise ValueError(f"Unknown diffusion schedule: {mode}")

        betas = torch.cat([torch.zeros(1), betas], dim=0)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars[0] = 1.0

        sigmas_flex = torch.sqrt(betas)
        posterior_variance = torch.zeros_like(betas)
        posterior_variance[1:] = (
            (1 - alpha_bars[:-1]) / (1 - alpha_bars[1:]) * betas[1:]
        )
        sigmas_inflex = torch.sqrt(torch.clamp(posterior_variance, min=0.0))

        self.num_steps = int(num_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            1, self.num_steps + 1, (batch_size,), device=device, dtype=torch.long
        )

    def get_sigmas(self, t: int, flexibility: float = 0.0) -> torch.Tensor:
        if not 0.0 <= flexibility <= 1.0:
            raise ValueError("flexibility must be in [0, 1]")
        return (
            self.sigmas_flex[t] * flexibility
            + self.sigmas_inflex[t] * (1 - flexibility)
        )
