"""Load a trained PPO checkpoint for inference. No skrl or Isaac dependency."""

from __future__ import annotations

import torch

SCALER_CLIP = 5.0
SCALER_EPS = 1e-8


class LegsPolicy:
    """Deterministic low-level driving policy: 44-dim observation to 2 wheel commands."""

    def __init__(self, checkpoint: str, device: str = "cuda:0", hidden: tuple[int, ...] = (256, 256)):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        layers: list[torch.nn.Module] = []
        in_dim = ckpt["policy"]["net.0.weight"].shape[1]
        for size in hidden:
            layers += [torch.nn.Linear(in_dim, size), torch.nn.LayerNorm(size), torch.nn.SiLU()]
            in_dim = size
        layers.append(torch.nn.Linear(in_dim, 2))
        self.net = torch.nn.Sequential(*layers).to(device).eval()
        self.net.load_state_dict({k[len("net."):]: v for k, v in ckpt["policy"].items() if k.startswith("net.")})
        scaler = ckpt["state_preprocessor"]
        self.mean = scaler["running_mean"].float()
        self.std = scaler["running_variance"].float().sqrt()

    @torch.inference_mode()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        scaled = (obs - self.mean) / (self.std + SCALER_EPS)
        return self.net(scaled.clamp(-SCALER_CLIP, SCALER_CLIP)).clamp(-1.0, 1.0)
