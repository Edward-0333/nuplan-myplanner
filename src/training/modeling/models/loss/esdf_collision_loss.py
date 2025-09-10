import torch
import torch.nn as nn


class ESDFCollisionLoss(nn.Module):
    """Placeholder ESDF-based collision loss.

    Given a best trajectory (B, T, 6) and a cost map (B, H, W) this dummy
    implementation returns zero to keep training stable when cost maps are not
    available. Replace with real ESDF sampling if desired.
    """

    def forward(self, trajectory: torch.Tensor, cost_map: torch.Tensor) -> torch.Tensor:
        # Intentionally return zero; add real sampling against cost_map if needed.
        return torch.zeros(1, device=trajectory.device, dtype=trajectory.dtype)

