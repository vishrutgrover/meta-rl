"""Actor-critic MLP policy for the GTM environment."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Dirichlet


class GTMActorCritic(nn.Module):
    """Small actor-critic network with task-specific action heads.

    Heads:
        budget    — Dirichlet over channels        (simplex)
        segment   — Dirichlet over segments        (simplex)
        messaging — Dirichlet over 6 dimensions    (simplex)
        experiment— Categorical (incl. "none")
        pricing   — Categorical (incl. "none")

    Concentrations for the Dirichlet heads are produced by softplus(linear)+1
    so they are always positive and start near a uniform distribution.
    """

    def __init__(
        self,
        obs_dim: int,
        n_channels: int,
        n_segments: int,
        n_messaging: int,
        n_experiments: int,
        n_pricing: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_channels = n_channels
        self.n_segments = n_segments
        self.n_messaging = n_messaging
        self.n_experiments = n_experiments
        self.n_pricing = n_pricing

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.budget_head = nn.Linear(hidden_dim, n_channels)
        self.segment_head = nn.Linear(hidden_dim, n_segments)
        self.messaging_head = nn.Linear(hidden_dim, n_messaging)
        self.experiment_head = nn.Linear(hidden_dim, n_experiments)
        self.pricing_head = nn.Linear(hidden_dim, n_pricing)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[Dict[str, torch.distributions.Distribution], torch.Tensor]:
        h = self.trunk(obs)

        budget_alpha = F.softplus(self.budget_head(h)) + 1.0
        segment_alpha = F.softplus(self.segment_head(h)) + 1.0
        messaging_alpha = F.softplus(self.messaging_head(h)) + 1.0

        # Avoid sampling exact zeros which would break Dirichlet log_prob
        budget_alpha = budget_alpha.clamp(min=1e-3)
        segment_alpha = segment_alpha.clamp(min=1e-3)
        messaging_alpha = messaging_alpha.clamp(min=1e-3)

        dists: Dict[str, torch.distributions.Distribution] = {
            "budget": Dirichlet(budget_alpha),
            "segment": Dirichlet(segment_alpha),
            "messaging": Dirichlet(messaging_alpha),
            "experiment": Categorical(logits=self.experiment_head(h)),
            "pricing": Categorical(logits=self.pricing_head(h)),
        }
        value = self.value_head(h).squeeze(-1)
        return dists, value

    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Sample (or pick the mode of) an action from the policy.

        Returns: (sample_dict, total_log_prob, value)
        """
        dists, value = self.forward(obs)
        sample: Dict[str, torch.Tensor] = {}
        log_probs = []
        for name, dist in dists.items():
            if deterministic:
                if isinstance(dist, Dirichlet):
                    # mean of a Dirichlet
                    s = dist.concentration / dist.concentration.sum(dim=-1, keepdim=True)
                else:
                    s = dist.probs.argmax(dim=-1)
            else:
                s = dist.sample()
            sample[name] = s
            log_probs.append(dist.log_prob(s))
        total_log_prob = torch.stack(log_probs, dim=0).sum(dim=0)
        return sample, total_log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute log-probs, entropy, and value for stored (obs, action) pairs."""
        dists, value = self.forward(obs)
        log_probs = []
        entropies = []
        for name, dist in dists.items():
            log_probs.append(dist.log_prob(actions[name]))
            entropies.append(dist.entropy())
        total_log_prob = torch.stack(log_probs, dim=0).sum(dim=0)
        total_entropy = torch.stack(entropies, dim=0).sum(dim=0)
        return total_log_prob, total_entropy, value
