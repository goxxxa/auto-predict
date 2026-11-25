"""First-order Sugeno ANFIS and its hybrid training procedure."""

from itertools import product
from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CONSEQUENT_PENALTY = 0.001
MIN_MEMBERSHIP_WIDTH = 0.05


def weight_observations(
    design: Tensor, target: Tensor, weights: Tensor | None
) -> tuple[Tensor, Tensor]:
    if weights is None:
        return design, target
    scale = (weights / weights.mean()).sqrt()
    return design * scale[:, None], target * scale


class ANFIS(nn.Module):
    """Gaussian membership functions per feature and a rule per combination."""

    def __init__(self, n_features: int, terms: int = 2) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.zeros(n_features, terms))
        self.log_widths = nn.Parameter(torch.zeros(n_features, terms))
        rules = torch.tensor(list(product(range(terms), repeat=n_features)))
        self.register_buffer("rule_terms", rules)
        self.register_buffer("feature_indices", torch.arange(n_features))
        self.register_buffer("consequents", torch.zeros(len(rules) * (n_features + 1)))

    def rule_weights(
        self,
        features: Tensor,
        *,
        centers: Tensor | None = None,
        log_widths: Tensor | None = None,
    ) -> Tensor:
        centers = self.centers if centers is None else centers
        log_widths = self.log_widths if log_widths is None else log_widths
        widths = log_widths.exp() + MIN_MEMBERSHIP_WIDTH
        offsets = (features[..., None] - centers.unsqueeze(-3)) / widths.unsqueeze(-3)
        log_membership = -0.5 * offsets.square()
        log_strengths = log_membership[..., self.feature_indices, self.rule_terms].sum(-1)
        return log_strengths.softmax(dim=-1)

    def design_matrix(
        self,
        features: Tensor,
        *,
        centers: Tensor | None = None,
        log_widths: Tensor | None = None,
    ) -> Tensor:
        weights = self.rule_weights(features, centers=centers, log_widths=log_widths)
        linear_inputs = torch.cat((torch.ones_like(features[:, :1]), features), dim=1)
        return (weights[..., None] * linear_inputs[:, None, :]).flatten(start_dim=-2)

    def forward(self, features: Tensor) -> Tensor:
        return self.design_matrix(features) @ self.consequents

    def rule_contributions(self, features: Tensor) -> Tensor:
        weighted_terms = self.design_matrix(features) * self.consequents
        return weighted_terms.reshape(len(features), len(self.rule_terms), -1).sum(dim=-1)

    @torch.no_grad()
    def fit_consequents(
        self,
        features: Tensor,
        target: Tensor,
        *,
        penalty: float = CONSEQUENT_PENALTY,
        weights: Tensor | None = None,
    ) -> None:
        design = self.design_matrix(features)
        design, target = weight_observations(design, target, weights)
        n_coefficients = design.shape[1]
        # Ridge rows keep the system full-rank even when some rules rarely fire.
        ridge_rows = torch.eye(n_coefficients, device=design.device, dtype=design.dtype)
        ridge_rows *= sqrt(penalty * len(features))
        design = torch.cat((design, ridge_rows))
        target = torch.cat((target, target.new_zeros(n_coefficients)))
        self.consequents.copy_(torch.linalg.lstsq(design, target, driver="gels").solution)


def train_anfis(
    features: Tensor,
    target: Tensor,
    epochs: int,
    validation: tuple[Tensor, Tensor] | None = None,
) -> tuple[ANFIS, list[dict[str, float]]]:
    """Alternate gradient updates of memberships and least squares of consequents."""
    if epochs < 1:
        raise ValueError("Training requires at least one epoch.")

    model = ANFIS(features.shape[1]).to(features)
    with torch.no_grad():
        quartiles = torch.quantile(features, features.new_tensor([0.25, 0.75]), dim=0)
        model.centers.copy_(quartiles.T)
    model.fit_consequents(features, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    history = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        F.mse_loss(model(features), target).backward()
        optimizer.step()
        model.fit_consequents(features, target)

        if validation is not None:
            val_features, val_target = validation
            with torch.no_grad():
                history.append(
                    {
                        "epoch": epoch,
                        "train_mae": F.l1_loss(model(features), target).item(),
                        "validation_mae": F.l1_loss(model(val_features), val_target).item(),
                    }
                )
    return model.eval(), history
