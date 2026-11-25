"""Optimize Gaussian membership functions with a GPU-batched genetic algorithm."""

import torch
from torch import Tensor

from auto_predict.models.anfis import ANFIS, CONSEQUENT_PENALTY, weight_observations


def _fitness(
    model: ANFIS,
    population: Tensor,
    train: tuple[Tensor, Tensor],
    validation: tuple[Tensor, Tensor],
    *,
    penalty: float = CONSEQUENT_PENALTY,
    log_target: bool = False,
    weights: Tensor | None = None,
) -> Tensor:
    centers, log_widths = population.unbind(dim=1)
    features, target = train
    design = model.design_matrix(features, centers=centers, log_widths=log_widths)
    design, target = weight_observations(design, target, weights)
    transpose = design.transpose(-2, -1)
    gram = (transpose @ design) / len(target)
    gram.diagonal(dim1=-2, dim2=-1).add_(penalty)
    rhs = (transpose @ target).unsqueeze(-1) / len(target)
    coefficients = torch.linalg.solve(gram, rhs)
    del transpose, design

    val_features, val_target = validation
    val_design = model.design_matrix(val_features, centers=centers, log_widths=log_widths)
    predictions = (val_design @ coefficients).squeeze(-1)
    if log_target:
        predictions = predictions.exp()
    return (predictions - val_target).abs().mean(dim=1)


def _ordered_population(population: Tensor) -> Tensor:
    population[:, 0].clamp_(-3.0, 3.0)
    population[:, 1].clamp_(-2.0, 1.0)
    # Keep the lower-center term first and move its width with it.
    order = population[:, :1].argsort(dim=-1).expand_as(population)
    return population.gather(-1, order)


@torch.no_grad()
def search_memberships(
    model: ANFIS,
    train: tuple[Tensor, Tensor],
    validation: tuple[Tensor, Tensor],
    population_size: int,
    generations: int,
    seed: int,
    *,
    penalty: float = CONSEQUENT_PENALTY,
    log_target: bool = False,
    weights: Tensor | None = None,
    mutation_scale: float = 0.3,
) -> tuple[Tensor, list[float]]:
    """Return the best chromosome and the best validation MAE by generation."""
    if population_size < 4 or generations < 1:
        raise ValueError("The search requires at least four candidates and one generation.")
    if penalty <= 0 or mutation_scale <= 0:
        raise ValueError("The consequent penalty and mutation scale must be positive.")

    features, target = train
    if log_target:
        train = features, target.log()
    device = features.device
    generator = torch.Generator(device=device).manual_seed(seed)
    quantiles = torch.linspace(0.25, 0.75, model.centers.shape[1], device=device)
    quartiles = torch.quantile(features, quantiles, dim=0).T
    initial = torch.stack((quartiles, torch.zeros_like(quartiles)))
    noise = torch.randn((population_size, *initial.shape), device=device, generator=generator)
    population = initial + 0.5 * noise
    population[0] = initial
    population = _ordered_population(population)

    history = []
    for generation in range(generations):
        scores = _fitness(
            model, population, train, validation,
            penalty=penalty, log_target=log_target, weights=weights,
        )
        elites = scores.topk(2, largest=False).indices
        history.append(scores[elites[0]].item())
        if generation == generations - 1:
            break

        competitors = torch.randint(
            population_size, (population_size - 2, 2, 3), device=device, generator=generator
        )
        winner_positions = scores[competitors].argmin(dim=-1, keepdim=True)
        winners = competitors.gather(-1, winner_positions).squeeze(-1)
        first, second = population[winners].unbind(dim=1)
        mix = torch.rand(first.shape, device=device, generator=generator)
        children = first.lerp(second, mix)
        mutate = torch.rand(children.shape, device=device, generator=generator) < 0.25
        mutation = torch.randn(children.shape, device=device, generator=generator) * mutation_scale
        children += mutate * mutation
        population = torch.cat((population[elites], _ordered_population(children)))

    return population[elites[0]], history
