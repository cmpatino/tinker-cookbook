"""
Bootstrap and pass@k utilities for evaluation.

Implements pass@k metric (Chen et al., 2021, arXiv:2107.03374) and bootstrap-based
standard error estimation at the problem level.
"""

import random
from typing import Callable, TypeVar

import numpy as np

T = TypeVar("T")


def pass_at_k(all_scores: list[int], k: int, n: int) -> float:
    """
    Compute pass@k: probability that at least k of n samples are correct.

    Based on Chen et al., "Evaluating Large Language Models Trained on Code"
    (arXiv:2107.03374v2).

    Formula: 1 - prod(1 - k / range(n - c + 1, n + 1))
    where c is the number of correct predictions.

    Args:
        all_scores: List of binary scores (0 or 1) for n generations
        k: Number of correct samples required
        n: Total number of samples generated

    Returns:
        Pass@k probability (0.0 to 1.0)

    Examples:
        >>> pass_at_k([1, 0, 1, 0, 1], k=1, n=5)  # 3 correct out of 5
        0.6
        >>> pass_at_k([1, 1, 1], k=1, n=3)  # All correct
        1.0
        >>> pass_at_k([0, 0, 0], k=1, n=3)  # All wrong
        0.0
    """
    c = sum(all_scores)  # count correct predictions

    if n - c < k:
        # Fewer incorrect samples than k, so we must pick a correct one
        return 1.0

    # Compute pass@k using the Chen et al. formula
    # Formula computes: 1 - P(all k samples are incorrect)
    # P(all k incorrect) = prod((n-c-i)/(n-i) for i in range(k))
    #                    = prod(1 - k / (n - c + 1 + i) for i in range(k))
    # Using arange: prod(1 - k / range(n-c+1, n+1))
    denominator = np.arange(n - c + 1, n + 1)
    result = 1.0 - np.prod(1.0 - k / denominator)
    return float(result)


def mean_metric(values: list[float]) -> float:
    """Compute mean of a list of values."""
    if not values:
        return 0.0
    return float(np.mean(values))


def mean_stderr(values: list[float]) -> float:
    """
    Compute standard error of the mean.

    Args:
        values: List of values

    Returns:
        Standard error (std / sqrt(n))
    """
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def bootstrap_mean(
    metric_fn: Callable[[list[T]], float],
    population: list[T],
    num_iterations: int = 1000,
    seed: int | None = None,
) -> float:
    """
    Compute bootstrap estimate of the mean.

    Args:
        metric_fn: Function that computes metric from a sample
        population: Full population to resample from
        num_iterations: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        Mean of bootstrap estimates
    """
    if len(population) < 2:
        return metric_fn(population) if population else 0.0

    rng = random.Random(seed)
    estimates = []

    for _ in range(num_iterations):
        # Resample with replacement
        resampled = rng.choices(population, k=len(population))
        estimates.append(metric_fn(resampled))

    return mean_metric(estimates)


def bootstrap_stderr(
    metric_fn: Callable[[list[T]], float],
    population: list[T],
    num_iterations: int = 1000,
    seed: int | None = None,
) -> float:
    """
    Compute bootstrap estimate of standard error.

    Resamples the population with replacement, computes the metric on each
    bootstrap sample, and returns the standard error of those estimates.

    Args:
        metric_fn: Function that computes metric from a sample
        population: Full population to resample from
        num_iterations: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        Standard error of bootstrap estimates
    """
    if len(population) < 2:
        return 0.0

    rng = random.Random(seed)
    estimates = []

    for _ in range(num_iterations):
        # Resample with replacement
        resampled = rng.choices(population, k=len(population))
        estimates.append(metric_fn(resampled))

    return mean_stderr(estimates)
