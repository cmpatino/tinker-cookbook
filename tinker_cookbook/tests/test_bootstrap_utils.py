"""
Unit tests for bootstrap_utils.py

Tests pass@k computation and bootstrap standard error estimation.
"""

import pytest

from tinker_cookbook.eval.bootstrap_utils import (
    bootstrap_mean,
    bootstrap_stderr,
    mean_metric,
    mean_stderr,
    pass_at_k,
)


class TestPassAtK:
    """Tests for pass@k computation."""

    def test_all_correct(self):
        """When all samples are correct, pass@k should be 1.0."""
        scores = [1, 1, 1, 1, 1]
        assert pass_at_k(scores, k=1, n=5) == 1.0
        assert pass_at_k(scores, k=3, n=5) == 1.0
        assert pass_at_k(scores, k=5, n=5) == 1.0

    def test_all_incorrect(self):
        """When all samples are incorrect, pass@k should be 0.0."""
        scores = [0, 0, 0, 0, 0]
        assert pass_at_k(scores, k=1, n=5) == 0.0
        assert pass_at_k(scores, k=3, n=5) == 0.0

    def test_known_value(self):
        """Test pass@k with a known calculation."""
        # 3 correct out of 5, k=1
        # Expected: 1 - (2/3 * 3/4 * 4/5) = 1 - 0.4 = 0.6
        scores = [1, 0, 1, 0, 1]
        result = pass_at_k(scores, k=1, n=5)
        assert abs(result - 0.6) < 0.001

    def test_known_value_k2(self):
        """Test pass@k with k=2."""
        # 3 correct out of 5, k=2
        # Expected: 1 - prod(1 - 2/[3,4,5])
        #         = 1 - (1/3 * 1/2 * 3/5)
        #         = 1 - 0.1 = 0.9
        scores = [1, 0, 1, 0, 1]
        result = pass_at_k(scores, k=2, n=5)
        assert abs(result - 0.9) < 0.001

    def test_edge_case_k_equals_n(self):
        """When k equals n, we sample all n, so pass@n = 1.0 if any are correct."""
        scores = [1, 1, 0]
        # Sampling all 3 means we get both correct ones
        assert pass_at_k(scores, k=3, n=3) == 1.0

        scores = [1, 1, 1]
        assert pass_at_k(scores, k=3, n=3) == 1.0

        scores = [0, 0, 0]
        # Only case where pass@n = 0 is when all are incorrect
        assert pass_at_k(scores, k=3, n=3) == 0.0

    def test_edge_case_k_greater_than_n_minus_c(self):
        """When we have enough correct samples (c >= k), pass@k should be high."""
        # 4 correct out of 5, k=2
        scores = [1, 1, 1, 1, 0]
        result = pass_at_k(scores, k=2, n=5)
        # With 4 correct, we're guaranteed to get at least 2
        assert result == 1.0

    def test_single_sample(self):
        """Test with n=1."""
        assert pass_at_k([1], k=1, n=1) == 1.0
        assert pass_at_k([0], k=1, n=1) == 0.0

    def test_binary_scores(self):
        """Verify pass@k works with 0/1 scores only."""
        scores = [1, 0, 1, 1, 0, 1]
        # Should not raise
        result = pass_at_k(scores, k=1, n=6)
        assert 0.0 <= result <= 1.0


class TestMeanMetrics:
    """Tests for mean and stderr computation."""

    def test_mean_metric(self):
        """Test mean computation."""
        assert mean_metric([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0
        assert mean_metric([0.5, 0.5]) == 0.5

    def test_mean_metric_empty(self):
        """Empty list should return 0."""
        assert mean_metric([]) == 0.0

    def test_mean_stderr(self):
        """Test standard error computation."""
        # For [1, 2, 3, 4, 5]: std = sqrt(2), stderr = sqrt(2)/sqrt(5)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = mean_stderr(values)
        expected = 1.5811388300841898 / 2.23606797749979  # ~0.707
        assert abs(result - expected) < 0.001

    def test_mean_stderr_single_value(self):
        """Single value should return 0."""
        assert mean_stderr([1.0]) == 0.0

    def test_mean_stderr_empty(self):
        """Empty list should return 0."""
        assert mean_stderr([]) == 0.0


class TestBootstrap:
    """Tests for bootstrap estimation."""

    def test_bootstrap_stderr_deterministic(self):
        """With fixed seed, bootstrap should be reproducible."""
        population = [0.8, 0.6, 0.9, 0.7, 0.5]

        result1 = bootstrap_stderr(mean_metric, population, num_iterations=100, seed=42)
        result2 = bootstrap_stderr(mean_metric, population, num_iterations=100, seed=42)

        assert result1 == result2

    def test_bootstrap_stderr_different_seeds(self):
        """Different seeds should give different results."""
        population = [0.8, 0.6, 0.9, 0.7, 0.5]

        result1 = bootstrap_stderr(mean_metric, population, num_iterations=100, seed=42)
        result2 = bootstrap_stderr(mean_metric, population, num_iterations=100, seed=43)

        # Very unlikely to be exactly equal
        assert result1 != result2

    def test_bootstrap_mean_approximates_true_mean(self):
        """Bootstrap mean should approximate the true mean."""
        population = [0.8, 0.6, 0.9, 0.7, 0.5]
        true_mean = mean_metric(population)

        bootstrap_mean_estimate = bootstrap_mean(
            mean_metric, population, num_iterations=1000, seed=42
        )

        # Should be very close for large num_iterations
        assert abs(bootstrap_mean_estimate - true_mean) < 0.01

    def test_bootstrap_empty_population(self):
        """Empty population should return 0."""
        result = bootstrap_stderr(mean_metric, [], num_iterations=100, seed=42)
        assert result == 0.0

    def test_bootstrap_single_value(self):
        """Single value should return the metric value with 0 stderr."""
        population = [0.5]

        stderr = bootstrap_stderr(mean_metric, population, num_iterations=100, seed=42)
        assert stderr == 0.0

        mean = bootstrap_mean(mean_metric, population, num_iterations=100, seed=42)
        assert mean == 0.5

    def test_bootstrap_with_custom_metric(self):
        """Test bootstrap with a custom metric function."""

        def max_metric(values):
            return max(values) if values else 0.0

        population = [0.5, 0.8, 0.6, 0.9, 0.7]

        stderr = bootstrap_stderr(max_metric, population, num_iterations=100, seed=42)
        # Max is more variable than mean in resamples
        assert stderr > 0.0

        mean = bootstrap_mean(max_metric, population, num_iterations=100, seed=42)
        # Bootstrap mean of max should be close to but slightly less than true max
        assert 0.8 < mean <= 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
