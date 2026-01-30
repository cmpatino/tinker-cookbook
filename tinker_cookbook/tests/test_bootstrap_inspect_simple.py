"""
Simple test for bootstrapped inspect evaluator.

This test creates a minimal in-memory dataset to verify the bootstrapped
evaluator works end-to-end without requiring an API key or network access.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate


@task
def simple_math_task() -> Task:
    """
    Simple math task for testing pass@k evaluation.

    Each problem has a clear numeric answer that can be extracted.
    """
    return Task(
        name="simple_math",
        dataset=MemoryDataset(
            name="simple_math",
            samples=[
                Sample(input="What is 2 + 2?", target="4"),
                Sample(input="What is 5 * 3?", target="15"),
            ],
        ),
        solver=generate(),
        scorer=match(),
    )


if __name__ == "__main__":
    # This is just for development - actual tests would be run via pytest
    print("simple_math_task defined successfully")
