"""
Entrypoint for running Inspect evaluations with pass@k and bootstrap stderr.

This script extends run_inspect_evals.py to support:
- Pass@k evaluation (generate n completions per problem, compute pass@k)
- Bootstrap-based standard error estimation at the problem level

Example usage:
    python -m tinker_cookbook.eval.run_bootstrapped_inspect_evals \
        model_name="Qwen/Qwen3-8B-Base" \
        model_path="tinker://workspace_id:train:checkpoint_id/sampler_weights/run_name" \
        tasks=inspect_evals/aime2024 \
        renderer_name="qwen3_instruct" \
        max_tokens=16384 \
        temperature=0.6 \
        top_p=0.95 \
        num_choices=64 \
        pass_at_k=1 \
        bootstrap_stderr=True \
        bootstrap_iters=1000
"""

import asyncio
import logging

import chz
import tinker
from tinker_cookbook.eval.inspect_evaluators import (
    BootstrappedInspectEvaluator,
    InspectEvaluatorBuilder,
)

logger = logging.getLogger(__name__)


@chz.chz
class Config(InspectEvaluatorBuilder):
    """Configuration for bootstrapped inspect evaluation."""

    model_path: str | None = None

    # Override defaults for pass@k evaluation
    pass_at_k: int | None = 1  # Default to pass@1
    bootstrap_stderr: bool = True  # Enable bootstrap by default
    bootstrap_iters: int = 1000  # Default number of bootstrap iterations
    bootstrap_seed: int | None = 42  # Default seed for reproducibility


async def main(config: Config):
    logging.basicConfig(level=logging.INFO)

    # Create a sampling client from the model path
    service_client = tinker.ServiceClient()

    if config.model_path is None and config.model_name is None:
        raise ValueError("model_path or model_name must be provided")

    if config.model_path is not None:
        rest_client = service_client.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(config.model_path)
        if config.model_name:
            if config.model_name != training_run.base_model:
                raise ValueError(
                    f"Model name {config.model_name} does not match training run base model {training_run.base_model}"
                )
        else:
            config = chz.replace(config, model_name=training_run.base_model)

    logger.info(f"Using base model: {config.model_name}")

    # Log pass@k and bootstrap configuration
    logger.info(f"Pass@k configuration:")
    logger.info(f"  num_choices: {config.num_choices}")
    logger.info(f"  pass_at_k: {config.pass_at_k}")
    logger.info(f"  bootstrap_stderr: {config.bootstrap_stderr}")
    if config.bootstrap_stderr:
        logger.info(f"  bootstrap_iters: {config.bootstrap_iters}")
        logger.info(f"  bootstrap_seed: {config.bootstrap_seed}")

    sampling_client = service_client.create_sampling_client(
        model_path=config.model_path, base_model=config.model_name
    )

    # Run the evaluation
    logger.info(f"Running bootstrapped inspect evaluation for tasks: {config.tasks}")

    # Create the bootstrapped inspect evaluator
    evaluator = BootstrappedInspectEvaluator(config)
    metrics = await evaluator(sampling_client)

    # Print results
    logger.info("Bootstrapped inspect evaluation completed!")
    logger.info("Results:")
    for metric_name, metric_value in metrics.items():
        if "stderr" in metric_name:
            logger.info(f"  {metric_name}: {metric_value:.6f}")
        else:
            logger.info(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
