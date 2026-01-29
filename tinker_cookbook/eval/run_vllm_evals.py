import asyncio
import logging
import os
import subprocess
import time
from typing import Optional

import chz
from inspect_ai import eval_async
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import get_model

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    """
    Configuration for vLLM-based inspect evaluation.

    This class provides a structured way to configure inspect evaluation
    using models served via vLLM with multi-GPU support.

    Note: Unlike run_inspect_evals.py which uses custom renderers,
    this script uses the model's built-in chat template from HuggingFace.
    The renderer_name parameter is kept for interface compatibility but
    is not used in the same way.
    """

    # Required parameters
    tasks: list[str]
    model_name: str
    renderer_name: str | None = None  # Kept for compatibility, not used

    # vLLM server configuration
    vllm_server_url: str | None = None  # If None, will launch vLLM server
    vllm_host: str = "localhost"
    vllm_port: int = 8000
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism

    # Additional vLLM launch arguments
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"  # "auto", "half", "float16", "bfloat16", "float32"
    trust_remote_code: bool = False

    # Random seed for sampling. If None, sampling is non-deterministic.
    seed: int | None = None
    # If True, logs prompts and responses to the console (useful for debugging).
    verbose: bool = False

    # Generation parameters
    temperature: float = 1.0
    max_tokens: int = 1000
    top_p: float = 1.0
    top_k: int = -1
    num_choices: int = 1

    # Evaluation parameters
    limit: Optional[int] = None
    debug_errors: bool = True
    log_dir: Optional[str] = None
    max_connections: int = 512
    log_level: str = "INFO"


async def launch_vllm_server(config: Config) -> subprocess.Popen:
    """
    Launch a vLLM server with the specified configuration.

    Args:
        config: Configuration object containing vLLM settings

    Returns:
        The subprocess running the vLLM server
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", config.model_name,
        "--host", config.vllm_host,
        "--port", str(config.vllm_port),
        "--tensor-parallel-size", str(config.tensor_parallel_size),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--dtype", config.dtype,
    ]

    if config.max_model_len is not None:
        cmd.extend(["--max-model-len", str(config.max_model_len)])

    if config.trust_remote_code:
        cmd.append("--trust-remote-code")

    logger.info(f"Launching vLLM server with command: {' '.join(cmd)}")

    # Launch vLLM server
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for server to be ready
    server_url = f"http://{config.vllm_host}:{config.vllm_port}"
    max_wait_time = 300  # 5 minutes
    start_time = time.time()

    logger.info(f"Waiting for vLLM server to be ready at {server_url}...")

    while time.time() - start_time < max_wait_time:
        try:
            import requests
            response = requests.get(f"{server_url}/v1/models", timeout=5)
            if response.status_code == 200:
                logger.info("vLLM server is ready!")
                return process
        except Exception:
            pass

        # Check if process has terminated
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"vLLM server process terminated unexpectedly.\n"
                f"Stdout: {stdout}\nStderr: {stderr}"
            )

        await asyncio.sleep(2)

    # Timeout - kill process
    process.kill()
    raise TimeoutError(f"vLLM server did not become ready within {max_wait_time} seconds")


async def main(config: Config):
    logging.basicConfig(level=logging.INFO)

    # Determine server URL
    launched_server = None
    if config.vllm_server_url is None:
        # Launch vLLM server
        logger.info(f"No vLLM server URL provided. Launching vLLM server with {config.tensor_parallel_size} GPUs")
        launched_server = await launch_vllm_server(config)
        server_url = f"http://{config.vllm_host}:{config.vllm_port}/v1"
    else:
        server_url = config.vllm_server_url
        logger.info(f"Using existing vLLM server at: {server_url}")

    try:
        # Create the inspect model using OpenAI-compatible API
        # vLLM exposes an OpenAI-compatible endpoint
        model = get_model(
            f"openai/{config.model_name}",
            base_url=server_url,
            api_key="EMPTY",  # vLLM doesn't require an API key by default
            config=InspectAIGenerateConfig(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k if config.top_k > 0 else None,
                seed=config.seed,
                num_choices=config.num_choices,
            ),
        )

        # Run the evaluation
        logger.info(f"Running inspect evaluation for tasks: {config.tasks}")

        results = await eval_async(
            tasks=config.tasks,
            model=[model],
            limit=config.limit,
            debug_errors=config.debug_errors,
            retry_on_error=3,  # vLLM can handle retries
            fail_on_error=False,
            log_dir=config.log_dir or os.path.expanduser("~/inspect-logs"),
            max_connections=config.max_connections,
            log_level=config.log_level,
            log_realtime=False,
            log_buffer=1000,
        )

        # Extract metrics from results
        metrics = {}
        for task_result in results:
            if task_result.results is not None and task_result.results.scores is not None:
                for task_name, score in task_result.results.scores[0].metrics.items():
                    if task_result.eval.dataset is not None:
                        dataset_name = task_result.eval.dataset.name
                    else:
                        dataset_name = "unknown"
                    metrics[dataset_name + "/" + task_name] = score.value

        # Print results
        logger.info("Inspect evaluation completed!")
        logger.info("Results:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value}")

    finally:
        # Clean up launched server if we started one
        if launched_server is not None:
            logger.info("Shutting down vLLM server")
            launched_server.terminate()
            try:
                launched_server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM server did not terminate gracefully, killing it")
                launched_server.kill()


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
