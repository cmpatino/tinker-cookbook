"""
Script to merge LoRA adapters with base models and optionally push to HuggingFace Hub.

This script loads a base model, applies LoRA adapter weights, merges them into a single
model, and can save locally or push to the HuggingFace Hub.

Usage:
    # Merge and save locally
    python merge_adapters.py \
        base_model=meta-llama/Llama-3.1-8B \
        adapter_path=path/to/adapter \
        output_path=./merged_model

    # Merge and push to HuggingFace Hub
    python merge_adapters.py \
        base_model=meta-llama/Llama-3.1-8B \
        adapter_path=username/adapter-repo \
        output_path=username/merged-model \
        push_to_hub=True

    # Load adapter from local path
    python merge_adapters.py \
        base_model=meta-llama/Llama-3.1-8B \
        adapter_path=/local/path/to/adapter \
        output_path=./merged_model
"""

import logging
import os

import chz
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@chz.chz
class MergeConfig:
    """Configuration for merging LoRA adapters with base models."""

    # Required parameters
    base_model: str  # HuggingFace model ID or local path
    adapter_path: str  # HuggingFace adapter ID or local path to adapter
    output_path: str  # Local path or HuggingFace repo ID for merged model

    # Optional parameters
    push_to_hub: bool = False  # Whether to push merged model to HuggingFace Hub
    torch_dtype: str = "bfloat16"  # Data type for model weights
    device_map: str = "auto"  # Device mapping strategy
    trust_remote_code: bool = False  # Whether to trust remote code
    hf_token: str | None = None  # HuggingFace token for private repos/pushing

    # Advanced options
    max_memory: dict[int, str] | None = None  # Max memory per device (e.g., {0: "20GB"})
    low_cpu_mem_usage: bool = True  # Use less CPU memory during loading


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float": torch.float32,
        "half": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_str.lower() not in dtype_map:
        raise ValueError(
            f"Invalid torch_dtype: {dtype_str}. "
            f"Must be one of: {list(dtype_map.keys())}"
        )
    return dtype_map[dtype_str.lower()]


def merge_adapters(config: MergeConfig) -> str:
    """
    Merge LoRA adapter with base model and save/push the result.

    Args:
        config: Configuration for merging

    Returns:
        Path or URL to the saved/pushed model

    Example:
        >>> config = MergeConfig(
        ...     base_model="meta-llama/Llama-3.1-8B",
        ...     adapter_path="username/my-lora-adapter",
        ...     output_path="./merged_model"
        ... )
        >>> output = merge_adapters(config)
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "The 'peft' library is required for merging LoRA adapters. "
            "Install it with: pip install peft"
        )

    torch_dtype = get_torch_dtype(config.torch_dtype)
    token = config.hf_token or os.environ.get("HF_TOKEN")

    # Load base model
    logger.info(f"Loading base model: {config.base_model}")
    logger.info(f"  Using dtype: {torch_dtype}")
    logger.info(f"  Device map: {config.device_map}")

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch_dtype,
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
        low_cpu_mem_usage=config.low_cpu_mem_usage,
        max_memory=config.max_memory,
        token=token,
    )

    # Load LoRA adapter
    logger.info(f"Loading LoRA adapter from: {config.adapter_path}")
    model = PeftModel.from_pretrained(
        model,
        config.adapter_path,
        torch_dtype=torch_dtype,
        token=token,
    )

    # Merge adapter into base model
    logger.info("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=config.trust_remote_code,
        token=token,
    )

    # Save or push to hub
    if config.push_to_hub:
        logger.info(f"Pushing merged model to HuggingFace Hub: {config.output_path}")
        model.push_to_hub(
            config.output_path,
            token=token,
            private=False,  # Make public by default, change if needed
        )
        tokenizer.push_to_hub(
            config.output_path,
            token=token,
            private=False,
        )
        result = f"https://huggingface.co/{config.output_path}"
        logger.info(f"✓ Model pushed to: {result}")
    else:
        logger.info(f"Saving merged model locally to: {config.output_path}")
        os.makedirs(config.output_path, exist_ok=True)
        model.save_pretrained(config.output_path)
        tokenizer.save_pretrained(config.output_path)
        result = config.output_path
        logger.info(f"✓ Model saved to: {result}")

    return result


def main(config: MergeConfig):
    """Main entry point for the script."""
    logger.info("Starting LoRA adapter merge process")
    logger.info(f"Configuration:")
    logger.info(f"  Base model: {config.base_model}")
    logger.info(f"  Adapter path: {config.adapter_path}")
    logger.info(f"  Output path: {config.output_path}")
    logger.info(f"  Push to hub: {config.push_to_hub}")
    logger.info("")

    result = merge_adapters(config)

    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ Merge completed successfully!")
    logger.info(f"  Result: {result}")
    logger.info("=" * 60)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
