"""
Script to copy Tinker checkpoints without training.

Supports copying both state checkpoints (/weights/) and sampler checkpoints (/sampler_weights/).
For sampler checkpoints, the script will attempt to load from the corresponding state checkpoint
at the same path (e.g., /weights/000080 for /sampler_weights/000080).

Usage:
    # Copy a state checkpoint
    python copy_checkpoints.py \
        source_checkpoint=tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/weights/final \
        new_name=my_copy \
        save_type=both

    # Copy a sampler checkpoint (requires corresponding state checkpoint)
    python copy_checkpoints.py \
        source_checkpoint=tinker://run-id:train:0/sampler_weights/000080 \
        new_name=my_copy \
        save_type=sampler
"""

import asyncio

import chz
import tinker
from dotenv import load_dotenv

load_dotenv()


@chz.chz
class CopyConfig:
    """Configuration for copying checkpoints."""

    source_checkpoint: str
    new_name: str
    save_type: str = "both"  # "state", "sampler", or "both"
    base_url: str | None = None


async def copy_checkpoint(config: CopyConfig) -> dict[str, str]:
    """
    Load a checkpoint and save a copy with a new name.

    Supports both state checkpoints (/weights/) and sampler checkpoints (/sampler_weights/).
    For sampler checkpoints, the script will attempt to load from the corresponding state
    checkpoint (e.g., /weights/000080 for /sampler_weights/000080).

    Args:
        config: Configuration with source checkpoint, new name, and save type

    Returns:
        Dictionary with paths to saved checkpoints

    Raises:
        ValueError: If save_type is invalid or if copying a sampler checkpoint fails

    Example:
        >>> config = CopyConfig(
        ...     source_checkpoint="tinker://...",
        ...     new_name="my_copy",
        ...     save_type="both"
        ... )
        >>> results = await copy_checkpoint(config)
        >>> print(results["state_path"])
    """
    if config.save_type not in ["state", "sampler", "both"]:
        raise ValueError(f"save_type must be 'state', 'sampler', or 'both', got: {config.save_type}")

    service_client = tinker.ServiceClient(base_url=config.base_url)

    # Detect checkpoint type from path
    is_sampler_checkpoint = "/sampler_weights/" in config.source_checkpoint
    checkpoint_type = "sampler checkpoint" if is_sampler_checkpoint else "state checkpoint"

    # Load the checkpoint
    print(f"Loading checkpoint from: {config.source_checkpoint}")
    print(f"  Detected {checkpoint_type}")

    # Handle sampler checkpoints by trying to load from corresponding state checkpoint
    if is_sampler_checkpoint:
        # Try to find corresponding state checkpoint
        state_path = config.source_checkpoint.replace("/sampler_weights/", "/weights/")
        print(f"  Attempting to load from corresponding state checkpoint: {state_path}")

        try:
            training_client = await service_client.create_training_client_from_state_async(state_path)
            print(f"  ✓ Successfully loaded from state checkpoint")

            # Warn if user is trying to save as state when source was sampler
            if config.save_type == "state":
                print(f"  ⚠️  Note: Loading from state checkpoint, but source was sampler checkpoint")
        except Exception as e:
            # Provide helpful error message
            error_msg = (
                f"\nError: Cannot copy sampler-only checkpoint: {config.source_checkpoint}\n"
                f"Attempted to find corresponding state checkpoint at: {state_path}\n"
                f"API Error: {e}\n\n"
                f"Sampler checkpoints cannot be copied directly via the TrainingClient API.\n\n"
                f"Options:\n"
                f"  1. If a state checkpoint exists at the same step, copy it instead:\n"
                f"     python copy_checkpoints.py source_checkpoint={state_path} ...\n"
                f"  2. Download and re-upload the weights (see download_weights.py for examples)\n"
            )
            raise ValueError(error_msg) from e
    else:
        # Regular state checkpoint
        training_client = await service_client.create_training_client_from_state_async(
            config.source_checkpoint
        )

    # Save the checkpoint(s)
    results = {}
    if config.save_type in ["state", "both"]:
        print(f"Saving state checkpoint as: {config.new_name}")
        state_result = await training_client.save_state_async(config.new_name)
        state_path = (await state_result.result_async()).path
        results["state_path"] = state_path
        print(f"  State saved to: {state_path}")

    if config.save_type in ["sampler", "both"]:
        print(f"Saving sampler checkpoint as: {config.new_name}")
        sampler_result = await training_client.save_weights_for_sampler_async(config.new_name)
        sampler_path = (await sampler_result.result_async()).path
        results["sampler_path"] = sampler_path
        print(f"  Sampler saved to: {sampler_path}")

    return results


def main(config: CopyConfig):
    """Main entry point for the script."""
    results = asyncio.run(copy_checkpoint(config))
    print("\n✓ Checkpoint copied successfully!")
    print("Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
