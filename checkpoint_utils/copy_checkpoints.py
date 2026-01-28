"""
Script to copy Tinker checkpoints without training.

Usage:
    python copy_checkpoints.py \
        source_checkpoint=tinker://4a1939e6-04be-5a77-9e4e-910ccff9f27e:train:0/weights/final \
        new_name=my_copy \
        save_type=both
"""

import asyncio

import chz
import tinker


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

    Args:
        config: Configuration with source checkpoint, new name, and save type

    Returns:
        Dictionary with paths to saved checkpoints

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

    # Load the checkpoint
    print(f"Loading checkpoint from: {config.source_checkpoint}")
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
