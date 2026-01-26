#!/bin/bash
#SBATCH --job-name=8b_to_4b
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=hopper-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo "Missing .env file in $(pwd)" >&2
  exit 1
fi
source .venv/bin/activate

# python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
#         model_name=Qwen/Qwen3-4B-Instruct-2507 \
#         teacher_model=Qwen/Qwen3-8B \
#         dataset=deepmath_2.5k \
#         learning_rate=1e-4 \
#         group_size=4 \
#         groups_per_batch=1024 \
#         wandb_project=kd-thinky \
#         wandb_name=thinky-4b-to-8b

python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
        model_name=Qwen/Qwen3-4B-Instruct-2507 \
        teacher_model=Qwen/Qwen3-8B \
        dataset=deepmath_62k \
        learning_rate=1e-4 \
        lora_rank=32 \
        group_size=1 \
        groups_per_batch=1024 \
        wandb_project=kd-thinky \
        wandb_name='v04.00_(TM)' \
        save_every=10