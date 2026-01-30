#!/bin/bash
#SBATCH --job-name=aime_eval_bootstrap
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=hopper-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

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

python -m tinker_cookbook.eval.run_bootstrapped_inspect_evals \
      model_path="tinker://ef8cde95-2474-580d-b1a9-d376db8fb89c:train:0/sampler_weights/sft-r32" \
      tasks=inspect_evals/aime2024 \
      renderer_name="qwen3_instruct" \
      max_tokens=16384 \
      temperature=0.6 \
      top_p=0.95 \
      num_choices=4 \
      pass_at_k=1 \
      bootstrap_stderr=True \
      bootstrap_iters=1000

# python -m tinker_cookbook.eval.run_inspect_evals \
#       model_path="tinker://ef8cde95-2474-580d-b1a9-d376db8fb89c:train:0/sampler_weights/sft-r32" \
#       tasks=inspect_evals/aime2024 \
#       renderer_name="qwen3_instruct" \
#       max_tokens=16384 \
#       temperature=0.6 \
#       top_p=0.95

# python -m tinker_cookbook.eval.run_inspect_evals \
#       model_name="Qwen/Qwen3-8B" \
#       tasks=inspect_evals/aime2024 \
#       renderer_name="qwen3_instruct" \
#       max_tokens=16384 \
#       temperature=0.6 \
#       top_p=0.95

  