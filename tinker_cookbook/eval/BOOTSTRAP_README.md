# Pass@k Evaluation with Bootstrap Standard Error

This module implements pass@k evaluation with bootstrap-based standard error estimation for Tinker Cookbook.

## Overview

The implementation follows the pass@k metric from Chen et al. (2021, arXiv:2107.03374) and replicates lighteval's bootstrap approach for estimating standard error.

### Key Concepts

1. **Pass@k Metric**: Given n generations per problem, pass@k measures the probability that at least one of k randomly sampled generations is correct.

2. **Bootstrap at Problem Level**: Instead of bootstrapping individual generations, we:
   - Compute one pass@k value per problem (from n generations)
   - Bootstrap by resampling problems with replacement
   - Estimate standard error from bootstrap distribution

This approach is more robust for small sample sizes and correctly captures problem-level uncertainty.

## Usage

### Basic Usage

```bash
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
    bootstrap_iters=1000 \
    bootstrap_seed=42
```

### Parameters

- `num_choices`: Number of generations per problem (n in pass@k)
- `pass_at_k`: k value for pass@k metric (default: 1)
- `bootstrap_stderr`: Enable bootstrap stderr estimation (default: True)
- `bootstrap_iters`: Number of bootstrap iterations (default: 1000)
- `bootstrap_seed`: Random seed for reproducibility (default: 42)

### Output Metrics

For each dataset, the evaluator returns:

- `{dataset}/pass@k`: Mean pass@k across all problems
- `{dataset}/pass@k_bootstrap_stderr`: Bootstrap estimate of standard error
- `{dataset}/pass@k_bootstrap_mean`: Bootstrap mean (should match pass@k closely)

Example output:
```
aime2024/pass@1: 0.8500
aime2024/pass@1_bootstrap_stderr: 0.065000
aime2024/pass@1_bootstrap_mean: 0.8510
```

## Implementation Details

### Multi-Pass Evaluation Strategy

Since Inspect AI's `num_choices` parameter is only supported for specific providers (e.g., OpenAI), we use a multi-pass approach:

1. **Run N evaluation passes**: Set `num_choices=N` to run the evaluation N times
2. **Each pass generates 1 completion** per problem (compatible with all providers)
3. **Aggregate by problem ID**: Group the N completions for each problem
4. **Compute pass@k**: Apply pass@k formula to the N completions per problem

This approach is more robust and works with any model provider, including custom Tinker models.

**Performance note**: With `num_choices=64`, the evaluation runs 64 times sequentially. For a dataset with 30 problems, this means 30 × 64 = 1,920 total generations. Plan accordingly for API costs and time.

**Seed handling**: If a seed is specified, each evaluation pass uses `seed + iteration` to ensure different generations while maintaining reproducibility. If no seed is specified, sampling is non-deterministic.

### Pass@k Formula

From Chen et al. (2021):

```
pass@k = 1 - prod(1 - k / (n - c + 1 + i) for i in range(k))
```

Where:
- n = total number of generations
- c = number of correct generations
- k = number of samples to draw

Special cases:
- If n - c < k (fewer incorrect than k), return 1.0
- Formula computes: 1 - P(all k samples are incorrect)

### Bootstrap Algorithm

```python
def bootstrap_stderr(metric_fn, population, num_iterations=1000):
    estimates = []
    for _ in range(num_iterations):
        # Resample problems (not generations) with replacement
        resampled = random.choices(population, k=len(population))
        estimates.append(metric_fn(resampled))
    return std(estimates) / sqrt(num_iterations)
```

### Answer Extraction

The evaluator supports multiple answer formats:

1. **Boxed format**: `\boxed{42}`
2. **Explicit markers**: "ANSWER: 42", "The answer is 42"
3. **Final answer patterns**: "Therefore, the answer is 42"
4. **Fallback**: Last number in text (for simple numeric problems)

If no answer is extracted, the generation is marked as incorrect (score=0).

## Architecture

### Files

1. **bootstrap_utils.py**: Core pass@k and bootstrap implementations
2. **inspect_evaluators.py**:
   - `InspectEvaluatorBuilder`: Configuration builder
   - `InspectEvaluator`: Standard evaluator
   - `BootstrappedInspectEvaluator`: Pass@k evaluator with bootstrap
3. **run_bootstrapped_inspect_evals.py**: Command-line entrypoint

### Design Decisions

1. **Subclass approach**: `BootstrappedInspectEvaluator` extends `InspectEvaluator` for backward compatibility
2. **Per-problem scoring**: Each problem gets one pass@k value (not n individual scores)
3. **Conservative answer extraction**: If we can't extract an answer, mark as incorrect
4. **Automatic selection**: Builder returns `BootstrappedInspectEvaluator` if `pass_at_k` or `bootstrap_stderr` is enabled

## Testing

### Unit Tests

```bash
pytest tinker_cookbook/tests/test_bootstrap_utils.py -v
```

Tests cover:
- Pass@k formula correctness
- Bootstrap reproducibility
- Edge cases (all correct, all incorrect, k=n)

### Integration Testing

For a quick test with low API costs:

```bash
python -m tinker_cookbook.eval.run_bootstrapped_inspect_evals \
    model_path="tinker://your-model-path" \
    tasks=tinker_cookbook.tests.test_bootstrap_inspect_simple:simple_math_task \
    renderer_name="qwen3_instruct" \
    model_name="Qwen/Qwen3-8B-Base" \
    num_choices=3 \
    pass_at_k=1 \
    limit=2 \
    bootstrap_stderr=True
```

## References

- Chen et al., "Evaluating Large Language Models Trained on Code" (2021), arXiv:2107.03374
- Lighteval: https://github.com/huggingface/lighteval (Apache 2.0)
  - Implementation: `src/lighteval/metrics/metrics_sample.py` lines 1263-1327
