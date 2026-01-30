import logging
import os
from typing import Optional

import chz
import tinker
from inspect_ai import Tasks, eval_async
from inspect_ai.model import GenerateConfig as InspectAIGenerateConfig
from inspect_ai.model import Model as InspectAIModel
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling

# Set up logger
logger = logging.getLogger(__name__)


@chz.chz
class InspectEvaluatorBuilder:
    """
    Configuration for inspect evaluation.
    This class provides a structured way to configure inspect evaluation
    parameters that can be used both in training configs and evaluator builders.
    """

    # Required parameters
    tasks: Tasks
    renderer_name: str
    # TODO: remove model_name once the SDK adds a get_tokenizer method to sampling client
    model_name: str | None = None
    # Random seed for sampling. If None, sampling is non-deterministic.
    seed: int | None = None
    # If True, logs prompts and responses to the console (useful for debugging).
    verbose: bool = False

    # Generation parameters
    temperature: float = 1.0
    max_tokens: int = 1000
    top_p: float = 1.0
    # Top-k sampling. -1 disables top-k filtering (uses all tokens).
    top_k: int = -1
    # Number of independent responses to generate per prompt.
    # For BootstrappedInspectEvaluator: number of evaluation passes to run
    # (each pass generates 1 completion, so num_choices=64 means 64 total completions).
    # For standard InspectEvaluator: passed to Inspect AI's num_choices parameter
    # (only supported for specific providers like OpenAI).
    num_choices: int = 1

    # Evaluation parameters
    # Maximum number of samples to evaluate. If None, evaluates all samples.
    limit: Optional[int] = None
    debug_errors: bool = True
    log_dir: Optional[str] = None
    # Maximum concurrent sampling requests to Tinker.
    max_connections: int = 512
    log_level: str = "INFO"

    # Pass@k and bootstrap configuration
    # If pass_at_k is set, compute pass@k metric instead of standard accuracy.
    # Requires num_choices >= pass_at_k.
    pass_at_k: int | None = None
    # Enable bootstrap estimation of standard error at the problem level.
    bootstrap_stderr: bool = False
    # Number of bootstrap iterations for stderr estimation.
    bootstrap_iters: int = 1000
    # Random seed for bootstrap (if None, uses non-deterministic sampling).
    bootstrap_seed: int | None = None

    def __call__(self) -> SamplingClientEvaluator:
        if self.bootstrap_stderr or self.pass_at_k is not None:
            return BootstrappedInspectEvaluator(self)
        else:
            return InspectEvaluator(self)


class InspectEvaluator(SamplingClientEvaluator):
    """
    A SamplingClientEvaluator that runs inspect tasks and returns their metrics.
    """

    def __init__(self, config: InspectEvaluatorBuilder):
        """
        Initialize the InspectEvaluator.
        Args:
            config: Configuration object containing all evaluation parameters
        """
        self.config = config

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run inspect evaluation on the given sampling client and return metrics.
        Args:
            sampling_client: The sampling client to evaluate
        Returns:
            Dictionary of metrics from inspect evaluation
        """
        if self.config.model_name is None:
            raise ValueError("model_name must be set before running evaluation")
        # Create the inspect API wrapper
        api = InspectAPIFromTinkerSampling(
            renderer_name=self.config.renderer_name,
            model_name=self.config.model_name,
            sampling_client=sampling_client,
            verbose=self.config.verbose,
        )
        # Create the inspect model
        model = InspectAIModel(
            api=api,
            config=InspectAIGenerateConfig(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                seed=self.config.seed,
                num_choices=self.config.num_choices,
            ),
        )

        # Run evaluation
        results = await eval_async(
            tasks=self.config.tasks,
            model=[model],
            limit=self.config.limit,
            debug_errors=self.config.debug_errors,
            # Never retry - the tinker SDK is doing this for us already
            retry_on_error=0,
            # Although Tinker sampling tries very hard to only throw unrecoverable failures,
            # the inspect evaluation can still fail if e.g. the parser returns an error for
            # a given sample.
            fail_on_error=False,
            log_dir=self.config.log_dir or os.path.expanduser("~/inspect-logs"),
            max_connections=self.config.max_connections,
            log_level=self.config.log_level,
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
                    metrics[dataset_name + "/" + task_name] = score.value  # pyright: ignore[reportOptionalOperand]

        logger.info(f"Inspect evaluation completed. Metrics: {metrics}")
        return metrics


class BootstrappedInspectEvaluator(InspectEvaluator):
    """
    InspectEvaluator that computes per-problem pass@k with bootstrap stderr.

    This evaluator extends InspectEvaluator to:
    1. Generate n completions per problem (via num_choices)
    2. Compute pass@k for each problem (one value per problem)
    3. Bootstrap at the problem level to estimate standard error

    The bootstrap resamples problems (not individual generations), making it
    appropriate for estimating uncertainty in the mean pass@k across problems.
    """

    def __init__(self, config: InspectEvaluatorBuilder):
        """
        Initialize the BootstrappedInspectEvaluator.

        Args:
            config: Configuration with pass_at_k and bootstrap settings
        """
        super().__init__(config)

        # Validate configuration
        if self.config.pass_at_k is not None:
            if self.config.pass_at_k < 1:
                raise ValueError(f"pass_at_k must be >= 1, got {self.config.pass_at_k}")
            if self.config.pass_at_k > self.config.num_choices:
                raise ValueError(
                    f"pass_at_k ({self.config.pass_at_k}) cannot be greater than "
                    f"num_choices ({self.config.num_choices})"
                )

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run evaluation with pass@k and bootstrap stderr computation.

        Since Inspect AI's num_choices parameter is only supported for specific providers,
        we run the evaluation multiple times (num_choices iterations) and aggregate
        the results by problem to compute pass@k.

        Args:
            sampling_client: The sampling client to evaluate

        Returns:
            Dictionary of metrics including pass@k and bootstrap stderr
        """
        import re

        from tinker_cookbook.eval.bootstrap_utils import (
            bootstrap_mean,
            bootstrap_stderr,
            mean_metric,
            pass_at_k,
        )

        if self.config.model_name is None:
            raise ValueError("model_name must be set before running evaluation")

        logger.info(
            f"Running {self.config.num_choices} evaluation passes to generate "
            f"{self.config.num_choices} completions per problem"
        )

        # Run evaluation num_choices times to get multiple generations per problem
        all_results = []
        for iteration in range(self.config.num_choices):
            logger.info(f"Starting evaluation pass {iteration + 1}/{self.config.num_choices}")

            # Create the inspect API wrapper
            api = InspectAPIFromTinkerSampling(
                renderer_name=self.config.renderer_name,
                model_name=self.config.model_name,
                sampling_client=sampling_client,
                verbose=self.config.verbose,
            )

            # Create the inspect model with num_choices=1 for each pass
            # We'll aggregate results ourselves
            model = InspectAIModel(
                api=api,
                config=InspectAIGenerateConfig(
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    seed=self.config.seed if self.config.seed is None else self.config.seed + iteration,
                    num_choices=1,  # Always use 1 for compatibility
                ),
            )

            # Run evaluation
            results = await eval_async(
                tasks=self.config.tasks,
                model=[model],
                limit=self.config.limit,
                debug_errors=self.config.debug_errors,
                retry_on_error=0,
                fail_on_error=False,
                log_dir=self.config.log_dir or os.path.expanduser("~/inspect-logs"),
                max_connections=self.config.max_connections,
                log_level=self.config.log_level,
                log_realtime=False,
                log_buffer=1000,
            )

            all_results.append(results)

        logger.info(f"Completed all {self.config.num_choices} evaluation passes")

        # Extract standard Inspect metrics from first pass (if any)
        metrics = {}
        first_results = all_results[0]
        for task_result in first_results:
            if task_result.results is not None and task_result.results.scores is not None:
                for task_name, score in task_result.results.scores[0].metrics.items():
                    if task_result.eval.dataset is not None:
                        dataset_name = task_result.eval.dataset.name
                    else:
                        dataset_name = "unknown"
                    # Only include standard metrics from first pass
                    # (pass@k metrics will be computed separately)
                    if "pass@" not in task_name:
                        metrics[dataset_name + "/" + task_name] = score.value  # pyright: ignore[reportOptionalOperand]

        # Compute pass@k metrics with bootstrap by aggregating across all passes
        for task_idx, task_result in enumerate(first_results):
            if task_result.eval.dataset is not None:
                dataset_name = task_result.eval.dataset.name
            else:
                dataset_name = "unknown"

            # Aggregate results from all passes for this task
            aggregated_results = [results[task_idx] for results in all_results]

            # Compute per-problem pass@k scores
            problem_passatk_scores = self._compute_per_problem_passatk_multi_pass(
                aggregated_results
            )

            if not problem_passatk_scores:
                logger.warning(f"No valid samples for {dataset_name}, skipping pass@k metrics")
                continue

            k = self.config.pass_at_k if self.config.pass_at_k is not None else 1
            metric_name = f"pass@{k}"

            # Mean pass@k across problems
            mean_passatk = mean_metric(problem_passatk_scores)
            metrics[f"{dataset_name}/{metric_name}"] = mean_passatk

            # Bootstrap stderr (if enabled)
            if self.config.bootstrap_stderr and len(problem_passatk_scores) >= 2:
                stderr = bootstrap_stderr(
                    mean_metric,
                    problem_passatk_scores,
                    num_iterations=self.config.bootstrap_iters,
                    seed=self.config.bootstrap_seed,
                )
                metrics[f"{dataset_name}/{metric_name}_bootstrap_stderr"] = stderr

                # Also compute bootstrap mean for validation
                boot_mean = bootstrap_mean(
                    mean_metric,
                    problem_passatk_scores,
                    num_iterations=self.config.bootstrap_iters,
                    seed=self.config.bootstrap_seed,
                )
                metrics[f"{dataset_name}/{metric_name}_bootstrap_mean"] = boot_mean

                logger.info(
                    f"{dataset_name}/{metric_name}: {mean_passatk:.4f} ± {stderr:.4f} "
                    f"(bootstrap mean: {boot_mean:.4f}, n={len(problem_passatk_scores)})"
                )
            else:
                logger.info(
                    f"{dataset_name}/{metric_name}: {mean_passatk:.4f} "
                    f"(n={len(problem_passatk_scores)})"
                )

        logger.info(f"Bootstrapped evaluation completed. Metrics: {metrics}")
        return metrics

    def _compute_per_problem_passatk_multi_pass(self, task_results: list) -> list[float]:
        """
        Compute pass@k for each problem by aggregating results from multiple evaluation passes.

        This method handles the case where we run evaluation multiple times (num_choices passes)
        to get multiple generations per problem, since Inspect AI's num_choices parameter
        is not supported for all providers.

        Args:
            task_results: List of EvalLog objects, one per evaluation pass

        Returns:
            List of pass@k values (one per problem)
        """
        from collections import defaultdict

        from tinker_cookbook.eval.bootstrap_utils import pass_at_k

        # Group generations by problem ID
        # problem_generations[problem_id] = [(text, target), ...]
        problem_generations: dict[str | int, list[tuple[str, list[str]]]] = defaultdict(list)

        # Collect all generations from all passes
        for pass_idx, task_result in enumerate(task_results):
            for sample in task_result.samples:
                # Skip samples with no output
                if sample.output is None or sample.output.choices is None:
                    logger.debug(
                        f"Pass {pass_idx}, sample {sample.id}: no output or choices, skipping"
                    )
                    continue

                if len(sample.output.choices) == 0:
                    logger.debug(f"Pass {pass_idx}, sample {sample.id}: no choices generated")
                    continue

                # Get target answer(s)
                if isinstance(sample.target, list):
                    target_answers = [str(t).strip() for t in sample.target]
                else:
                    target_answers = [str(sample.target).strip()]

                # Extract the generated text from first choice (we set num_choices=1)
                choice = sample.output.choices[0]
                if choice.message is None or choice.message.content is None:
                    logger.debug(
                        f"Pass {pass_idx}, sample {sample.id}: no message content, skipping"
                    )
                    continue

                generated_text = (
                    choice.message.content
                    if isinstance(choice.message.content, str)
                    else str(choice.message.content)
                )

                # Store generation with its target
                problem_generations[sample.id].append((generated_text, target_answers))

        # Compute pass@k for each problem
        problem_passatk_scores = []
        k = self.config.pass_at_k if self.config.pass_at_k is not None else 1

        for problem_id, generations in problem_generations.items():
            # Score each generation for this problem
            generation_scores = []
            for generated_text, target_answers in generations:
                predicted_answer = self._extract_answer(generated_text)

                # Check correctness
                is_correct = predicted_answer is not None and any(
                    predicted_answer == target for target in target_answers
                )
                generation_scores.append(1 if is_correct else 0)

                if not is_correct and predicted_answer is not None:
                    logger.debug(
                        f"Problem {problem_id}: predicted='{predicted_answer}', "
                        f"target={target_answers}, marked incorrect"
                    )
                elif not is_correct:
                    logger.debug(
                        f"Problem {problem_id}: could not extract answer, marked incorrect"
                    )

            # Compute pass@k for this problem
            n = len(generation_scores)
            if n == 0:
                logger.warning(f"Problem {problem_id}: no valid generations, skipping")
                continue

            problem_score = pass_at_k(generation_scores, k=k, n=n)
            problem_passatk_scores.append(problem_score)

            logger.debug(
                f"Problem {problem_id}: {sum(generation_scores)}/{n} correct, "
                f"pass@{k}={problem_score:.3f}"
            )

        return problem_passatk_scores

    def _compute_per_problem_passatk(self, task_result) -> list[float]:
        """
        Compute pass@k for each problem in the task result.

        For each problem:
        1. Extract n generations from sample.output.choices
        2. Score each generation (correct=1, incorrect=0)
        3. Apply pass@k formula to get a single value per problem

        Args:
            task_result: EvalLog from inspect_ai.eval_async

        Returns:
            List of pass@k values (one per problem)
        """
        import re

        from tinker_cookbook.eval.bootstrap_utils import pass_at_k

        problem_passatk_scores = []

        for sample in task_result.samples:
            # Skip samples with no output
            if sample.output is None or sample.output.choices is None:
                logger.info(
                    f"Skipping sample {sample.id}: no output or choices "
                    f"(error: {sample.error})"
                )
                continue

            # Extract all n generations for this problem
            num_generations = len(sample.output.choices)
            if num_generations == 0:
                logger.info(f"Skipping sample {sample.id}: no choices generated")
                continue

            # Get target answer
            if isinstance(sample.target, list):
                # Multiple acceptable answers
                target_answers = [str(t).strip() for t in sample.target]
            else:
                target_answers = [str(sample.target).strip()]

            # Score each generation
            generation_scores = []
            for i, choice in enumerate(sample.output.choices):
                if choice.message is None or choice.message.content is None:
                    logger.info(
                        f"Sample {sample.id}, choice {i}: no message content, marking as incorrect"
                    )
                    generation_scores.append(0)
                    continue

                # Extract answer from the generated text
                generated_text = (
                    choice.message.content
                    if isinstance(choice.message.content, str)
                    else str(choice.message.content)
                )
                predicted_answer = self._extract_answer(generated_text)

                # Check correctness
                is_correct = predicted_answer is not None and any(
                    predicted_answer == target for target in target_answers
                )
                generation_scores.append(1 if is_correct else 0)

                if not is_correct and predicted_answer is not None:
                    logger.debug(
                        f"Sample {sample.id}, choice {i}: predicted='{predicted_answer}', "
                        f"target={target_answers}, marked incorrect"
                    )
                elif not is_correct:
                    logger.info(
                        f"Sample {sample.id}, choice {i}: could not extract answer, "
                        f"marked incorrect"
                    )

            # Compute pass@k for this problem
            k = self.config.pass_at_k if self.config.pass_at_k is not None else 1
            n = len(generation_scores)
            problem_score = pass_at_k(generation_scores, k=k, n=n)
            problem_passatk_scores.append(problem_score)

            logger.debug(
                f"Sample {sample.id}: {sum(generation_scores)}/{n} correct, "
                f"pass@{k}={problem_score:.3f}"
            )

        return problem_passatk_scores

    def _extract_answer(self, text: str) -> str | None:
        """
        Extract answer from model output.

        Supports common answer formats:
        - Boxed format: \\boxed{42}
        - Explicit markers: "ANSWER: 42", "The answer is 42"
        - Final answer patterns
        - Last number on last line (fallback for simple problems)

        Args:
            text: Generated text from model

        Returns:
            Extracted answer string or None if not found
        """
        import re

        # Try boxed format first (LaTeX)
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed_match:
            answer = boxed_match.group(1).strip()
            # Remove LaTeX formatting
            answer = answer.replace("$", "").strip()
            return answer

        # Try common answer patterns
        patterns = [
            r"ANSWER:\s*(.+?)(?:\n|$)",
            r"(?:the\s+)?answer\s+is:?\s*(.+?)(?:\n|$)",
            r"(?:the\s+)?final\s+answer\s+is:?\s*(.+?)(?:\n|$)",
            r"(?:therefore|thus|so),?\s+(?:the\s+)?answer\s+is:?\s*(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Remove common trailing punctuation
                answer = answer.rstrip(".,;:")
                # Remove dollar signs (LaTeX)
                answer = answer.replace("$", "").strip()
                return answer

        # Fallback: try to find last number in text (for simple numeric answers)
        # This helps with cases like "2 + 2 = 4" or "The result is 15."
        numbers = re.findall(r"\b\d+\b", text)
        if numbers:
            # Return the last number found
            return numbers[-1]

        # If no answer found, return None
        return None
