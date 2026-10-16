"""Exercise 1: compare models and temperatures on the same prompts.

The lab runs each model once per prompt and asks you to observe "variation
between multiple runs". Here every (model, setting, prompt) combination is
run several times, and the variation is measured instead of guessed:
  - distinct answers: how many of the runs gave a different text
  - similarity: average text overlap between every pair of runs (1.0 = identical)
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import combinations

from langchain_core.language_models import BaseChatModel


@dataclass
class RunResult:
    """All answers one model + setting gave to one prompt."""

    model: str
    setting: str
    prompt_type: str
    prompt: str
    answers: list[str] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)

    @property
    def distinct_answers(self) -> int:
        return len({answer.strip() for answer in self.answers})

    @property
    def similarity(self) -> float:
        return answer_similarity(self.answers)

    @property
    def avg_seconds(self) -> float:
        return sum(self.seconds) / len(self.seconds) if self.seconds else 0.0


def answer_similarity(answers: list[str]) -> float:
    """Average similarity (0 to 1) over every pair of answers."""
    pairs = list(combinations(answers, 2))
    if not pairs:  # fewer than 2 answers: nothing to compare
        return 1.0
    total = sum(SequenceMatcher(None, a, b).ratio() for a, b in pairs)
    return total / len(pairs)


def run_comparison(
    llms: dict[tuple[str, str], BaseChatModel],
    prompts: dict[str, str],
    runs: int = 3,
    on_answer: Callable[[RunResult, int], None] | None = None,
) -> list[RunResult]:
    """Send every prompt `runs` times to every (model name, setting) model.

    `on_answer(result, run_index)` is called after each answer, so a script
    can print progress.
    """
    results = []
    for (model_name, setting), llm in llms.items():
        for prompt_type, prompt in prompts.items():
            result = RunResult(model_name, setting, prompt_type, prompt)
            for run in range(runs):
                start = time.perf_counter()
                answer = llm.invoke(prompt).content
                result.seconds.append(time.perf_counter() - start)
                result.answers.append(answer)
                if on_answer:
                    on_answer(result, run)
            results.append(result)
    return results


def _in_prompt_order(results: list[RunResult]) -> list[RunResult]:
    """Group results by prompt, keeping the order the prompts were given in."""
    order = list(dict.fromkeys(r.prompt_type for r in results))
    return sorted(results, key=lambda r: order.index(r.prompt_type))


def summary_table(results: list[RunResult]) -> str:
    """A Markdown table with one row per (prompt, model, setting)."""
    lines = [
        "| Prompt type | Model | Setting | Distinct answers | Similarity | Avg time (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in _in_prompt_order(results):
        lines.append(
            f"| {r.prompt_type} | `{r.model}` | {r.setting} | "
            f"{r.distinct_answers}/{len(r.answers)} | {r.similarity:.2f} | {r.avg_seconds:.1f} |"
        )
    return "\n".join(lines)


def full_report(results: list[RunResult]) -> str:
    """Summary table plus every answer, as a Markdown document."""
    parts = ["# Exercise 1: model and temperature comparison", "", summary_table(results)]

    current_prompt = None
    for r in _in_prompt_order(results):
        if r.prompt_type != current_prompt:
            current_prompt = r.prompt_type
            parts += ["", f"## {r.prompt_type}: {r.prompt}"]
        parts += ["", f"### `{r.model}`, {r.setting}"]
        for i, answer in enumerate(r.answers, start=1):
            parts += ["", f"**Run {i}:**", "", answer.strip()]

    return "\n".join(parts) + "\n"
