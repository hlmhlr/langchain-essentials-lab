"""Exercise 1: compare two models at two temperatures on three prompt types.

Aim
    See how the model choice and the temperature setting change the answers
    to a creative, a factual and an instruction-following prompt.

What you'll see
    Granite (granite4:tiny-h) and Llama (llama3.2:3b) answer every prompt
    several times at T=0.8 and T=0.1. A summary table shows, per combination,
    how many runs gave different answers, how similar the runs were, and the
    average time. All answers are saved to outputs/model_comparison.md.

What to notice
    - Low temperature: runs are (nearly) identical, good for facts and formats.
    - High temperature: more variety, good for creative text, but less
      predictable on factual and formatting tasks.
    - The two models differ in style, length and speed for the same prompt.

Run:  uv run python scripts/03_compare_models.py              (3 runs each)
      uv run python scripts/03_compare_models.py --runs 1     (quicker)
"""

import argparse
from pathlib import Path

from lclab.compare import full_report, run_comparison, summary_table
from lclab.data import ROOT
from lclab.llm import MODEL, SECOND_MODEL, get_llm
from lclab.utils import print_header

# Same two temperatures as the lab's solution.
SETTINGS = {
    "creative (T=0.8)": {"temperature": 0.8},
    "precise (T=0.1)": {"temperature": 0.1},
}

# One prompt of each type the lab asks for (my own examples).
PROMPTS = {
    "Creative writing": "Write a four-line poem about a city waking up in the rain.",
    "Factual question": "What is the difference between RAM and storage in a computer? Answer in three sentences.",
    "Instruction-following": "List 5 tips for preparing for a job interview. One short line per tip, numbered.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="answers per prompt and setting")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_comparison.md")
    args = parser.parse_args()

    # Model outer, setting inner: each model is loaded into the GPU once,
    # instead of Ollama swapping between the two models on every call.
    llms = {
        (model, setting): get_llm(params, model=model)
        for model in (MODEL, SECOND_MODEL)
        for setting, params in SETTINGS.items()
    }

    total = len(llms) * len(PROMPTS) * args.runs
    done = 0

    def progress(result, run):
        nonlocal done
        done += 1
        print(f"[{done:>3}/{total}] {result.model:<18} {result.setting:<17} "
              f"{result.prompt_type:<22} run {run + 1}  ({result.seconds[-1]:.1f}s)")

    print_header(f"Running {total} generations")
    results = run_comparison(llms, PROMPTS, runs=args.runs, on_answer=progress)

    print_header("Summary")
    print(summary_table(results))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(full_report(results), encoding="utf-8")
    print(f"\nAll answers saved to {args.output}")


if __name__ == "__main__":
    main()
