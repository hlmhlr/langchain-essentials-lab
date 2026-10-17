"""Exercise 6: product-review pipeline with SequentialChain and with LCEL.

Aim
    Build a three-step pipeline (sentiment -> key points -> reply to the
    customer) both ways, run it on real-looking reviews, and compare the
    two approaches.

What you'll see
    For each review (positive, negative, mixed): the sentiment, the key
    points and the customer reply from each pipeline, plus how long each
    pipeline took.

What to notice
    - Both pipelines send the same prompts, so differences in the text come
      from the model's randomness, not from the approach.
    - Both take about the same time: the cost is the three model calls,
      not the chaining code.
    - The mixed review is the hardest case for the sentiment step.
    The comparison table printed at the end summarises the trade-offs.

Run:  uv run python scripts/15_review_pipeline.py
"""

import time
import warnings

from lclab.chains import lcel_pipeline, traditional_pipeline
from lclab.llm import get_llm
from lclab.utils import print_block, print_header

# The legacy classes print deprecation warnings; they are used on purpose here.
warnings.filterwarnings("ignore", message=".*was deprecated.*")

REVIEWS = {
    "Positive": """These wireless headphones are fantastic. The noise cancelling blocks out the
train completely, the battery easily lasts my whole week of commuting, and they are light
enough to wear all day. Pairing with my phone and laptop was instant.""",
    "Negative": """The robot vacuum was a waste of money. It gets stuck under the sofa every day,
the app disconnects constantly, and after one month the side brush fell off. Customer support
has not answered my two emails.""",
    "Mixed": """The air fryer cooks evenly and the chips come out crispy, which I love. But it is
much louder than I expected and the basket is too small for a family of four, so I often have
to cook in two batches.""",
}

COMPARISON = """
| | SequentialChain (traditional) | LCEL |
|---|---|---|
| How steps connect | output_key names must match the next prompt's variables | `\\|` pipes and RunnablePassthrough.assign |
| Readability | Each step is a separate object; the flow is in a list | The whole flow reads top to bottom in one expression |
| Flexibility | Steps must be LLMChains | Any runnable or Python function can be a step |
| Extras | — | Streaming, batching and async come built in |
| Status | Deprecated, in langchain-classic | Recommended for new code |
"""


def run(name: str, pipeline, review: str) -> None:
    start = time.perf_counter()
    result = pipeline.invoke({"review": review})
    seconds = time.perf_counter() - start

    print(f"\n### {name} ({seconds:.1f}s)")
    print_block("Sentiment", result["sentiment"])
    print_block("Key points", result["summary"])
    print_block("Response", result["response"])


def main() -> None:
    llm = get_llm()
    traditional = traditional_pipeline(llm)
    lcel = lcel_pipeline(llm)

    for label, review in REVIEWS.items():
        print_header(f"{label} review")
        print(" ".join(review.split()))
        run("SequentialChain", traditional, review)
        run("LCEL", lcel, review)

    print_header("Comparison")
    print(COMPARISON)


if __name__ == "__main__":
    main()
