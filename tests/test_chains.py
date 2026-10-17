"""Tests for the Exercise 6 pipelines, with a fake model (no Ollama)."""

import pytest

from lclab.chains import lcel_pipeline, traditional_pipeline

ANSWERS = ("SENTIMENT: positive", "- Great battery", "Thank you for your review!")

# LLMChain is deprecated but used on purpose (it is the lab's traditional approach).
pytestmark = pytest.mark.filterwarnings("ignore:.*was deprecated")


@pytest.mark.parametrize("build", [traditional_pipeline, lcel_pipeline])
def test_pipeline_returns_all_steps(build, fake_llm_factory):
    result = build(fake_llm_factory(*ANSWERS)).invoke({"review": "Great headphones."})
    assert result["review"] == "Great headphones."
    assert (result["sentiment"], result["summary"], result["response"]) == ANSWERS


def test_lcel_passes_earlier_results_to_later_prompts(fake_llm_factory):
    llm = fake_llm_factory(*ANSWERS)
    prompts_seen = []
    # Record every prompt the model receives.
    llm_with_spy = llm.with_listeners(on_start=lambda run: prompts_seen.append(str(run.inputs)))

    lcel_pipeline(llm_with_spy).invoke({"review": "Great headphones."})

    assert len(prompts_seen) == 3
    assert "SENTIMENT: positive" in prompts_seen[1]      # summary step saw the sentiment
    assert "- Great battery" in prompts_seen[2]          # response step saw the summary
