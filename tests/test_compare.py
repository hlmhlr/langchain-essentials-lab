"""Tests for lclab.compare (Exercise 1), using fake models."""

import pytest

from lclab.compare import answer_similarity, full_report, run_comparison, summary_table

PROMPTS = {"Creative": "Write a poem.", "Factual": "What is RAM?"}


def test_similarity_of_identical_answers_is_one():
    assert answer_similarity(["same", "same", "same"]) == 1.0


def test_similarity_with_one_answer_is_one():
    assert answer_similarity(["only one"]) == 1.0


def test_similarity_of_different_answers_is_lower():
    assert answer_similarity(["the cat sat", "a dog ran far away"]) < 0.5


def test_run_comparison_collects_every_answer(fake_llm_factory):
    llms = {
        ("model-a", "creative"): fake_llm_factory("p1", "p2", "r1", "r1"),
        ("model-a", "precise"): fake_llm_factory("p", "p", "r", "r"),
    }
    results = run_comparison(llms, PROMPTS, runs=2)

    assert len(results) == 4  # 2 models/settings x 2 prompts
    creative_poem = results[0]
    assert (creative_poem.setting, creative_poem.prompt_type) == ("creative", "Creative")
    assert creative_poem.answers == ["p1", "p2"]
    assert creative_poem.distinct_answers == 2
    assert results[1].distinct_answers == 1  # "r1", "r1"
    assert len(creative_poem.seconds) == 2


def test_progress_callback_is_called_for_each_answer(fake_llm_factory):
    calls = []
    llms = {("m", "s"): fake_llm_factory("a", "b", "c", "d")}
    run_comparison(llms, PROMPTS, runs=2, on_answer=lambda r, i: calls.append(i))
    assert calls == [0, 1, 0, 1]


@pytest.fixture
def two_results(fake_llm_factory):
    llms = {
        ("model-a", "creative"): fake_llm_factory("poem A", "fact A"),
        ("model-b", "creative"): fake_llm_factory("poem B", "fact B"),
    }
    return run_comparison(llms, PROMPTS, runs=1)


def test_summary_table_groups_rows_by_prompt(two_results):
    rows = summary_table(two_results).splitlines()[2:]  # skip header lines
    assert [row.split("|")[1].strip() for row in rows] == [
        "Creative", "Creative", "Factual", "Factual",
    ]


def test_full_report_contains_every_answer(two_results):
    report = full_report(two_results)
    for text in ("poem A", "poem B", "fact A", "fact B", "## Creative: Write a poem."):
        assert text in report
