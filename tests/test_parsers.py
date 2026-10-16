"""Tests for Exercise 2 and the list parser, using fake models."""

from langchain_core.output_parsers import CommaSeparatedListOutputParser

from lclab.parsers import BOOK_KEYS, PROMPT, book_chain

BOOK_JSON = '{"title": "1984", "author": "George Orwell", "year": 1949, "genre": "Dystopian"}'


def test_prompt_contains_book_name_and_format():
    text = PROMPT.format(book_name="1984")
    assert '"1984"' in text
    assert "RESPONSE FORMAT" in text


def test_book_chain_returns_dict(fake_llm_factory):
    result = book_chain(fake_llm_factory(BOOK_JSON)).invoke({"book_name": "1984"})
    assert isinstance(result, dict)
    assert tuple(result) == BOOK_KEYS
    assert result["year"] == 1949


def test_book_chain_handles_markdown_fences(fake_llm_factory):
    # Small models often wrap JSON in ```json ... ```; the parser strips it.
    fenced = f"```json\n{BOOK_JSON}\n```"
    result = book_chain(fake_llm_factory(fenced)).invoke({"book_name": "1984"})
    assert result["author"] == "George Orwell"


def test_comma_separated_list_parser():
    assert CommaSeparatedListOutputParser().parse("Python, Go, Rust") == ["Python", "Go", "Rust"]
