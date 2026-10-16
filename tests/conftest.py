"""Shared test fixtures: fake models, so tests run without Ollama.

pytest loads this file automatically and makes every fixture below
available to any test that names it as an argument.
"""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


def make_fake_llm(*answers: str) -> GenericFakeChatModel:
    """A chat model that returns the given answers, in order, one per call."""
    return GenericFakeChatModel(messages=iter([AIMessage(content=a) for a in answers]))


@pytest.fixture
def fake_llm_factory():
    """Lets a test build a fake model with its own scripted answers."""
    return make_fake_llm


@pytest.fixture
def fake_embeddings():
    """Embeddings that give the same vector for the same text, every time."""
    return DeterministicFakeEmbedding(size=16)
