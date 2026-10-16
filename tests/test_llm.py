"""Tests for lclab.llm: settings are mapped correctly. No Ollama needed."""

from lclab import llm


def test_defaults_match_the_lab():
    model = llm.get_llm()
    assert model.model == llm.MODEL
    assert model.num_predict == 256
    assert model.temperature == 0.2
    assert model.top_k is None  # not set unless asked for


def test_params_override_defaults():
    model = llm.get_llm({"temperature": 0.8, "max_new_tokens": 50, "top_k": 1})
    assert model.temperature == 0.8
    assert model.num_predict == 50
    assert model.top_k == 1


def test_second_model_uses_its_own_name():
    assert llm.get_second_llm().model == llm.SECOND_MODEL


def test_embeddings_use_embed_model():
    assert llm.get_embeddings().model == llm.EMBED_MODEL


def test_llm_model_returns_plain_text(monkeypatch, fake_llm_factory):
    # Replace the real model with a fake one for this test only.
    monkeypatch.setattr(llm, "get_llm", lambda params=None: fake_llm_factory("Paris"))
    assert llm.llm_model("Capital of France?") == "Paris"


def test_fake_embeddings_are_repeatable(fake_embeddings):
    assert fake_embeddings.embed_query("cat") == fake_embeddings.embed_query("cat")
    assert len(fake_embeddings.embed_query("cat")) == 16
