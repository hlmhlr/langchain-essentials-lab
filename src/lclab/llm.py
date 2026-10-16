"""Connect LangChain to local Ollama models.

This replaces the lab's watsonx.ai setup (`ModelInference`, `WatsonxLLM` and
`WatsonxEmbeddings`). The models are still IBM Granite, but they run on local computer through Ollama instead of IBM's cloud.
"""

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Read settings from the .env file (if it exists) into environment variables.
load_dotenv()

# Model names and the Ollama address. The second argument is the fallback
# used when .env doesn't set the value.
MODEL = os.getenv("OLLAMA_MODEL", "granite4:tiny-h")
SECOND_MODEL = os.getenv("OLLAMA_SECOND_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "granite-embedding:278m")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Same default generation settings as the lab's `parameters` dictionary.
DEFAULT_PARAMS = {
    "max_new_tokens": 256,  # maximum length of the answer, in tokens
    "temperature": 0.2,     # randomness: low = focused, high = creative
}

# Lab parameter name -> Ollama parameter name, for the optional extras.
OPTIONAL_PARAMS = {"top_p": "top_p", "top_k": "top_k", "seed": "seed"}


def get_llm(params: dict | None = None, model: str | None = None) -> ChatOllama:
    """Create a LangChain chat model.

    `params` uses the lab's names and overrides the defaults, for example
    {"temperature": 0.8}. `model` picks another Ollama model (default: MODEL).
    """
    settings = {**DEFAULT_PARAMS, **(params or {})}

    # Pass top_p / top_k / seed only when they were given, so Ollama keeps
    # its own defaults otherwise.
    extras = {
        ollama_name: settings[lab_name]
        for lab_name, ollama_name in OPTIONAL_PARAMS.items()
        if lab_name in settings
    }

    return ChatOllama(
        model=model or MODEL,
        base_url=BASE_URL,
        num_predict=settings["max_new_tokens"],  # Ollama's name for max_new_tokens
        temperature=settings["temperature"],
        **extras,
    )


def get_second_llm(params: dict | None = None) -> ChatOllama:
    """The second model used to compare responses in Exercise 1."""
    return get_llm(params, model=SECOND_MODEL)


def get_embeddings() -> OllamaEmbeddings:
    """Embedding model that turns text into vectors (replaces WatsonxEmbeddings)."""
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=BASE_URL)


def llm_model(prompt_txt: str, params: dict | None = None) -> str:
    """Send one prompt to the main model and return its answer as plain text."""
    response = get_llm(params).invoke(prompt_txt)  # returns an AIMessage
    return response.content                        # keep only the text
