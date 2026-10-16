"""Check that Ollama is running and the three models work.

Run:  uv run python scripts/00_check_setup.py
"""

import json
import sys
import urllib.request

from lclab.llm import BASE_URL, EMBED_MODEL, MODEL, SECOND_MODEL, get_embeddings, get_llm


def installed_models() -> set[str]:
    """Ask Ollama which models are downloaded (GET /api/tags)."""
    with urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=5) as response:
        data = json.load(response)
    return {m["name"] for m in data["models"]}


def main() -> int:
    # 1. Is Ollama reachable?
    try:
        models = installed_models()
    except OSError as error:
        print(f"[FAIL] Cannot reach Ollama at {BASE_URL}: {error}")
        print("       Start the Ollama app, or run 'ollama serve'.")
        return 1
    print(f"[OK]   Ollama is running at {BASE_URL}")

    # 2. Are the three models downloaded?
    missing = [m for m in (MODEL, SECOND_MODEL, EMBED_MODEL) if m not in models]
    for name in missing:
        print(f"[FAIL] Model not found: {name}   ->  run: ollama pull {name}")
    if missing:
        return 1
    print(f"[OK]   Models found: {MODEL}, {SECOND_MODEL}, {EMBED_MODEL}")

    # 3. Does each chat model answer?
    for name in (MODEL, SECOND_MODEL):
        answer = get_llm({"max_new_tokens": 10}, model=name).invoke("Say hello in one word.")
        print(f"[OK]   {name} answered: {answer.content.strip()!r}")

    # 4. Does the embedding model return a vector?
    vector = get_embeddings().embed_query("hello")
    print(f"[OK]   {EMBED_MODEL} returned a vector with {len(vector)} numbers")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
