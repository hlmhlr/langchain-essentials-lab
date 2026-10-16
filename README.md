# LangChain Essentials, running locally on IBM Granite

> Based on IBM's **"Build Smarter AI Apps: Empower LLMs with LangChain"** lab
> (*Develop Generative AI Applications: Get Started*, Coursera), rebuilt to run
> fully locally and free on IBM Granite via Ollama.

Work in progress. The full README comes in the final stage.

```bash
ollama pull granite4:tiny-h
ollama pull llama3.2:3b
ollama pull granite-embedding:278m
uv sync
uv run python scripts/00_check_setup.py
uv run python scripts/00_fetch_data.py
uv run pytest
```
