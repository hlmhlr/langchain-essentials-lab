"""Lab sections "Model" and "Chat model".

Aim
    Send a first prompt to the local model, and see what a LangChain chat
    model returns compared with a plain-text helper.

What you'll see
    1. A prompt and its answer as plain text (llm_model).
    2. The same kind of call with llm.invoke(): the type of the result
       (AIMessage), its text (.content) and its token counts.

What to notice
    - The lab's WatsonxLLM returned a string; ChatOllama returns a message
      object, which also carries metadata such as token usage.
    - Token counts matter for cost and for staying within the context window.

Run:  uv run python scripts/01_model.py
"""



from lclab.llm import get_llm, llm_model
from lclab.utils import print_block, print_header

# 1. Model: send a prompt, get text back.
print_header("1. Model: prompt in, text out")
prompt = "Complete this sentence in one short line: The best part of working from home is"
print_block("Prompt", prompt)
print_block("Answer", llm_model(prompt))

# 2. Chat model: the same call through LangChain returns a message object.
print_header("2. Chat model: invoke() returns an AIMessage")
llm = get_llm()
response = llm.invoke("Which animal is often called man's best friend, and why? Answer in two sentences.")

print_block("Type of the response", type(response).__name__)
print_block("response.content", response.content)

# The message also says how many tokens were used, which the lab's
# string-returning WatsonxLLM did not show.
usage = response.usage_metadata or {}
print_block(
    "response.usage_metadata",
    f"input tokens: {usage.get('input_tokens')}, "
    f"output tokens: {usage.get('output_tokens')}, "
    f"total: {usage.get('total_tokens')}",
)
