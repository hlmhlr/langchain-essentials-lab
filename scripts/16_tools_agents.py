"""Lab section "Tools and Agents": tools, a toolkit and a ReAct agent.

Aim
    Give the model abilities it doesn't have on its own (exact maths, a
    weather lookup) and let an agent decide which tool to use.

What you'll see
    1. A Tool built with the Tool class, called directly.
    2. A tool built with the @tool decorator.
    3. A ReAct agent (verbose) answering several questions: you see its
       Thought / Action / Action Input / Observation lines, then the final answer.

What to notice
    - The tool description is what the agent reads to choose a tool.
    - Maths is done by the tool, not guessed by the model.
    - A small model sometimes breaks the ReAct text format; handle_parsing_errors
      lets the executor ask it to try again instead of crashing.
    - The lab's Python REPL tool runs any code the model writes. It is replaced
      here by a calculator that only accepts arithmetic (see lclab/tools.py).

Run:  uv run python scripts/16_tools_agents.py
"""

import warnings

from langchain_core.tools import Tool

from lclab.llm import get_llm
from lclab.tools import calculator, react_executor, search_weather
from lclab.utils import print_header

# The legacy agent classes print deprecation warnings; they are used on purpose here.
warnings.filterwarnings("ignore", message=".*was deprecated.*")

# 1. Tool class: wrap a plain function as a tool.
print_header("1. Tool class")
calculator_tool = Tool(
    name="calculator",
    func=calculator,
    description="Useful for when you need to perform calculations. Input should be a math expression.",
)
print(calculator_tool.invoke("3 + 1"))
print(calculator_tool.invoke("__import__('os').remove('x')"))  # rejected: not arithmetic

# 2. @tool decorator: the function's docstring becomes the description.
print_header("2. @tool decorator")
print(f"{search_weather.name}: {search_weather.description}")
print(search_weather.invoke("Rome"))

# 3. Toolkit + ReAct agent
print_header("3. ReAct agent")
tools = [calculator_tool, search_weather]  # a toolkit is a list of tools
agent_executor = react_executor(get_llm(), tools, verbose=True)

queries = [
    "What is the square root of 729?",
    "What's 1234 * 56?",
    "What's the weather in Rome?",
    "If it's sunny in Turin, what would be a good outdoor activity?",
]
for query in queries:
    print(f"\n{'=' * 60}\nQUERY: {query}\n{'=' * 60}")
    result = agent_executor.invoke({"input": query})
    print(f"\nFINAL ANSWER: {result['output']}")
