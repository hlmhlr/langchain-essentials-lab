"""Exercise 7: an agent with a calculator and a text formatter, two ways.

Aim
    Build the exercise's agent with the lab's text-based ReAct approach and
    with the current tool-calling approach, and compare them on the same
    questions.

What you'll see
    For each question, from each agent: the tool calls it made (tool and
    input) and its final answer.

What to notice
    - ReAct: the model must write "Action: format_text / Action Input:
      uppercase: hello world" exactly right. The lab itself warns that
      "convert 'hello world' to uppercase" may fail because of the input format.
    - Tool calling: format_text takes two typed arguments (style, text), and
      style can only be uppercase / lowercase / titlecase, so there is no
      string format for the model to get wrong.
    - Both agents run on the same local model; the difference is only in
      how tool use is expressed.

Run:  uv run python scripts/17_agent_exercise.py
"""

import warnings

from langchain.agents import create_agent
from langchain_core.tools import Tool

from lclab.llm import get_llm
from lclab.tools import TOOLS_TYPED, calculator, format_text, react_executor, tool_calls_made
from lclab.utils import print_header

warnings.filterwarnings("ignore", message=".*was deprecated.*")

# Tools for the ReAct agent: plain functions wrapped with the Tool class (as in the lab)
react_tools = [
    Tool(name="calculator", func=calculator,
         description="Useful for performing simple math calculations. Input: a math expression."),
    Tool(name="format_text", func=format_text,
         description="Useful for formatting text. Input format: 'uppercase: text', "
                     "'lowercase: text' or 'titlecase: text'."),
]

test_questions = [
    "What is 48 + 175?",
    "Can you convert 'good morning milano' to uppercase?",
    "Calculate 23 * 19",
    "titlecase: the quick brown fox",
    "What is 12.5% of 640?",
]


def main() -> None:
    llm = get_llm()
    react_agent = react_executor(llm, react_tools)
    modern_agent = create_agent(llm, tools=TOOLS_TYPED[:2],
                                system_prompt="Use the tools for calculations and text formatting.")

    for question in test_questions:
        print_header(question)

        result = react_agent.invoke({"input": question})
        steps = [f"{action.tool}({action.tool_input!r})" for action, _ in result["intermediate_steps"]]
        print(f"ReAct        tools: {steps or 'none'}")
        print(f"             answer: {result['output']}")

        result = modern_agent.invoke({"messages": [{"role": "user", "content": question}]})
        print(f"Tool calling tools: {tool_calls_made(result['messages']) or 'none'}")
        print(f"             answer: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
