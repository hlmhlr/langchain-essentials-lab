"""Tools and agents (Tools and Agents section, Exercise 7).

Purpose
    The tools the agents can call, and the two ways of building an agent:
      - the lab's text-based ReAct agent (create_react_agent + AgentExecutor,
        legacy, from langchain-classic): the model writes "Action: ..." lines
        and LangChain parses the text
      - the current tool-calling agent (create_agent): the model returns
        structured tool calls with typed arguments, so no text parsing

Safety change from the lab
    The lab runs model-written code with PythonREPL and eval(). Both can run
    any Python, e.g. delete files. Here the calculator only evaluates
    arithmetic: the expression is parsed with `ast` and anything that is not
    a number, an operator or an allowed math function is rejected.

Contents
    safe_eval()            arithmetic only: + - * / // % **, sqrt, abs, round
    calculator()           the lab's calculator, using safe_eval
    format_text()          the lab's text formatter ("uppercase: hello")
    search_weather         the lab's fake weather tool
    react_executor()       ReAct agent + AgentExecutor (legacy)
    TOOLS_TYPED            the same tools with typed arguments, for create_agent
    tool_calls_made()      which tools a create_agent run called
"""

import ast
import math
import operator
from typing import Literal

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, tool

# --- Safe calculator -------------------------------------------------------

_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCTIONS = {"sqrt": math.sqrt, "abs": abs, "round": round}


def safe_eval(expression: str) -> float | int:
    """Evaluate an arithmetic expression without running arbitrary code."""

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCTIONS and not node.keywords):
            return _FUNCTIONS[node.func.id](*[evaluate(arg) for arg in node.args])
        raise ValueError(f"not allowed: {ast.unparse(node)}")

    # Small models sometimes wrap the input in quotes or backticks.
    cleaned = expression.strip().strip("`'\"").replace("^", "**")
    return evaluate(ast.parse(cleaned, mode="eval").body)


def calculator(expression: str) -> str:
    """A simple calculator. Input should be a math expression like '2 + 2' or 'sqrt(144)'."""
    try:
        return f"Result: {safe_eval(expression)}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


# --- Text formatter (the lab's solution) -----------------------------------

def format_text(text: str) -> str:
    """Format text to uppercase, lowercase, or title case.
    Input should be in format: '[format_type]: [text]'
    where format_type is 'uppercase', 'lowercase', or 'titlecase'."""
    try:
        if ":" not in text:
            return f"Missing format. Example: titlecase: {text} -> {text.title()}"
        format_type, content = text.split(":", 1)
        format_type, content = format_type.strip().lower(), content.strip()

        if format_type == "uppercase":
            return content.upper()
        if format_type == "lowercase":
            return content.lower()
        if format_type == "titlecase":
            return content.title()
        return f"Unknown format {format_type}. Use: uppercase, lowercase, or titlecase"
    except Exception as e:
        return f"Error formatting text: {str(e)}"


@tool
def search_weather(location: str) -> str:
    """Search for the current weather in the specified location."""
    # In a real application, this would call a weather API
    return f"The weather in {location} is currently sunny and 22°C."


# --- Legacy: text-based ReAct agent ----------------------------------------

REACT_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant who can use tools to help with simple tasks.
You have access to these tools:

{tools}

The available tools are: {tool_names}

Follow this format:

Question: the user's question
Thought: think about what to do
Action: the tool to use, should be one of [{tool_names}]
Action Input: the input to the tool
Observation: the result from the tool
Thought: I now know the final answer
Final Answer: your final answer to the user's question

Question: {input}
{agent_scratchpad}
"""
)


def react_executor(llm: BaseChatModel, tools: list[BaseTool], prompt: PromptTemplate = REACT_PROMPT,
                   verbose: bool = False) -> AgentExecutor:
    """The lab's create_react_agent + AgentExecutor.

    return_intermediate_steps=True keeps the (tool, input, result) of every step;
    max_iterations stops a confused small model from looping forever.
    """
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=verbose,
        handle_parsing_errors=True, return_intermediate_steps=True, max_iterations=5,
    )


# --- Current: tool-calling agent with typed arguments -----------------------

@tool("calculator")
def calculator_typed(expression: str) -> str:
    """Calculate a math expression like '25 + 63', '15 * 7' or 'sqrt(144)'."""
    return calculator(expression)


@tool("format_text")
def format_text_typed(style: Literal["uppercase", "lowercase", "titlecase"], text: str) -> str:
    """Change the letter case of a text."""
    # The style is a separate, checked argument: no "style: text" string to parse.
    return format_text(f"{style}: {text}")


TOOLS_TYPED = [calculator_typed, format_text_typed, search_weather]


def tool_calls_made(messages) -> list[str]:
    """'tool(args)' for every tool call in a create_agent result."""
    return [
        f"{call['name']}({call['args']})"
        for message in messages
        for call in getattr(message, "tool_calls", None) or []
    ]
