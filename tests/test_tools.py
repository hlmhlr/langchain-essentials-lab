"""Tests for lclab.tools: safe calculator, formatter and both agent types."""

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from lclab.tools import (
    TOOLS_TYPED, calculator, format_text, format_text_typed, react_executor, safe_eval, tool_calls_made,
)

pytestmark = pytest.mark.filterwarnings("ignore:.*was deprecated")


@pytest.mark.parametrize("expression, expected", [
    ("25 + 63", 88), ("15 * 7", 105), ("sqrt(144)", 12.0), ("2 ^ 10", 1024),
    ("-3 + 10 // 4", -1), ("`12.5 / 100 * 640`", 80.0),
])
def test_safe_eval_arithmetic(expression, expected):
    assert safe_eval(expression) == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').remove('x')", "open('secret.txt').read()", "x + 1", "10 ** 1000",
])
def test_safe_eval_rejects_code(expression):
    with pytest.raises(ValueError):
        safe_eval(expression)


def test_calculator_reports_errors_as_text():
    assert calculator("2 + 2") == "Result: 4"
    assert calculator("2 +").startswith("Error calculating")


def test_format_text():
    assert format_text("uppercase: hello world") == "HELLO WORLD"
    assert format_text("titlecase: langchain is awesome") == "Langchain Is Awesome"
    assert format_text("hello").startswith("Missing format")
    assert format_text("bold: hi").startswith("Unknown format")


def test_typed_formatter_rejects_unknown_style():
    assert format_text_typed.invoke({"style": "lowercase", "text": "HI"}) == "hi"
    with pytest.raises(Exception):  # the argument schema only allows three styles
        format_text_typed.invoke({"style": "bold", "text": "hi"})


def test_react_executor_runs_a_tool(fake_llm_factory):
    llm = fake_llm_factory(
        "Thought: I need maths\nAction: calculator\nAction Input: 25 + 63",
        "Thought: I now know the final answer\nFinal Answer: 88",
    )
    result = react_executor(llm, TOOLS_TYPED[:1]).invoke({"input": "What is 25 + 63?"})
    action, observation = result["intermediate_steps"][0]
    assert (action.tool, action.tool_input, observation) == ("calculator", "25 + 63", "Result: 88")
    assert result["output"] == "88"


class ToolCallingFake(GenericFakeChatModel):
    """GenericFakeChatModel can't bind tools; the scripted messages contain the tool calls."""

    def bind_tools(self, tools, **kwargs):
        return self


def test_tool_calling_agent_uses_typed_arguments():
    llm = ToolCallingFake(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "format_text", "id": "1",
                                           "args": {"style": "uppercase", "text": "hello world"}}]),
        AIMessage(content="HELLO WORLD"),
    ]))
    result = create_agent(llm, tools=TOOLS_TYPED).invoke({"messages": [{"role": "user", "content": "upper"}]})

    assert tool_calls_made(result["messages"]) == ["format_text({'style': 'uppercase', 'text': 'hello world'})"]
    assert result["messages"][-2].content == "HELLO WORLD"  # the tool's output
    assert result["messages"][-1].content == "HELLO WORLD"  # the agent's answer
