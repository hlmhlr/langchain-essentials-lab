"""Tests for lclab.messages.build_messages."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from lclab.messages import build_messages


def test_question_only():
    messages = build_messages("Hi?")
    assert messages == [HumanMessage(content="Hi?")]


def test_system_history_and_question_in_order():
    messages = build_messages(
        "How long?",
        system="Be brief.",
        history=[("Dinner idea?", "Curry."), ("Spicy?", "Mildly.")],
    )
    assert [type(m) for m in messages] == [
        SystemMessage, HumanMessage, AIMessage, HumanMessage, AIMessage, HumanMessage,
    ]
    assert messages[0].content == "Be brief."
    assert messages[2].content == "Curry."
    assert messages[-1].content == "How long?"


def test_messages_work_with_a_chat_model(fake_llm_factory):
    llm = fake_llm_factory("About 25 minutes.")
    answer = llm.invoke(build_messages("How long?", system="Be brief."))
    assert answer.content == "About 25 minutes."
