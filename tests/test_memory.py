"""Tests for lclab.memory with a fake model (no Ollama)."""

import itertools

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from lclab.memory import ask_chatbot, chat_simulation, memory_chatbot


def endless_fake_llm():
    """A fake model that answers 'reply 0', 'reply 1', ... forever."""
    return GenericFakeChatModel(messages=(AIMessage(content=f"reply {i}") for i in itertools.count()))


def test_chat_simulation_returns_every_turn(capsys):
    turns = chat_simulation(lambda text: text.upper(), ["hi", "bye"])
    assert turns == [("hi", "HI"), ("bye", "BYE")]
    assert "Human: hi" in capsys.readouterr().out


def test_chatbot_remembers_within_a_thread():
    bot = memory_chatbot(endless_fake_llm(), "Be brief.")
    assert ask_chatbot(bot, "I am Luca", thread_id="t1") == "reply 0"
    assert ask_chatbot(bot, "Who am I?", thread_id="t1") == "reply 1"

    saved = bot.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert [m.content for m in saved] == ["I am Luca", "reply 0", "Who am I?", "reply 1"]


def test_threads_do_not_share_memory():
    bot = memory_chatbot(endless_fake_llm(), "Be brief.")
    ask_chatbot(bot, "I am Luca", thread_id="t1")
    ask_chatbot(bot, "Who am I?", thread_id="t2")

    saved = bot.get_state({"configurable": {"thread_id": "t2"}}).values["messages"]
    assert [m.content for m in saved] == ["Who am I?", "reply 1"]
