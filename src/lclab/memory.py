"""Chatbot memory helpers (Memory section and Exercise 5).

Purpose
    The lab's memory classes (ConversationBufferMemory, ConversationChain,
    ConversationSummaryMemory) are deprecated in LangChain 1.x. The current
    way is an agent with a checkpointer: it saves the conversation after
    every turn and reloads it on the next one, separately for each thread_id
    (one thread = one user or one conversation).

Contents
    chat_simulation()  the lab's helper: run a list of inputs and print replies
    memory_chatbot()   a modern chatbot that remembers, built with create_agent
    ask_chatbot()      send one message to that chatbot on a given thread
"""

from collections.abc import Callable

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver


def chat_simulation(ask: Callable[[str], str], inputs: list[str]) -> list[tuple[str, str]]:
    """Run inputs through a chatbot and print each turn (from the lab).

    `ask` is any function text -> reply, so the same helper works for the
    legacy ConversationChain and the modern chatbot.
    """
    print("\n=== Beginning Chat Simulation ===")
    turns = []
    for i, user_input in enumerate(inputs, start=1):
        print(f"\n--- Turn {i} ---")
        print(f"Human: {user_input}")
        reply = ask(user_input)
        print(f"AI: {reply.strip()}")
        turns.append((user_input, reply))
    print("\n=== End of Chat Simulation ===")
    return turns


def memory_chatbot(llm: BaseChatModel, system_prompt: str):
    """A chatbot (an agent with no tools) whose history is kept per thread."""
    return create_agent(llm, tools=[], system_prompt=system_prompt, checkpointer=InMemorySaver())


def ask_chatbot(chatbot, text: str, thread_id: str = "default") -> str:
    """Send one user message on `thread_id` and return the reply text."""
    config = {"configurable": {"thread_id": thread_id}}
    result = chatbot.invoke({"messages": [{"role": "user", "content": text}]}, config)
    return result["messages"][-1].content  # the last message is the new reply
