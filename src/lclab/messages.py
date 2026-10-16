"""Build the list of chat messages a chat model takes as input.

A chat model doesn't receive one long string. It receives a list of
messages, and each message has a role:
  - SystemMessage: instructions that set the assistant's behaviour
  - HumanMessage:  what the user says
  - AIMessage:     what the assistant said earlier
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def build_messages(
    question: str,
    system: str | None = None,
    history: list[tuple[str, str]] | None = None,
) -> list[BaseMessage]:
    """Return [system?, human, ai, human, ai, ..., human(question)].

    `history` is a list of earlier (user said, assistant answered) pairs.
    """
    messages: list[BaseMessage] = []

    if system:
        messages.append(SystemMessage(content=system))

    for user_text, ai_text in history or []:
        messages.append(HumanMessage(content=user_text))
        messages.append(AIMessage(content=ai_text))

    messages.append(HumanMessage(content=question))
    return messages
