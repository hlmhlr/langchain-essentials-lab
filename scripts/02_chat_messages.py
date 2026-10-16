"""Lab section "Chat message": System, Human and AI messages.

Aim
    Understand the three message roles and how chat history is passed to a
    chat model as a list of messages.

What you'll see
    Each example prints the messages sent (with their role) and the answer:
    1. system + human message (the system message sets role and style)
    2. a follow-up question with earlier turns included
    3. the same follow-up without history
    4. a human message on its own

What to notice
    - The model has no memory: it only knows what is in the message list.
      In example 3 it has to guess what "it" refers to.
    - This is the idea behind the Memory section later: memory = resending
      earlier messages.

Run:  uv run python scripts/02_chat_messages.py
"""

from lclab.llm import get_llm
from lclab.messages import build_messages
from lclab.utils import print_block, print_header

llm = get_llm()


def show(messages) -> None:
    """Print each message with its role, then the model's answer."""
    for message in messages:
        print(f"  [{message.type}] {message.content}")
    print_block("Answer", llm.invoke(messages).content)


# 1. A system message sets the assistant's role and style.
print_header("1. System + human message")
show(build_messages(
    system="You are a helpful assistant that recommends one podcast, in one short sentence.",
    question="I like history and long interviews. What should I listen to?",
))

# 2. Passing earlier turns lets the model understand a follow-up question.
#    "it" in the last question only makes sense with the earlier messages.
print_header("2. Chat history: the follow-up refers to the earlier answer")
show(build_messages(
    system="You are a friendly cooking assistant who answers in one short sentence.",
    history=[("I want a quick vegetarian dinner.", "Try a chickpea and spinach curry.")],
    question="How long does it take to make?",
))

# 3. The same follow-up without history: the model has to guess what "it" is.
print_header("3. Same follow-up without history")
show(build_messages(question="How long does it take to make?"))

# 4. The system message is optional.
print_header("4. Human message only")
show(build_messages(question="What month comes before March?"))
