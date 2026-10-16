"""Exercise 5: a chatbot with memory, three ways.

Aim
    Build a chatbot that remembers facts the user told it, and compare
    two legacy memory types with the current LangChain approach.

What you'll see
    The same five messages (name, favourite food, hobby, then questions
    about them) sent to:
      A. ConversationChain + ConversationBufferMemory (lab, legacy):
         keeps every message word for word; final memory is printed.
      B. ConversationChain + ConversationSummaryMemory (lab, legacy):
         keeps an LLM-written summary instead; final summary is printed,
         and the sizes of both memories are compared.
      C. create_agent + checkpointer (current way): remembers per thread;
         a question on a new thread shows that threads don't share memory.

What to notice
    - Buffer memory is exact but grows with every turn.
    - Summary memory stays small but costs an extra model call per turn,
      and a small model can drop details from the summary.
    - Threads keep different users' conversations apart, which the
      legacy single-memory object could not do.

Run:  uv run python scripts/13_memory_chatbot.py
      uv run python scripts/13_memory_chatbot.py --interactive   (chat yourself, 'quit' to exit)
"""

import argparse
import warnings

from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.chat_history import InMemoryChatMessageHistory

from lclab.llm import get_llm
from lclab.memory import ask_chatbot, chat_simulation, memory_chatbot
from lclab.utils import print_header

# The legacy classes print deprecation warnings; they are used on purpose here.
# (Set after the imports: importing LangChain resets the warning filters.)
warnings.filterwarnings("ignore", message=".*was deprecated.*")

FIRST_USER = "Hello, my name is Luca."
FIRST_AI = "Hello Luca! Nice to meet you. How can I help you today?"
TEST_INPUTS = [
    "My favorite food is pizza.",
    "I enjoy cycling along the lake.",
    "What weekend activities would you recommend for me?",
    "What was my favorite food again?",
    "Can you remember both my name and my favorite food?",
]
SYSTEM_PROMPT = "You are a friendly assistant. Keep answers to two sentences."


def interactive(llm) -> None:
    """Chat with the modern chatbot in the terminal."""
    bot = memory_chatbot(llm, SYSTEM_PROMPT)
    print("Chat with the bot. Type 'quit' to exit.")
    while (text := input("\nYou: ").strip()).lower() != "quit":
        print(f"Bot: {ask_chatbot(bot, text)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive", action="store_true", help="chat with the bot yourself")
    args = parser.parse_args()

    llm = get_llm()
    if args.interactive:
        interactive(llm)
        return

    # A. Buffer memory, starting from an existing history (as in the lab)
    print_header("A. ConversationBufferMemory (legacy)")
    history = InMemoryChatMessageHistory()
    history.add_user_message(FIRST_USER)
    history.add_ai_message(FIRST_AI)
    conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory(chat_memory=history))
    chat_simulation(lambda text: conversation.invoke(input=text)["response"], TEST_INPUTS)
    print("\nFinal memory contents:")
    print(conversation.memory.buffer)

    # B. Summary memory: the model rewrites a running summary after every turn
    print_header("B. ConversationSummaryMemory (legacy)")
    summary_memory = ConversationSummaryMemory(llm=llm)
    summary_memory.save_context({"input": FIRST_USER}, {"output": FIRST_AI})
    summary_conversation = ConversationChain(llm=llm, memory=summary_memory)
    chat_simulation(lambda text: summary_conversation.invoke(input=text)["response"], TEST_INPUTS)
    print("\nFinal summary memory contents:")
    print(summary_memory.buffer)

    print_header("Memory size comparison")
    print(f"Buffer memory:  {len(conversation.memory.buffer)} characters")
    print(f"Summary memory: {len(summary_memory.buffer)} characters")

    # C. Current way: an agent with a checkpointer, memory kept per thread
    print_header("C. create_agent + checkpointer (current)")
    bot = memory_chatbot(llm, SYSTEM_PROMPT)
    chat_simulation(lambda text: ask_chatbot(bot, text, thread_id="luca"), [FIRST_USER, *TEST_INPUTS])

    print("\nSame question on a new thread (a different user):")
    print(f"AI: {ask_chatbot(bot, 'What is my name?', thread_id='someone-else')}")


if __name__ == "__main__":
    main()
