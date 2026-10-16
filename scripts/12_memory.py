"""Lab section "Memory": chat message history and conversation buffer memory.

Aim
    See how a chatbot "remembers": the earlier messages are stored and sent
    again with every new message.

What you'll see
    1. A chat message history: messages are added to it, sent to the model,
       and the model's reply is added back.
    2. A ConversationChain with buffer memory (verbose): the full prompt is
       printed each turn, so you can watch the history grow, and the last
       question ("Who am I?") is answered from that history.

What to notice
    - The model itself stores nothing. Memory = the history pasted into the prompt.
    - The prompt grows every turn, so long chats cost more tokens and can
      reach the model's context limit.
    - These classes are the lab's legacy API (deprecated in LangChain 1.x);
      scripts/13_memory_chatbot.py also shows the current way.

Run:  uv run python scripts/12_memory.py
"""

import warnings

from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory

from lclab.llm import get_llm
from lclab.utils import print_header

# The legacy classes print deprecation warnings; they are used on purpose here.
# (Set after the imports: importing LangChain resets the warning filters.)
warnings.filterwarnings("ignore", message=".*was deprecated.*")

chat = get_llm()

# 1. Chat message history
print_header("1. Chat message history")
history = InMemoryChatMessageHistory()  # the lab's ChatMessageHistory
history.add_ai_message("hi!")
history.add_user_message("What is the capital of Japan?")
print(history.messages)

ai_response = chat.invoke(history.messages)  # send the whole history to the model
print(f"\nAI: {ai_response.content}")

history.add_ai_message(ai_response.content)  # store the reply for the next turn
print(f"\nMessages in history now: {len(history.messages)}")

# 2. Conversation buffer memory: the chain stores every turn automatically.
print_header("2. ConversationChain + ConversationBufferMemory")
conversation = ConversationChain(llm=chat, verbose=True, memory=ConversationBufferMemory())

for text in ["Hello, I am a chef from Naples. Who are you?", "What can you do?", "Who am I?"]:
    print(f"\n>>> {text}")
    print(f"AI: {conversation.invoke(input=text)['response']}")
