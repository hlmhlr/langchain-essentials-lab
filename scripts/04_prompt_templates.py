"""Lab section "Prompt templates": string, chat and MessagesPlaceholder.

Run:  uv run python scripts/04_prompt_templates.py
"""

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

from lclab.llm import get_llm
from lclab.utils import print_block, print_header

llm = get_llm()

# 1. String prompt template: fills {placeholders} into one string.
print_header("1. PromptTemplate")
prompt = PromptTemplate.from_template("Give me one {adjective} fact about {topic}.")
input_ = {"adjective": "surprising", "topic": "octopuses"}
print(prompt.invoke(input_))

# 2. Chat prompt template: a list of messages with placeholders.
print_header("2. ChatPromptTemplate")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant."),
    ("user", "Suggest one thing to do in {city}."),
])
print(prompt.invoke({"city": "Lisbon"}))

# 3. MessagesPlaceholder: slot a whole list of messages into the template.
print_header("3. MessagesPlaceholder + chain")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("msgs"),
])
input_ = {"msgs": [HumanMessage(content="What is the day after Friday?")]}
print(prompt.invoke(input_))

chain = prompt | llm  # the filled template goes straight into the model
print_block("Answer", chain.invoke(input_).content)
