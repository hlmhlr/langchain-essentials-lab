"""Lab section "Output parsers": JSON parser and comma-separated list parser.

Aim
    Turn the model's free-text answer into Python data that code can use
    directly, instead of parsing strings by hand.

What you'll see
    1. JsonOutputParser with a Pydantic schema (Riddle): the answer comes
       back as a Python dict with "question" and "answer".
    2. CommaSeparatedListOutputParser: "five programming languages" comes
       back as a Python list.
    Each result is printed with its Python type (dict / list).

What to notice
    - get_format_instructions() writes the formatting rules into the prompt
      for you; the parser then converts the reply.
    - Parsing only works if the model follows the format. Small models
      sometimes add extra text, which makes the parser fail (an error).

Run:  uv run python scripts/05_output_parsers.py
"""

from langchain_core.output_parsers import CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field  # the lab's langchain_core.pydantic_v1 was removed

from lclab.llm import get_llm
from lclab.utils import print_header

llm = get_llm()


# 1. JSON parser with a Pydantic schema.
class Riddle(BaseModel):
    question: str = Field(description="the riddle")
    answer: str = Field(description="the answer to the riddle")


print_header("1. JsonOutputParser")
output_parser = JsonOutputParser(pydantic_object=Riddle)
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)
chain = prompt | llm | output_parser
result = chain.invoke({"query": "Tell me a riddle."})
print(type(result).__name__, result)

# 2. Comma-separated list parser.
print_header("2. CommaSeparatedListOutputParser")
output_parser = CommaSeparatedListOutputParser()
prompt = PromptTemplate(
    template="Answer the user query. {format_instructions}\nList five {subject}.",
    input_variables=["subject"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)
chain = prompt | llm | output_parser
result = chain.invoke({"subject": "programming languages"})
print(type(result).__name__, result)
