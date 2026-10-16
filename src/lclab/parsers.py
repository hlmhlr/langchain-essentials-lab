"""Exercise 2: a chain that returns book information as a Python dict.

Same pattern as the lab's movie example: prompt -> model -> JsonOutputParser.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

BOOK_KEYS = ("title", "author", "year", "genre")

FORMAT_INSTRUCTIONS = """RESPONSE FORMAT: Return ONLY a single JSON object, no markdown, no extra keys. It must look exactly like:
{
  "title": "book title",
  "author": "author name",
  "year": 2000,
  "genre": "book genre"
}"""

PROMPT = PromptTemplate(
    template="""You are a JSON-only assistant.

Task: Generate info about the book "{book_name}" in JSON format.

{format_instructions}
""",
    input_variables=["book_name"],
    partial_variables={"format_instructions": FORMAT_INSTRUCTIONS},
)


def book_chain(llm: BaseChatModel):
    """prompt | llm | parser: input {"book_name": ...}, output a dict."""
    return PROMPT | llm | JsonOutputParser()
