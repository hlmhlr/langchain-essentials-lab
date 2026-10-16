"""Exercise 2: JSON output parser, tested with three books.

Run:  uv run python scripts/06_book_json.py
"""

from lclab.llm import get_llm
from lclab.parsers import book_chain
from lclab.utils import print_header

chain = book_chain(get_llm())

for book_name in ["1984", "The Hobbit", "Sapiens"]:
    print_header(book_name)
    result = chain.invoke({"book_name": book_name})

    # The result is a normal Python dict, so fields can be read directly.
    print(f"Type:   {type(result).__name__}")
    print(f"Title:  {result['title']}")
    print(f"Author: {result['author']}")
    print(f"Year:   {result['year']}")
    print(f"Genre:  {result['genre']}")
