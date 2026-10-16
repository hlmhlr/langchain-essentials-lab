"""Exercise 2: JSON output parser, tested with three books.

Aim
    Build a prompt | llm | JsonOutputParser chain that returns structured
    book information (title, author, year, genre).

What you'll see
    For "1984", "The Hobbit" and "Sapiens": the type of the result (dict)
    and each field read from it.

What to notice
    - The fields are read with result["title"] etc. like any Python dict.
    - The values are only as accurate as the model's knowledge: a small
      model may get a year or genre wrong even when the format is right.

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
