"""Lab section "Chains": simple chain and sequential chain, traditional vs LCEL.

Aim
    Connect prompts and model calls into a workflow, where one step's
    output becomes the next step's input.

What you'll see
    1. Simple chain: a classic dish from a region, with LLMChain (legacy)
       and with LCEL (prompt | llm | parser).
    2. Sequential chain: region -> dish -> recipe -> cooking time, with
       SequentialChain (legacy) and with LCEL + RunnablePassthrough.assign.
       Both print every intermediate result.

What to notice
    - Breaking a task into small steps keeps each prompt focused.
    - Traditional chains link steps by matching output_key names.
      LCEL links them with | and RunnablePassthrough.assign, which keeps
      all earlier results in the dictionary.
    - The lab's LCEL example builds the sequential chain but never runs it;
      here it is invoked.

Run:  uv run python scripts/14_chains.py
"""

import warnings
from pprint import pprint

from langchain_classic.chains import LLMChain, SequentialChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from lclab.llm import get_llm
from lclab.utils import print_block, print_header

# The legacy classes print deprecation warnings; they are used on purpose here.
# (Set after the imports: importing LangChain resets the warning filters.)
warnings.filterwarnings("ignore", message=".*was deprecated.*")

llm = get_llm()
REGION = "Sicily"  # the lab uses China

location_template = """Your job is to come up with a classic dish from the area that the users suggests.
{location}

YOUR RESPONSE:
"""
dish_template = """Given a meal {meal}, give a short and simple recipe on how to make that dish at home.

YOUR RESPONSE:
"""
time_template = """Given the recipe {recipe}, estimate how much time I need to cook it.

YOUR RESPONSE:
"""

# 1. Simple chain
print_header("1a. Simple chain: LLMChain (traditional)")
location_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(location_template), output_key="meal")
print(location_chain.invoke(input={"location": REGION})["meal"])

print_header("1b. Simple chain: LCEL")
location_chain_lcel = PromptTemplate.from_template(location_template) | llm | StrOutputParser()
print(location_chain_lcel.invoke({"location": REGION}))

# 2. Sequential chain
print_header("2a. Sequential chain: SequentialChain (traditional)")
dish_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(dish_template), output_key="recipe")
recipe_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(time_template), output_key="time")

overall_chain = SequentialChain(
    chains=[location_chain, dish_chain, recipe_chain],
    input_variables=["location"],
    output_variables=["meal", "recipe", "time"],
)
pprint(overall_chain.invoke(input={"location": REGION}))

print_header("2b. Sequential chain: LCEL + RunnablePassthrough.assign")
dish_chain_lcel = PromptTemplate.from_template(dish_template) | llm | StrOutputParser()
time_chain_lcel = PromptTemplate.from_template(time_template) | llm | StrOutputParser()

overall_chain_lcel = (
    RunnablePassthrough.assign(meal=lambda x: location_chain_lcel.invoke({"location": x["location"]}))
    | RunnablePassthrough.assign(recipe=lambda x: dish_chain_lcel.invoke({"meal": x["meal"]}))
    | RunnablePassthrough.assign(time=lambda x: time_chain_lcel.invoke({"recipe": x["recipe"]}))
)
result = overall_chain_lcel.invoke({"location": REGION})  # the lab forgets to call invoke here
for key in ["meal", "recipe", "time"]:
    print_block(key, result[key])
