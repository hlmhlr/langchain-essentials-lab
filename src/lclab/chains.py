"""Exercise 6: a three-step product-review pipeline, built two ways.

Purpose
    The same pipeline (sentiment -> summary -> customer response) built with
    the lab's traditional LLMChain + SequentialChain and with modern LCEL,
    so the two approaches can be compared on identical prompts.

Contents
    SENTIMENT_PROMPT, SUMMARY_PROMPT, RESPONSE_PROMPT   the lab's three templates
    traditional_pipeline()   LLMChain x3 + SequentialChain (legacy, langchain-classic)
    lcel_pipeline()          prompt | llm | parser steps joined with RunnablePassthrough.assign

Both pipelines take {"review": ...} and return a dict with the keys
review, sentiment, summary and response.
"""

from langchain_classic.chains import LLMChain, SequentialChain
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

SENTIMENT_PROMPT = PromptTemplate.from_template(
    """Analyze the sentiment of the following product review as positive, negative, or neutral.
Provide your analysis in the format: "SENTIMENT: [positive/negative/neutral]"

Review: {review}

Your analysis:
"""
)

SUMMARY_PROMPT = PromptTemplate.from_template(
    """Summarize the following product review into 3-5 key bullet points.
Each bullet point should be concise and capture an important aspect mentioned in the review.

Review: {review}
Sentiment: {sentiment}

Key points:
"""
)

RESPONSE_PROMPT = PromptTemplate.from_template(
    """Write a helpful response to a customer based on their product review.
If the sentiment is positive, thank them for their feedback. If negative, express understanding
and suggest a solution or next steps. Personalize based on the specific points they mentioned.

Review: {review}
Sentiment: {sentiment}
Key points: {summary}

Response to customer:
"""
)


def traditional_pipeline(llm: BaseChatModel) -> SequentialChain:
    """Three LLMChains connected by their output_key names (lab's approach)."""
    sentiment_chain = LLMChain(llm=llm, prompt=SENTIMENT_PROMPT, output_key="sentiment")
    summary_chain = LLMChain(llm=llm, prompt=SUMMARY_PROMPT, output_key="summary")
    response_chain = LLMChain(llm=llm, prompt=RESPONSE_PROMPT, output_key="response")

    return SequentialChain(
        chains=[sentiment_chain, summary_chain, response_chain],
        input_variables=["review"],
        output_variables=["sentiment", "summary", "response"],
    )


def lcel_pipeline(llm: BaseChatModel):
    """The same steps with LCEL.

    RunnablePassthrough.assign(key=step) runs `step` on the current dict and
    adds its result under `key`, keeping every earlier key. Each prompt only
    reads the keys it needs, so the step can be passed directly (the lab
    wraps each step in a lambda, which does the same thing).
    """
    parser = StrOutputParser()
    return (
        RunnablePassthrough.assign(sentiment=SENTIMENT_PROMPT | llm | parser)
        | RunnablePassthrough.assign(summary=SUMMARY_PROMPT | llm | parser)
        | RunnablePassthrough.assign(response=RESPONSE_PROMPT | llm | parser)
    )
