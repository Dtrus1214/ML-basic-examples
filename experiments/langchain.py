"""
LangChain Sample - Core concepts explained with runnable examples.

LangChain is a framework for building applications powered by large language models (LLMs).
It provides: prompt templates, chains, output parsers, and integrations with many LLM providers.

Install: pip install langchain langchain-openai langchain-core
Set OPENAI_API_KEY in your environment to run the examples.
"""

import os
from typing import Optional

# ---------------------------------------------------------------------------
# 1. LLM INVOCATION - Calling an LLM (e.g. OpenAI)
# ---------------------------------------------------------------------------

def example_llm_basic():
    """Simple call to an LLM. Requires OPENAI_API_KEY."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # temperature=0 for deterministic
    response = llm.invoke("What is 2 + 2? Reply in one short sentence.")
    print("--- LLM basic ---")
    print(response.content)


# ---------------------------------------------------------------------------
# 2. PROMPT TEMPLATES - Reusable prompts with placeholders
# ---------------------------------------------------------------------------

def example_prompt_template():
    """Use a template so you can reuse the same prompt structure with different inputs."""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers in one short sentence."),
        ("human", "What is the capital of {country}?"),
    ])
    # Format with variables
    messages = prompt.format_messages(country="Japan")
    print("--- Prompt template (formatted) ---")
    print(messages)

    # Chain prompt + LLM
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm
    result = chain.invoke({"country": "France"})
    print(result.content)


# ---------------------------------------------------------------------------
# 3. CHAINS (LCEL) - Pipe prompts and LLMs together
# ---------------------------------------------------------------------------

def example_chain():
    """LangChain Expression Language (LCEL): use | to chain components."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_messages([
        ("human", "Translate this to {language}: {text}"),
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # chain = prompt -> llm
    chain = prompt | llm
    out = chain.invoke({"language": "Spanish", "text": "Hello, how are you?"})
    print("--- Chain (translation) ---")
    print(out.content)


# ---------------------------------------------------------------------------
# 4. OUTPUT PARSERS - Get structured data from LLM text
# ---------------------------------------------------------------------------

def example_output_parser():
    """Parse LLM output into a Python structure (e.g. dict, list, Pydantic model)."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    class Answer(BaseModel):
        value: int = Field(description="The numeric answer")
        unit: Optional[str] = Field(default=None, description="Unit if applicable")

    parser = PydanticOutputParser(pydantic_object=Answer)
    prompt = ChatPromptTemplate.from_messages([
        ("human", "How many days are in a leap year? Reply in JSON: {format_instructions}"),
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Inject format instructions into the prompt
    chain = (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | llm
        | parser
    )
    result = chain.invoke({})
    print("--- Output parser ---")
    print("Parsed:", result)  # Answer(value=366, unit=None)


# ---------------------------------------------------------------------------
# 5. SIMPLE SEQUENTIAL CHAIN - Multi-step flow
# ---------------------------------------------------------------------------

def example_multi_step():
    """Run two steps: generate an idea, then expand it."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    step1 = ChatPromptTemplate.from_messages([("human", "Give one short book title about {topic}.")]) | llm
    step2 = ChatPromptTemplate.from_messages([
        ("human", "In one sentence, describe what the book '{title}' is about."),
    ]) | llm

    title_msg = step1.invoke({"topic": "robots"})
    title = title_msg.content
    desc_msg = step2.invoke({"title": title})
    print("--- Multi-step ---")
    print("Title:", title)
    print("Description:", desc_msg.content)


# ---------------------------------------------------------------------------
# 6. STREAMING - Stream tokens as they are generated
# ---------------------------------------------------------------------------

def example_streaming():
    """Stream the LLM response token by token."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([("human", "Count from 1 to 5, one number per line.")])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm

    print("--- Streaming ---")
    for chunk in chain.stream({}):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


# ---------------------------------------------------------------------------
# RUN EXAMPLES (comment out if no API key)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run examples. Showing structure only.\n")
        print("Concepts covered:")
        print("  1. LLM invocation (ChatOpenAI)")
        print("  2. Prompt templates (ChatPromptTemplate)")
        print("  3. Chains with LCEL (prompt | llm)")
        print("  4. Output parsers (PydanticOutputParser)")
        print("  5. Multi-step flows")
        print("  6. Streaming (chain.stream())")
    else:
        example_llm_basic()
        print()
        example_prompt_template()
        print()
        example_chain()
        print()
        try:
            example_output_parser()
        except Exception as e:
            print("Output parser example (needs pydantic):", e)
        print()
        example_multi_step()
        print()
        example_streaming()
