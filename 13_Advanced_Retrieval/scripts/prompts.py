from langchain_core.prompts import ChatPromptTemplate

_template = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

PROMPT = ChatPromptTemplate.from_template(_template)
