from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient

import retrievers
from prompts import PROMPT


def get_naive(
    docs: list[Document],
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    client: QdrantClient,
) -> RunnableSerializable:
    retriever = retrievers.get_naive_retriever(docs, embeddings, client)
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": PROMPT | llm, "context": itemgetter("context")}
    )
    return chain


def get_bm25(docs: list[Document], llm: ChatOpenAI) -> RunnableSerializable:
    retriever = retrievers.get_bm25_retriever(docs)
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": PROMPT | llm, "context": itemgetter("context")}
    )
    return chain


def get_cohere(
    docs: list[Document],
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    client: QdrantClient,
) -> RunnableSerializable:
    retriever = retrievers.get_cohere_retriever(docs, embeddings, client)
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": PROMPT | llm, "context": itemgetter("context")}
    )
    return chain


def get_multiquery(
    docs: list[Document],
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    client: QdrantClient,
) -> RunnableSerializable:
    retriever = retrievers.get_multiquery_retriever(docs, llm, embeddings, client)
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": PROMPT | llm, "context": itemgetter("context")}
    )
    return chain


def get_parent(
    docs: list[Document],
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    client: QdrantClient,
) -> RunnableSerializable:
    retriever = retrievers.get_parent_retriever(docs, embeddings, client)
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": PROMPT | llm, "context": itemgetter("context")}
    )
    return chain


def get_ensemble(
    docs: list[Document],
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    client: QdrantClient,
) -> RunnableSerializable:
    retriever = retrievers.get_ensemble_retriever(docs, llm, embeddings, client)
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": PROMPT | llm, "context": itemgetter("context")}
    )
    return chain


def get_naive_with_semantic(
    docs: list[Document],
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    client: QdrantClient,
) -> RunnableSerializable:
    retriever = retrievers.get_semantic_retriever(docs, embeddings, client)
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": PROMPT | llm, "context": itemgetter("context")}
    )
    return chain
