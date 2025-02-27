from operator import itemgetter

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSerializable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from prompts import PROMPT

# Cache ------------------------
FUSION_RETRIEVER = None  #     |
# ------------------------------


def get_fusion_retriever(
    docs: list[Document],
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    client: QdrantClient,
) -> ContextualCompressionRetriever:
    global FUSION_RETRIEVER

    if FUSION_RETRIEVER:
        return FUSION_RETRIEVER

    client.create_collection(
        collection_name="Fusion",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(
        collection_name="Fusion",
        embedding=embeddings,
        client=client,
    )
    vectorstore.add_documents(docs)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm
    )
    compressor = CohereRerank(model="rerank-english-v3.0")
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=multiquery_retriever
    )
    FUSION_RETRIEVER = retriever
    return FUSION_RETRIEVER


def get_fusion(
    docs: list[Document],
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    client: QdrantClient,
) -> RunnableSerializable:
    retriever = get_fusion_retriever(docs, llm, embeddings, client)
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": PROMPT | llm, "context": itemgetter("context")}
    )
    return chain
