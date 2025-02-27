from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.storage import InMemoryStore
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from langchain_experimental.text_splitter import SemanticChunker


# Cache ------------------------
PARENT_RETRIEVER = None  #     |
MULTIQUERY_RETRIEVER = None  # |
COHERE_RETRIEVER = None  #     |
BM25_RETRIEVER = None  #       |
NAIVE_RETRIEVER = None  #      |
ENSEMBLE_RETRIEVER = None  #   |
SEMANTIC_RETRIEVER = None  #   |
# ------------------------------


def get_naive_retriever(
    docs: list[Document],
    embeddings: OpenAIEmbeddings,
    client: QdrantClient,
) -> VectorStoreRetriever:
    global NAIVE_RETRIEVER

    if NAIVE_RETRIEVER:
        return NAIVE_RETRIEVER

    client.create_collection(
        collection_name="Naive",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(
        collection_name="Naive",
        embedding=embeddings,
        client=client,
    )
    vectorstore.add_documents(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    NAIVE_RETRIEVER = retriever
    return NAIVE_RETRIEVER


def get_bm25_retriever(docs: list[Document]) -> BM25Retriever:
    global BM25_RETRIEVER

    if BM25_RETRIEVER:
        return BM25_RETRIEVER

    retriever = BM25Retriever.from_documents(docs)
    BM25_RETRIEVER = retriever
    return BM25_RETRIEVER


def get_cohere_retriever(
    docs: list[Document],
    embeddings: OpenAIEmbeddings,
    client: QdrantClient,
) -> ContextualCompressionRetriever:
    global COHERE_RETRIEVER

    if COHERE_RETRIEVER:
        return COHERE_RETRIEVER

    client.create_collection(
        collection_name="Cohere",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(
        collection_name="Cohere",
        embedding=embeddings,
        client=client,
    )
    vectorstore.add_documents(docs)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    compressor = CohereRerank(model="rerank-english-v3.0")
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    COHERE_RETRIEVER = retriever
    return COHERE_RETRIEVER


def get_multiquery_retriever(
    docs: list[Document],
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    client: QdrantClient,
) -> MultiQueryRetriever:
    global MULTIQUERY_RETRIEVER

    if MULTIQUERY_RETRIEVER:
        return MULTIQUERY_RETRIEVER

    client.create_collection(
        collection_name="MultiQuery",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(
        collection_name="MultiQuery",
        embedding=embeddings,
        client=client,
    )
    vectorstore.add_documents(docs)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    MULTIQUERY_RETRIEVER = retriever
    return MULTIQUERY_RETRIEVER


def get_parent_retriever(
    docs: list[Document],
    embeddings: OpenAIEmbeddings,
    client: QdrantClient,
) -> ParentDocumentRetriever:
    global PARENT_RETRIEVER

    if PARENT_RETRIEVER:
        return PARENT_RETRIEVER

    parent_docs = docs
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
    client.create_collection(
        collection_name="ParentFullDocuments",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    parent_document_vectorstore = QdrantVectorStore(
        collection_name="ParentFullDocuments",
        embedding=embeddings,
        client=client,
    )
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=parent_document_vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )
    retriever.add_documents(parent_docs, ids=None)
    PARENT_RETRIEVER = retriever
    return PARENT_RETRIEVER


def get_ensemble_retriever(
    docs: list[Document],
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    client: QdrantClient,
) -> EnsembleRetriever:
    global ENSEMBLE_RETRIEVER

    if ENSEMBLE_RETRIEVER:
        return ENSEMBLE_RETRIEVER

    retriever_list = [
        get_bm25_retriever(docs),
        get_naive_retriever(docs, embeddings, client),
        get_parent_retriever(docs, embeddings, client),
        get_cohere_retriever(docs, embeddings, client),
        get_multiquery_retriever(docs, llm, embeddings, client),
    ]
    equal_weighting = [1 / len(retriever_list)] * len(retriever_list)
    retriever = EnsembleRetriever(retrievers=retriever_list, weights=equal_weighting)
    ENSEMBLE_RETRIEVER = retriever
    return ENSEMBLE_RETRIEVER


def get_semantic_retriever(
    docs: list[Document],
    embeddings: OpenAIEmbeddings,
    client: QdrantClient,
) -> VectorStoreRetriever:
    global SEMANTIC_RETRIEVER

    if SEMANTIC_RETRIEVER:
        return SEMANTIC_RETRIEVER

    chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    chunked_docs = chunker.split_documents(docs)
    client.create_collection(
        collection_name="Semantic",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )
    vectorstore = QdrantVectorStore(
        collection_name="Semantic",
        embedding=embeddings,
        client=client,
    )
    vectorstore.add_documents(chunked_docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    SEMANTIC_RETRIEVER = retriever
    return SEMANTIC_RETRIEVER
