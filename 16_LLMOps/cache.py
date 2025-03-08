import uuid

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams


class SemanticCache:
    def __init__(self, llm, client) -> None:
        self._llm = llm
        self._emb = OpenAIEmbeddings()
        self._client = client
        self._vectorstore = self._setup_vectorstore()
        self._retriever = self._vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 1}
        )

    async def __call__(self, *args: tuple, **kwargs: dict) -> str:
        messages = args[0].messages
        query = messages[-1].content
        cached_answer = await self._retriever.ainvoke(query)
        if cached_answer:
            return f'[from cache]\n\n{cached_answer[0].metadata["response"]}'
        else:
            response = await self._llm.ainvoke(messages)
            document = Document(page_content=query, metadata={"response": response})
            await self._vectorstore.aadd_documents([document])
            return response

    def _setup_vectorstore(self):
        collection_name = f"cache_{uuid.uuid4()}"
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        return QdrantVectorStore(
            client=self._client, collection_name=collection_name, embedding=self._emb
        )
