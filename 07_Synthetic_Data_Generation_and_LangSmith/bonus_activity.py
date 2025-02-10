import asyncio
import json
import pdb
import random
from operator import itemgetter
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableSerializable
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from tqdm.asyncio import tqdm

import prompts
import settings
from utils import load_files
from synthesizers import (
    BaseSynthesizer,
    SimpleEvolutionQuerySynthesizer,
    ReasoningQuerySynthesizer,
    MultiContextQuerySynthesizer,
)
from splitter import WordsTextSplitter

load_dotenv()
stg = settings.EXPENSIVE_SETTINGS


class CustomGraph:
    def __init__(
        self, llm_model: ChatOpenAI, embeddings_model: OpenAIEmbeddings
    ) -> None:
        self.llm = llm_model
        self.emb = embeddings_model
        self.vectorstore: QdrantVectorStore = QdrantVectorStore(
            client=self._get_new_client(),
            embedding=self.emb,
            collection_name="Custom-RAGAS",
        )
        self.splitter = WordsTextSplitter()
        self.docs: list[Document] = []
        self.chunks: list[dict] = []
        self._retriever: VectorStoreRetriever | None = None

        self._simple_evol_chain = self._build_simple_evol_chain()
        self._multi_context_evol_chain = self._build_multi_context_evol_chain()
        self._reasoning_evol_chain = self._build_reasoning_evol_chain()

        # Context items storage for "Multi Context Evolution"
        # Required to provide the context elements used for the final test suite
        self._mce_context_tmp_storage = []

    async def load_docs(self, docs: list[Document]) -> None:
        self.docs.extend(docs)
        chunks = self.splitter.split_documents(docs)

        for chunk in tqdm(chunks, total=len(chunks), desc="Processing chunks"):
            c_id = str(uuid4())
            c_summary = await self.aget_summary(chunk)
            c_themes = await self.aget_themes(chunk)
            c_persona = await self.aget_persona(chunk)

            self.chunks.append(
                {
                    "id": c_id,
                    "text": c_summary,
                    "metadata": {
                        "text": chunk,
                        "summary": c_summary,
                        "themes": c_themes,
                        "persona": c_persona,
                    },
                }
            )

        await self.vectorstore.aadd_texts(
            texts=[i["text"] for i in self.chunks],
            metadatas=[i["metadata"] for i in self.chunks],
            ids=[i["id"] for i in self.chunks],
        )

    async def generate(
        self, query_distribution: list[BaseSynthesizer], testset_size: int = 10
    ) -> list[dict]:
        assert sum([item.percentage for item in query_distribution]) == 100

        actions_map = {
            "SimpleEvolutionQuerySynthesizer": self._process_simple_evol,
            "MultiContextQuerySynthesizer": self._process_multi_context_evol,
            "ReasoningQuerySynthesizer": self._process_reasoning_evol,
        }
        n_runs = {
            item.name: int((item.percentage / 100) * testset_size)
            for item in query_distribution
        }

        result = []
        for synthesizer_name, runs_num in n_runs.items():
            for _ in tqdm(range(runs_num), total=runs_num, desc=synthesizer_name):
                random_chunk: dict = random.choice(self.chunks)
                function_to_call = actions_map[synthesizer_name]
                result.append(await function_to_call(random_chunk))
        return result

    async def aget_summary(self, text: str) -> str:
        placeholder = str({"Input": {"text": text}})
        prompt = prompts.SUMMARIZATION_PROMPT.replace("<placeholder>", placeholder)
        chain = self.llm | JsonOutputParser()
        response = await chain.ainvoke(prompt)
        return response["text"]

    async def aget_themes(self, text: str, max_num: int = 10) -> str:
        placeholder = str({"Input": {"text": text, "max_num": max_num}})
        prompt = prompts.THEMES_PROMPT.replace("<placeholder>", placeholder)
        chain = self.llm | JsonOutputParser()
        response = await chain.ainvoke(prompt)
        return response["output"]

    async def aget_persona(self, text: str) -> str:
        placeholder = str({"Input": {"text": text}})
        prompt = prompts.PERSONAS_PROMPT.replace("<placeholder>", placeholder)
        chain = self.llm | JsonOutputParser()
        response = await chain.ainvoke(prompt)
        return response

    async def _asearch_random_k(self, query: str, max_k: int = 4) -> list[Document]:
        return await self.vectorstore.asimilarity_search(
            query, k=random.randint(2, max_k)
        )

    async def _process_simple_evol(self, chunk: dict) -> dict:
        placeholder = {
            "Input": {
                "persona": chunk["metadata"]["persona"],
                "term": random.choice(chunk["metadata"]["themes"]),
                "query_style": self._get_random_query_style(),
                "query_length": self._get_random_query_length(),
                "context": chunk["metadata"]["text"],
            }
        }
        response = await self._simple_evol_chain.ainvoke(
            {
                "template": prompts.SIMPLE_EVOLUTION_PROMPT,
                "placeholder": str(placeholder),
            }
        )
        return {
            "ID": str(uuid4()),
            "Type:": "SimpleEvolution",
            "Question": response["query"],
            "Answer": response["answer"],
            "Context": chunk["metadata"]["text"],
        }

    async def _process_multi_context_evol(self, chunk: dict):
        placeholder = {
            "Input": {
                "persona": chunk["metadata"]["persona"],
                "themes": [],
                "query_style": self._get_random_query_style(),
                "query_length": self._get_random_query_length(),
                "context": [],
            }
        }

        response = await self._multi_context_evol_chain.ainvoke(
            {
                "template": prompts.MULTI_CONTEXT_EVOLUTION_PROMPT,
                "placeholder": placeholder,
                "context": chunk["metadata"]["text"],
            }
        )
        result = {
            "ID": str(uuid4()),
            "Type:": "MultiContextEvolution",
            "Question": response["query"],
            "Answer": response["answer"],
            # Extract context items used for generation
            "Context": self._mce_context_tmp_storage,
        }
        # Reset storage
        self._mce_context_tmp_storage = []
        return result

    async def _process_reasoning_evol(self, chunk: dict) -> dict:
        placeholder = {
            "Input": {
                "persona": chunk["metadata"]["persona"],
                "term": random.choice(chunk["metadata"]["themes"]),
                "query_style": self._get_random_query_style(),
                "query_length": self._get_random_query_length(),
                "context": chunk["metadata"]["text"],
            }
        }
        response = await self._simple_evol_chain.ainvoke(
            {
                "template": prompts.REASONING_EVOLUTION_PROMPT,
                "placeholder": str(placeholder),
            }
        )
        return {
            "ID": str(uuid4()),
            "Type:": "ReasoningEvolution",
            "Question": response["query"],
            "Answer": response["answer"],
            "Context": chunk["metadata"]["text"],
        }

    def _build_simple_evol_chain(self) -> RunnableSerializable:
        return RunnableLambda(self._process_template) | self.llm | JsonOutputParser()

    def _build_multi_context_evol_chain(self) -> RunnableSerializable:
        return (
            {
                "docs": itemgetter("context") | RunnableLambda(self._asearch_random_k),
                "placeholder": itemgetter("placeholder"),
                "template": itemgetter("template"),
            }
            | RunnableLambda(self._prepare_data)
            | RunnableLambda(self._process_template)
            | self.llm
            | JsonOutputParser()
        )

    def _build_reasoning_evol_chain(self) -> RunnableSerializable:
        return RunnableLambda(self._process_template) | self.llm | JsonOutputParser()

    def _prepare_data(self, _dict: dict) -> dict:
        placeholder = _dict["placeholder"]

        for num, doc in enumerate(_dict["docs"], start=1):
            placeholder["Input"]["themes"].append(random.choice(doc.metadata["themes"]))
            placeholder["Input"]["context"].append(
                f"<{num}-hop> {doc.metadata["text"]}"
            )
            # Append context item so we can extract it later
            self._mce_context_tmp_storage.append(doc.metadata["text"])

        return {
            "template": _dict["template"],
            "placeholder": str(placeholder),
        }

    def _process_template(self, _dict: dict[str]) -> str:
        return _dict["template"].replace("<placeholder>", _dict["placeholder"])

    def _get_random_query_style(self) -> str:
        options = ["MISSPELLED", "POOR_GRAMMAR", "WEB_SEARCH_LIKE", "PERFECT_GRAMMAR"]
        weights = [0.25, 0.25, 0.25, 0.25]
        return random.choices(options, weights)[0]

    def _get_random_query_length(self) -> str:
        options = ["SHORT", "MEDIUM", "LONG"]
        weights = [0.25, 0.50, 0.25]
        return random.choices(options, weights)[0]

    def _get_new_client(self) -> QdrantClient:
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name="Custom-RAGAS",
            vectors_config=VectorParams(size=stg.model_dims, distance=Distance.COSINE),
        )
        return client


async def main():
    print(f"LLM model: {stg.llm_model}\nEmbedding model: {stg.embeddings_model}")
    graph = CustomGraph(
        llm_model=ChatOpenAI(model=stg.llm_model, temperature=0),
        embeddings_model=OpenAIEmbeddings(model=stg.embeddings_model),
    )
    await graph.load_docs(load_files("data/", extensions=["*.html"]))
    testset = await graph.generate(
        testset_size=10,
        query_distribution=[
            SimpleEvolutionQuerySynthesizer(percentage=20),
            MultiContextQuerySynthesizer(percentage=40),
            ReasoningQuerySynthesizer(percentage=40),
        ],
    )
    # Just save dict to .json file
    filename = f"output/{stg.name}_testset-{str(uuid4())[:4]}.json"
    with open(filename, "w") as fp:
        json.dump(testset, fp)
    print(f"Success! Path to file: {filename}")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
