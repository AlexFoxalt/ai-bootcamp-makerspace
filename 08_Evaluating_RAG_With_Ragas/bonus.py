import asyncio
import pdb
from copy import deepcopy

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from typing_extensions import TypedDict, Literal
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.testset_schema import Testset
from tqdm.asyncio import tqdm
from ragas import EvaluationDataset
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity,
)
from ragas import evaluate, RunConfig
from ragas.dataset_schema import EvaluationResult
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()

RAG_PROMPT = """
You are a helpful assistant who answers questions based on provided context. 

You must only use the provided context, and cannot use your own knowledge.

### Question
{question}

### Context
{context}
"""


class State(TypedDict):
    question: str
    context: list[Document]
    response: str


class RAGApplication:
    def __init__(self):
        self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._emb = OpenAIEmbeddings(model="text-embedding-3-small")
        self._emb_dims = 1536
        self._coll_name = "Eval-With-Ragas"
        self._vectorstore: QdrantVectorStore = QdrantVectorStore(
            client=self._get_client(),
            embedding=self._emb,
            collection_name=self._coll_name,
        )
        self._semantic_chunker = SemanticChunker(self._emb)
        self._recursive_chunker = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        self._chunker_map = {
            "semantic": self._semantic_chunker,
            "recursive": self._recursive_chunker,
        }

    async def load_files(
        self,
        path_to_data: str,
        extensions: list[str],
        chunker: Literal["semantic", "recursive"],
    ):
        docs = self._load_from_path(path_to_data, extensions)
        chunks = self._chunker_map[chunker].split_documents(docs)
        await self._vectorstore.aadd_documents(chunks)

    async def build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(State)
        graph.add_node("retrieve", self._acohere_retrieve)
        graph.add_node("generate", self._acall_model)
        graph.add_edge("retrieve", "generate")
        graph.add_edge(START, "retrieve")
        return graph.compile()

    async def _acohere_retrieve(self, state: State) -> dict[str, list[Document]]:
        compressor = CohereRerank(model="rerank-v3.5")
        retriever = self._vectorstore.as_retriever(search_kwargs={"k": 15})
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever, search_kwargs={"k": 3}
        )
        return {"context": await compression_retriever.ainvoke(state["question"])}

    async def _acall_model(self, state: State) -> dict[str, str]:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        prompt = await rag_prompt.aformat_messages(
            question=state["question"], context=docs_content
        )
        response = await self._llm.ainvoke(prompt)
        return {"response": response.content}

    def _get_client(self) -> QdrantClient:
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name=self._coll_name,
            vectors_config=VectorParams(size=self._emb_dims, distance=Distance.COSINE),
        )
        return client

    def _load_from_path(
        self, path_to_data: str, extensions: list[str]
    ) -> list[Document]:
        res = []
        for ext in extensions:
            res.extend(DirectoryLoader(path_to_data, glob=ext).load())
        return res


class EvaluationApp:
    def __init__(self):
        self._llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
        self._emb = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model="text-embedding-3-small")
        )
        self._gen = TestsetGenerator(llm=self._llm, embedding_model=self._emb)

    def generate_dataset(
        self, path_to_data: str, extensions: list[str], size: int = 10
    ):
        docs = self._load_from_path(path_to_data, extensions)
        return self._gen.generate_with_langchain_docs(docs, testset_size=size)

    async def convert_to_evaluation(
        self,
        testset: Testset,
        graph: CompiledStateGraph,
        *,
        testset_name: str = "Default",
    ) -> EvaluationDataset:
        testset_copy = Testset.from_pandas(testset.to_pandas())

        for test_row in tqdm(testset_copy, total=len(testset_copy), desc=testset_name):
            response = await graph.ainvoke(
                {"question": test_row.eval_sample.user_input}
            )
            test_row.eval_sample.response = response["response"]
            test_row.eval_sample.retrieved_contexts = [
                context.page_content for context in response["context"]
            ]

        return EvaluationDataset.from_pandas(testset_copy.to_pandas())

    def judge(
        self, model: str, testset: EvaluationDataset, timeout: int = 360
    ) -> EvaluationResult:
        custom_run_config = RunConfig(timeout=timeout)
        return evaluate(
            dataset=testset,
            metrics=[
                LLMContextRecall(),
                Faithfulness(),
                FactualCorrectness(),
                ResponseRelevancy(),
                ContextEntityRecall(),
                NoiseSensitivity(),
            ],
            llm=ChatOpenAI(model=model, temperature=0),
            run_config=custom_run_config,
        )

    def compare(
        self,
        x: EvaluationResult,
        y: EvaluationResult,
        x_name: str = "Model 1",
        y_name: str = "Model 2",
    ) -> None:
        dict1 = self._normalize_eval_result(x)
        dict2 = self._normalize_eval_result(y)

        assert dict1.keys() == dict2.keys()

        metrics = list(dict1.keys())
        values1 = list(dict1.values())
        values2 = list(dict2.values())

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots()
        ax.bar(x - width / 2, values1, width, label=x_name, color="blue", alpha=0.7)
        ax.bar(x + width / 2, values2, width, label=y_name, color="orange", alpha=0.7)

        ax.set_xlabel("Metrics")
        ax.set_ylabel("Values")
        ax.set_title("Comparison of Models")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()

        plt.ylim(0, 1)
        plt.show()

    def _normalize_eval_result(self, item: EvaluationResult) -> dict:
        return {k: round(v, 4) for k, v in item._repr_dict.items()}

    def _load_from_path(
        self, path_to_data: str, extensions: list[str]
    ) -> list[Document]:
        res = []
        for ext in extensions:
            res.extend(DirectoryLoader(path_to_data, glob=ext).load())
        return res


async def main():
    recursive_app = RAGApplication()

    await recursive_app.load_files("data/", extensions=["*.html"], chunker="recursive")
    recursive_graph = await recursive_app.build_graph()

    semantic_app = RAGApplication()
    await semantic_app.load_files("data/", extensions=["*.html"], chunker="semantic")
    semantic_graph = await semantic_app.build_graph()

    eval_app = EvaluationApp()

    init_testset = eval_app.generate_dataset("data/", extensions=["*.html"], size=10)
    recursive_testset = await eval_app.convert_to_evaluation(
        init_testset, recursive_graph, testset_name="Recursive"
    )
    semantic_testset = await eval_app.convert_to_evaluation(
        init_testset, semantic_graph, testset_name="Semantic"
    )

    judge_model = "gpt-4o-mini"
    recursive_result = eval_app.judge(model=judge_model, testset=recursive_testset)
    semantic_result = eval_app.judge(model=judge_model, testset=semantic_testset)

    eval_app.compare(
        x=recursive_result,
        y=semantic_result,
        x_name="Recursive Chunking",
        y_name="Semantic Chunking",
    )


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
