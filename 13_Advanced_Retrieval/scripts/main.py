import asyncio
from collections import deque
from datetime import datetime, timedelta
from pprint import pprint

from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain_core.runnables.base import RunnableSerializable
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from ragas import EvaluationDataset
from ragas import evaluate, RunConfig
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity,
)
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.testset_schema import Testset
from tqdm.asyncio import tqdm
import numpy as np
import matplotlib.pyplot as plt

import chains
import rag_fusion as fusion
from logger import logger

load_dotenv()


def load_docs() -> list[Document]:
    documents = []
    for i in range(1, 5):
        loader = CSVLoader(
            file_path=f"john_wick_{i}.csv",
            metadata_columns=[
                "Review_Date",
                "Review_Title",
                "Review_Url",
                "Author",
                "Rating",
            ],
        )
        movie_docs = loader.load()
        for doc in movie_docs:
            doc.metadata["Movie_Title"] = f"John Wick {i}"
            doc.metadata["Rating"] = (
                int(doc.metadata["Rating"]) if doc.metadata["Rating"] else 0
            )
            doc.metadata["last_accessed_at"] = datetime.now() - timedelta(days=4 - i)
        documents.extend(movie_docs)
    return documents


def gen_testset(docs: list[Document], size: int = 10):
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    gen = TestsetGenerator(llm=llm, embedding_model=emb)
    return gen.generate_with_langchain_docs(docs, testset_size=size)


async def aask_chain(
    testset: Testset,
    chain: RunnableSerializable,
    *,
    testset_name: str = "Default",
) -> EvaluationDataset:
    testset_copy = Testset.from_pandas(testset.to_pandas())

    for test_row in tqdm(
        testset_copy, total=len(testset_copy), desc=f"Asking {testset_name}"
    ):
        response = await chain.ainvoke({"question": test_row.eval_sample.user_input})
        test_row.eval_sample.response = response["response"].content
        test_row.eval_sample.retrieved_contexts = [
            context.page_content for context in response["context"]
        ]

    return EvaluationDataset.from_pandas(testset_copy.to_pandas())


def judge(
    testset: EvaluationDataset, model: str = "gpt-4o-mini", timeout: int = 360
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


def draw_metrics(
    data: list[EvaluationResult] | list[dict], names: list[str]
) -> dict[str, dict]:
    if not data:
        return {}
    if isinstance(data[0], EvaluationResult):
        prepared_data = [
            {k: round(v, 4) for k, v in item._repr_dict.items()} for item in data
        ]
    elif isinstance(data[0], dict):
        prepared_data = data
    else:
        raise Exception("Unexpected type.")

    categories = list(prepared_data[0].keys())
    values = np.array([[item[key] for key in categories] for item in prepared_data])
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(10, 6))

    padding_factor = 1.5  # Adjust for spacing
    bar_width = min(0.8 / (len(values) + padding_factor), 0.15)
    colors = deque(
        [
            "blue",
            "orange",
            "green",
            "red",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
    )
    for i, (val, name) in enumerate(zip(values, names)):
        ax.bar(
            x + i * bar_width, val, width=bar_width, color=colors.popleft(), label=name
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Values")
    ax.set_title("Comparison of Retrievals")
    ax.legend()

    plt.show()

    return {name: item for item, name in zip(prepared_data, names)}


async def main():
    basic_llm = ChatOpenAI()
    embedding_llm = OpenAIEmbeddings(model="text-embedding-3-small")
    client = QdrantClient(location=":memory:")

    docs = load_docs()
    logger.info(f"Loaded {len(docs)} documents")

    naive_chain = chains.get_naive(docs, basic_llm, embedding_llm, client)
    logger.info("Created Naive chain")

    naive_sematic_chain = chains.get_naive_with_semantic(
        docs, basic_llm, embedding_llm, client
    )
    logger.info("Created Naive-with-Semantic chain")

    bm25_chain = chains.get_bm25(docs, basic_llm)
    logger.info("Created BM25 chain")

    cohere_chain = chains.get_cohere(docs, basic_llm, embedding_llm, client)
    logger.info("Created Contextual Compression chain")

    multiquery_chain = chains.get_multiquery(docs, basic_llm, embedding_llm, client)
    logger.info("Created Multi-Query chain")

    parent_chain = chains.get_parent(docs, basic_llm, embedding_llm, client)
    logger.info("Created Parent-Document chain")

    ensemble_chain = chains.get_ensemble(docs, basic_llm, embedding_llm, client)
    logger.info("Created Ensemble chain")

    fusion_chain = fusion.get_fusion(docs, basic_llm, embedding_llm, client)
    logger.info("Created Fusion chain")

    logger.info("All chains created successfully")

    testset = gen_testset(docs, 10)
    logger.info(f"Testset with len {len(testset)} generated successfully")

    naive_answers = await aask_chain(testset, naive_chain, testset_name="Naive")
    naive_semantic_answers = await aask_chain(
        testset, naive_sematic_chain, testset_name="Naive-with-Semantic"
    )
    bm25_answers = await aask_chain(testset, bm25_chain, testset_name="BM25")
    cohere_answers = await aask_chain(
        testset, cohere_chain, testset_name="Contextual Compression"
    )
    multiquery_answers = await aask_chain(
        testset, multiquery_chain, testset_name="Multi-Query"
    )
    parent_answers = await aask_chain(testset, parent_chain, testset_name="Parent")
    ensemble_answers = await aask_chain(
        testset, ensemble_chain, testset_name="Ensemble"
    )
    fusion_answers = await aask_chain(testset, fusion_chain, testset_name="Fusion")

    naive_verdict = judge(naive_answers)
    naive_semantic_verdict = judge(naive_semantic_answers)
    bm25_verdict = judge(bm25_answers)
    cohere_verdict = judge(cohere_answers)
    multiquery_verdict = judge(multiquery_answers)
    parent_verdict = judge(parent_answers)
    ensemble_verdict = judge(ensemble_answers)
    fusion_verdict = judge(fusion_answers)

    graph_data = draw_metrics(
        [
            naive_verdict,
            naive_semantic_verdict,
            bm25_verdict,
            cohere_verdict,
            multiquery_verdict,
            parent_verdict,
            ensemble_verdict,
            fusion_verdict,
        ],
        [
            "Naive",
            "Naive-with-Semantic",
            "BM25",
            "Contextual Compression",
            "Multi-Query",
            "Parent",
            "Ensemble",
            "Fusion",
        ],
    )

    pprint(graph_data)
    logger.info("Success")


if __name__ == "__main__":
    asyncio.run(main())
