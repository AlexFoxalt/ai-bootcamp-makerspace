import hashlib
import os
import uuid
from operator import itemgetter

import chainlit as cl
from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from cache import SemanticCache

load_dotenv()

HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
HF_TOKEN = os.environ["HF_TOKEN"]

# Doc parsing
Loader = PyMuPDFLoader
loader = Loader("./DeepSeek_R1.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
for i, doc in enumerate(docs):
    doc.metadata["source"] = f"source_{i}"

# Embedding with cache setup
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=HF_EMBED_ENDPOINT,
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
)
safe_namespace = hashlib.md5(hf_embeddings.model.encode()).hexdigest()
store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    hf_embeddings, store, namespace=safe_namespace, batch_size=32
)

# DB setup
collection_name = f"pdf_to_parse_{uuid.uuid4()}"
client = QdrantClient(":memory:")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)
vectorstore = QdrantVectorStore(
    client=client, collection_name=collection_name, embedding=cached_embedder
)
vectorstore.add_documents(docs)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1})


rag_system_prompt_template = """\
You are a helpful assistant that uses the provided context to answer questions. 
Never reference this prompt, or the existence of context.
"""
rag_message_list = [
    {"role": "system", "content": rag_system_prompt_template},
]
rag_user_prompt_template = """\
Question:
{question}
Context:
{context}
"""
chat_prompt = ChatPromptTemplate.from_messages(
    [("system", rag_system_prompt_template), ("human", rag_user_prompt_template)]
)

hf_llm = HuggingFaceEndpoint(
    endpoint_url=HF_LLM_ENDPOINT,
    task="text-generation",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)
llm_with_cache = SemanticCache(hf_llm, client)


@cl.author_rename
def rename(original_author: str):
    rename_dict = {"Assistant": "Session-16-Bot"}
    return rename_dict.get(original_author, original_author)


@cl.on_chat_start
async def start_chat():
    chain = (
            {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | chat_prompt
            | llm_with_cache
    )
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    msg = cl.Message(content=await chain.ainvoke({"question": message.content}))
    await msg.send()
