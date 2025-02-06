import os
from operator import itemgetter
from langchain_core.runnables.base import RunnableSerializable
import chainlit as cl
import aiofiles
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langgraph.errors import GraphRecursionError

import messages as msg
from parsers import ParsersMap
from splitters import TextSplitter
from models import EmbeddingLLM, MiniLLM
from prompts import RAG_PROMPT
from constructors import (
    construct_research_graph,
    construct_authoring_graph,
    construct_super_graph,
    construct_correctness_graph,
)
from utils import enter_chain


@cl.on_chat_start
async def init_chat():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content=msg.INIT_MSG,
            accept=["text/plain", "application/pdf"],
        ).send()

    file = files[0]
    _, ext = os.path.splitext(file.name)
    parser = ParsersMap[ext]

    async with aiofiles.open(file.path, "rb") as f:
        file_content = await f.read()

    file_text = parser.load(file_content)
    status_msg_text = """
    Processing...please wait
    [1/4] Read file: {read_status}
    [2/4] Chunk file: {split_status}
    [3/4] Load file to DB: {load_status}
    [4/4] Build graph: {graph_status}
    """
    status_msg = cl.Message(
        status_msg_text.format(
            read_status="🟢", split_status="🔹", load_status="🔹", graph_status="🔹"
        )
    )
    await status_msg.send()

    documents = TextSplitter.split_text(file_text)
    status_msg.content = status_msg_text.format(
        read_status="🟢", split_status="🟢", load_status="🔹", graph_status="🔹"
    )
    await status_msg.send()

    db = Qdrant.from_texts(
        documents,
        EmbeddingLLM,
        location=":memory:",
        collection_name="multi-agent-chatbot",
    )
    status_msg.content = status_msg_text.format(
        read_status="🟢", split_status="🟢", load_status="🟢", graph_status="🔹"
    )
    retriever = db.as_retriever()
    await status_msg.update()

    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | rag_prompt
        | MiniLLM
        | StrOutputParser()
    )
    research_chain = (
        enter_chain | construct_research_graph(research_chain=chain).compile()
    )
    authoring_chain = enter_chain | construct_authoring_graph().compile()
    correctness_chain = enter_chain | construct_correctness_graph().compile()
    super_chain = (
        enter_chain
        | construct_super_graph(
            research_chain, authoring_chain, correctness_chain
        ).compile()
    )

    cl.user_session.set("chain", super_chain)
    status_msg.content = status_msg_text.format(
        read_status="🟢", split_status="🟢", load_status="🟢", graph_status="🟢"
    )
    await status_msg.update()

    await cl.Message(msg.ASK_FOR_QUERY_MSG).send()


@cl.on_message
async def process_chat(message: cl.Message):
    chain: RunnableSerializable = cl.user_session.get("chain")
    try:
        async for s in chain.astream(message.content, {"recursion_limit": 30}):
            if "__end__" not in s:
                await cl.Message(s).send()
    except GraphRecursionError:
        await cl.Message("Max recursion depth reached").send()
    # todo: Not finished idea to return generated file to chat
    # response = cl.Message(content="")
    # for token in result:
    #     await response.stream_token(token)
    #
    # await response.send()
    # elements = [
    #     cl.File(
    #         name="hello.py",
    #         path="./hello.py",
    #         display="inline",
    #     ),
    # ]
    # await cl.Message(
    #     content="This message has a file element", elements=elements
    # ).send()
