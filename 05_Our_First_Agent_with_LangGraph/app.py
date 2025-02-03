from typing import TypedDict, Annotated

import chainlit as cl
from dotenv import load_dotenv
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain.tools import tool

load_dotenv()


system_prompt = """
You are a highly intelligent AI assistant designed to provide helpful, accurate, and well-structured responses to user queries. 
You use Retrieval-Augmented Generation (RAG) to incorporate relevant external knowledge. Follow these rules:

1. **Use Retrieved Knowledge First**: 
   - When external documents are available, prioritize them.
   - Summarize instead of copying text.
   - If sources conflict, highlight differences.

2. **Fallback to General Knowledge**: 
   - If no retrieved data, use your own knowledge.
   - If uncertain, clarify the limitation.

3. **Provide Clear, Structured Responses**: 
   - Use bullet points, step-by-step formats, and concise explanations.
   - Provide citations where needed.

4. **Maintain Context & Engagement**: 
   - Keep track of prior conversation.
   - If a query is unclear, ask for clarification.

5. **Security & Ethics**: 
   - Avoid harmful, illegal, or biased content.
   - Never ask for personal information.
"""


class State(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "action"
    return END


def parse_output(input_state):
    return input_state["messages"][-1].content


model = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

wiki_desc = """
Use it only if the query is about history.
A wrapper around Wikipedia.
Useful for when you need to answer general questions about 
people, places, companies, facts, historical events, or other subjects.
Input should be a search query.
"""
wiki = WikipediaQueryRun(
    description=wiki_desc,
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100),
)

tav_desc = """
Use it only if the query is about news.
A search engine optimized for comprehensive, accurate, and trusted results.
Useful for when you need to answer questions about current events.
Input should be a search query.
"""
tav = TavilySearchResults(description=tav_desc, max_results=1)

arx_desc = """
Use it only if the query is about science.
A wrapper around Arxiv.org
Useful for when you need to answer questions about Physics, Mathematics, 
Computer Science, Quantitative Biology, Quantitative Finance, Statistics, 
Electrical Engineering, and Economics from scientific articles on arxiv.org.
Input should be a search query.
"""
arx = ArxivQueryRun(description=arx_desc)


@tool("IdiotQueryRun", return_direct=True)
def idiot_query_run(query: str) -> str:
    """
    Use it only if the query is about strange or unreal things.
    A wrapper around memes from internet.
    Input should be a search query.
    """
    return "Don't believe everything you read in internet. (c)Albert Einstein"


tool_belt = [wiki, tav, arx, idiot_query_run]

model = model.bind_tools(tool_belt)
tool_node = ToolNode(tool_belt)


uncompiled_graph = StateGraph(State)
uncompiled_graph.add_node("agent", call_model)
uncompiled_graph.add_node("action", tool_node)
uncompiled_graph.set_entry_point("agent")
uncompiled_graph.add_conditional_edges("agent", should_continue)
uncompiled_graph.add_edge("action", "agent")
graph = uncompiled_graph.compile()


@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content="Hello! Please ask your questions.")
    await msg.send()


@cl.on_message
async def main(message):
    inputs = {"messages": [SystemMessage(system_prompt), HumanMessage(message.content)]}
    messages = graph.invoke(inputs)["messages"]
    response = messages[-1].content
    if isinstance(messages[-2], ToolMessage):
        response += f"\n\nThis response was generated using `{messages[-2].name}` tool"
    msg = cl.Message(content="")
    for i in response:
        await msg.stream_token(i)
    await msg.send()
