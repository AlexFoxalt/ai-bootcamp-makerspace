import functools

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableSerializable
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from models import TurboLLM, MiniLLM
from states import ResearchTeamState, DocWritingState, State, CorrectnessState
from tools import (
    TaviliTool,
    write_document,
    edit_document,
    read_document,
    create_outline,
)
from utils import agent_node, prelude, get_last_message, join_graph


def construct_agent(
    llm: ChatOpenAI, tools: list, system_prompt: str, members: list
) -> AgentExecutor:
    """Create a function-calling agent and add it to the graph."""
    system_prompt += (
        "\nWork autonomously according to your specialty, using the tools available to you."
        " Do not ask for clarification."
        " Your other team members (and other teams) will collaborate with you with their own specialties."
        " You are chosen for a reason! You are one of the following team members: {team_members}."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    ).partial(team_members=", ".join(members))
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def construct_supervisor(
    llm: ChatOpenAI, system_prompt: str, members: list
) -> RunnableSerializable:
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )
    return chain


def construct_research_graph(research_chain: RunnableSerializable) -> StateGraph:
    retrieve_information = Tool(
        name="RetrieveInformationTool",
        func=lambda query: research_chain.invoke({"question": query}),
        description="Use Retrieval Augmented Generation to retrieve information from the file provided by user.",
    )
    research_agent = construct_agent(
        MiniLLM,
        [retrieve_information],
        "You are a research assistant who can provide specific information on the provided by user file."
        "You must only respond with information about the paper related to the request.",
        ["Search", "FileDataRetriever"],
    )
    research_node = functools.partial(
        agent_node, agent=research_agent, name="FileDataRetriever"
    )

    search_agent = construct_agent(
        MiniLLM,
        [TaviliTool],
        "You are a research assistant who can search for up-to-date info using the Tavily search engine.",
        ["Search", "FileDataRetriever"],
    )
    search_node = functools.partial(agent_node, agent=search_agent, name="Search")

    supervisor = construct_supervisor(
        TurboLLM,
        (
            "You are a supervisor tasked with managing a conversation between the following workers:\n"
            "{team_members}\n"
            "Given the following user request, determine the subject to be researched and respond with the worker to act next.\n"
            "Each worker will perform a task and respond with their results and status.\n"
            "You should never ask your team to do anything beyond research. They are not required to write content or posts."
            "You should only pass tasks to workers that are specifically research focused.\n"
            "In most cases you always should use FileDataRetriever, "
            "because it's core feature of app that will provide the most useful context.\n"
            "When finished, respond with FINISH."
        ),
        ["Search", "FileDataRetriever"],
    )
    graph = StateGraph(ResearchTeamState)
    graph.add_node("Search", search_node)
    graph.add_node("FileDataRetriever", research_node)
    graph.add_node("supervisor", supervisor)
    graph.add_edge("Search", "supervisor")
    graph.add_edge("FileDataRetriever", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "Search": "Search",
            "FileDataRetriever": "FileDataRetriever",
            "FINISH": END,
        },
    )
    graph.set_entry_point("supervisor")
    return graph


def construct_authoring_graph() -> StateGraph:
    doc_writer_agent = construct_agent(
        MiniLLM,
        [write_document, edit_document, read_document],
        (
            "You are an expert writing technical LinkedIn posts.\n"
            "Below are files currently in your directory:\n{current_files}"
        ),
        ["DocWriter", "NoteTaker", "DopenessEditor", "CopyEditor"],
    )
    context_aware_doc_writer_agent = prelude | doc_writer_agent
    doc_writing_node = functools.partial(
        agent_node, agent=context_aware_doc_writer_agent, name="DocWriter"
    )

    note_taking_agent = construct_agent(
        MiniLLM,
        [create_outline, read_document],
        (
            "You are an expert senior researcher tasked with writing a LinkedIn post outline and"
            " taking notes to craft a LinkedIn post.\n{current_files}"
        ),
        ["DocWriter", "NoteTaker", "DopenessEditor", "CopyEditor"],
    )
    context_aware_note_taking_agent = prelude | note_taking_agent
    note_taking_node = functools.partial(
        agent_node, agent=context_aware_note_taking_agent, name="NoteTaker"
    )

    copy_editor_agent = construct_agent(
        MiniLLM,
        [write_document, edit_document, read_document],
        (
            "You are an expert copy editor who focuses on fixing grammar, spelling, and tone issues\n"
            "Below are files currently in your directory:\n{current_files}"
        ),
        ["DocWriter", "NoteTaker", "DopenessEditor", "CopyEditor"],
    )
    context_aware_copy_editor_agent = prelude | copy_editor_agent
    copy_editing_node = functools.partial(
        agent_node, agent=context_aware_copy_editor_agent, name="CopyEditor"
    )

    dopeness_editor_agent = construct_agent(
        MiniLLM,
        [write_document, edit_document, read_document],
        (
            "You are an expert in dopeness, litness, coolness, etc - you edit the document to make sure it's dope. Make sure to use a number of emojis."
            "Below are files currently in your directory:\n{current_files}"
        ),
        ["DocWriter", "NoteTaker", "DopenessEditor", "CopyEditor"],
    )
    context_aware_dopeness_editor_agent = prelude | dopeness_editor_agent
    dopeness_node = functools.partial(
        agent_node, agent=context_aware_dopeness_editor_agent, name="DopenessEditor"
    )

    supervisor = construct_supervisor(
        TurboLLM,
        (
            "You are a supervisor tasked with managing a conversation between the"
            " following workers: {team_members}. You should always verify the technical"
            " contents after any edits are made. "
            "Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When each team is finished,"
            " you must respond with FINISH."
        ),
        ["DocWriter", "NoteTaker", "DopenessEditor", "CopyEditor"],
    )

    graph = StateGraph(DocWritingState)
    graph.add_node("DocWriter", doc_writing_node)
    graph.add_node("NoteTaker", note_taking_node)
    graph.add_node("CopyEditor", copy_editing_node)
    graph.add_node("DopenessEditor", dopeness_node)
    graph.add_node("supervisor", supervisor)

    graph.add_edge("DocWriter", "supervisor")
    graph.add_edge("NoteTaker", "supervisor")
    graph.add_edge("CopyEditor", "supervisor")
    graph.add_edge("DopenessEditor", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "DocWriter": "DocWriter",
            "NoteTaker": "NoteTaker",
            "CopyEditor": "CopyEditor",
            "DopenessEditor": "DopenessEditor",
            "FINISH": END,
        },
    )

    graph.set_entry_point("supervisor")
    return graph


def construct_correctness_graph() -> StateGraph:
    style_agent = prelude | construct_agent(
        MiniLLM,
        [edit_document, read_document],
        (
            "You are an expert in analyzing LinkedIn posts.\n"
            "Please verify the produced paper fits the theme and style of selected social media platform.\n"
            "Make edits to the file to make it perfect for posting on social media.\n"
            "Below are files currently in your directory:\n"
            "{current_files}\n"
        ),
        ["StyleChecker", "EthicChecker", "FactChecker"],
    )
    style_node = functools.partial(agent_node, agent=style_agent, name="StyleChecker")
    ethic_agent = prelude | construct_agent(
        MiniLLM,
        [edit_document, read_document],
        (
            "You are an expert in analyzing LinkedIn posts.\n"
            "Please verify the produced paper does not violate the platform rules, "
            "does not offend anyone and will not cause moral damage to anyone\n"
            "Make edits to the file to make it perfect for posting on social media.\n"
            "Below are files currently in your directory:\n"
            "{current_files}\n"
        ),
        ["StyleChecker", "EthicChecker", "FactChecker"],
    )
    ethic_node = functools.partial(agent_node, agent=ethic_agent, name="EthicChecker")
    fact_agent = prelude | construct_agent(
        MiniLLM,
        [edit_document, read_document, TaviliTool],
        (
            "You are an expert in analyzing LinkedIn posts.\n"
            "Please verify the produced paper corresponds to reality, "
            "there is no false or distorted information, "
            "that there is no obvious slander.\n"
            "For fact-checking, use the Tavili search engine, "
            "to which you have access in the form of a tool.\n"
            "Make edits to the file to make it perfect for posting on social media.\n"
            "Below are files currently in your directory:\n"
            "{current_files}\n"
        ),
        ["StyleChecker", "EthicChecker", "FactChecker"],
    )
    fact_node = functools.partial(agent_node, agent=fact_agent, name="FactChecker")
    supervisor = construct_supervisor(
        TurboLLM,
        (
            "You are a supervisor tasked with managing a conversation between the following workers: {team_members}.\n"
            "You should always verify the technical contents after any edits are made.\n"
            "Try to use the maximum number of workers, because each of them significantly affects the quality of "
            "the generated response and the rule 'The More, The Better' works here, "
            "so if you are not sure which workers to choose, choose all of them\n"
            "Given the following user request, respond with the worker to act next.\n"
            "Each worker will perform a task and respond with their results and status.\n"
            "When each team is finished, you must respond with FINISH.\n"
        ),
        ["StyleChecker", "EthicChecker", "FactChecker"],
    )
    graph = StateGraph(CorrectnessState)
    graph.add_node("StyleChecker", style_node)
    graph.add_node("EthicChecker", ethic_node)
    graph.add_node("FactChecker", fact_node)
    graph.add_node("supervisor", supervisor)

    graph.add_edge("StyleChecker", "supervisor")
    graph.add_edge("EthicChecker", "supervisor")
    graph.add_edge("FactChecker", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "StyleChecker": "StyleChecker",
            "EthicChecker": "EthicChecker",
            "FactChecker": "FactChecker",
            "FINISH": END,
        },
    )

    graph.set_entry_point("supervisor")
    return graph


def construct_super_graph(
    research_chain: RunnableSerializable,
    authoring_chain: RunnableSerializable,
    correctness_chain: RunnableSerializable,
) -> StateGraph:
    supervisor_node = construct_supervisor(
        TurboLLM,
        "You are a supervisor tasked with managing a conversation between the"
        " following teams: {team_members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When all workers are finished,"
        " you must respond with FINISH.",
        ["Research team", "LinkedIn team", "Correctness team"],
    )
    super_graph = StateGraph(State)
    super_graph.add_node(
        "Research team", get_last_message | research_chain | join_graph
    )
    super_graph.add_node(
        "LinkedIn team", get_last_message | authoring_chain | join_graph
    )
    super_graph.add_node(
        "Correctness team", get_last_message | correctness_chain | join_graph
    )
    super_graph.add_node("supervisor", supervisor_node)

    super_graph.add_edge("Research team", "supervisor")
    super_graph.add_edge("LinkedIn team", "supervisor")
    super_graph.add_edge("Correctness team", "supervisor")

    super_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "LinkedIn team": "LinkedIn team",
            "Research team": "Research team",
            "Correctness team": "Correctness team",
            "FINISH": END,
        },
    )
    super_graph.set_entry_point("supervisor")
    return super_graph
