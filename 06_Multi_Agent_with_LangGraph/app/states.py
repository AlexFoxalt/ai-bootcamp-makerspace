import operator
from typing import TypedDict

from langchain_core.messages import BaseMessage
from typing import Annotated, List


class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str


class DocWritingState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    current_files: str


class CorrectnessState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str
