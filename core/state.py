from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage

class ResearchState(TypedDict):
    """The state of the Deep Research graph."""
    messages: Annotated[List[BaseMessage], operator.add]
    research_topic: str
    plan: List[str]
    tasks_completed: List[str]
    search_context: Annotated[List[str], operator.add]
    draft_report: str
    fact_check_feedback: str
    is_valid: bool
    iteration_count: int
