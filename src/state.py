"""
Shared State for the Multi-Agent System (Plan-and-Execute architecture).
"""
from typing import Annotated, Literal, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MultiAgentState(TypedDict):
    """
    Shared state with planning support.

    Attributes:
        messages: Conversation history (auto-appended)
        current_task: The original user request
        task_status: Current status of the task
        iteration_count: Loop counter for safety
        plan: Ordered list of agents to execute, e.g. ["researcher", "coder", "reviewer"]
        current_step: Index of the current step in the plan
        last_output: Output of the last executed agent (context passing)
        researcher_output: Pure research findings
        coder_output: Pure Python code output
        reviewer_output: Optional[str]
    """
    messages: Annotated[list[BaseMessage], add_messages]
    current_task: str
    task_status: Literal["planning", "executing", "completed", "failed"]
    iteration_count: int
    plan: list[str]
    current_step: int
    last_output: str
    researcher_output: str
    coder_output: str
    reviewer_output: Optional[str]