from typing_extensions import Annotated, NotRequired, TypedDict
from langgraph.managed import RemainingSteps
from langgraph.checkpoint.memory import InMemorySaver
from typing import Sequence
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.types import Command, Send
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    remaining_steps: NotRequired[RemainingSteps]

    instructions: NotRequired[str]

    current_agent: NotRequired[str]

    last_msg: NotRequired[str]


def make_agent_subgraph(state_cls, node_name, agent_node):

    def emit_node(state: AgentState) -> AgentState:
        last_msg = [state["messages"][-1]]
        return Command(
            graph=Command.PARENT,
            update={"messages": last_msg}
        )

    sg = StateGraph(state_cls)
    sg.add_node(node_name, agent_node)
    sg.add_node("emit", emit_node)
    sg.add_edge(node_name, "emit")
    sg.set_entry_point(node_name)
    return sg.compile(checkpointer=InMemorySaver())