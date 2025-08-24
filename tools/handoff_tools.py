from typing import List, Annotated

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState
from langgraph.types import Command, Send

def create_handoff_tool(
    *, agent_name: str, sender_name: str, description: str | None = None
):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the supervisor LLM
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        # these parameters are ignored by the LLM
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        # task_description_message = {"role": "user", "content": task_description}
        task_description_message = HumanMessage(
            content=task_description,
            name=sender_name,
        )
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,)
        
        # handoff_messages = state["messages"] + [tool_message, task_description_message]
        handoff_messages = state["messages"] + [tool_message]

        # agent_input = {"messages": [task_description_message]}
        agent_input = {"instructions": task_description}
        return Command(
            goto=Send(agent_name, agent_input),
            graph=Command.PARENT,
                update={**state, "messages": handoff_messages, "current_agent": agent_name},

        )
    return handoff_tool


def create_research_handoff_tool(*, agent_name: str, description: str | None = None, sender: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer to {agent_name}"

    @tool(name, description=description)
    def handoff_tool(
        task_description: Annotated[
            str,
            "Information/context you want to share with the next agent. Should be brief (less than 50 words)",
        ],
        state: Annotated[MessagesState, InjectedState], 
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }

        task_description_message = {
            "role": "user",
            "content": task_description,
            "name": sender,
        }

        return Command(
            goto=agent_name,
            update={"messages": [state["messages"][-1]] + [tool_message]},
            graph=Command.PARENT,
        )
    return handoff_tool

