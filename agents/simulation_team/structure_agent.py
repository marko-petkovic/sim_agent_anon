from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from agents.simulation_team.agent_utils import AgentState, make_agent_subgraph
from langchain_core.messages import HumanMessage

from tools.file_tools import (
    read_plan,
    write_summary,
    read_file,
    write_file,
    list_directory,
    copy_file,
    list_example_runs,
    delete_file
)


def create_structure_agent(model):
    struct_model = ChatOpenAI(model="gpt-5-mini")
    structure_agent = create_react_agent(
    model=struct_model,
    name="structure_agent",
        prompt=(
                "Role: Place the required `.cif` (ONE) structure file into the simulation folder specified by the supervisor (structure_agent).\n\n"
        "Available tools: copy_file, read_file, write_file, list_example_runs, "
        "list_directory, read_plan, write_plan.\n\n"
        "Instructions:\n"
        "1. Read the plan to understand the required structure and context.\n"
        "2. Search for the structure file in the specified directories. Do not read the file. If there are multiple structures, select ONE as placeholder.\n"
        "3. If found, copy it to the target folder. Only copy one representative structure.\n"
        "4. If no example exists, create a minimal placeholder `.cif` file and clearly state that it is a placeholder.\n\n"
        "Always use the folder path given by the supervisor. Return a summary of all files placed or created, including file names and target folder.\n"
        "If you receive feedback from the supervisor, use 'delete_file' to remove any unwanted files.\n"
        "Use 'write_summary' to update the plan with your actions (be concise)."
    ),
    state_schema=AgentState,
    tools=[copy_file, read_file, write_file, list_example_runs, list_directory, read_plan, write_summary, delete_file]
)
    def structure_agent_node(state: AgentState):
        new_messages = list(state.get("messages", [])) + [HumanMessage(content=state["instructions"], name="supervisor")]
        result = structure_agent.invoke({"messages": new_messages})

        new_messages.extend(result["messages"])

        return {"messages": new_messages}

    agent_subgraph = make_agent_subgraph(AgentState, "run", structure_agent_node)

    return agent_subgraph