from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from agents.simulation_team.agent_utils import AgentState, make_agent_subgraph
from langchain_core.messages import HumanMessage

from tools.file_tools import (
    get_unit_cell_size,
    read_plan,
    write_summary,
    read_file,
    write_file,
    list_directory,
    count_atom_type_in_cif,
    list_example_simulation_inputs,
    copy_file
)



def create_simulation_input_agent(model):
    # sim_input_model = ChatOpenAI(model="gpt-4o", temperature=0.2)
    # sim_input_model = ChatOpenAI(model="gpt-4o")
    sim_input_model = ChatOpenAI(model="gpt-5")
    simulation_input_agent = create_react_agent(
    model=sim_input_model,
    name="simulation_input_agent",
    prompt=(
        "Role: Write the `simulation.input` file in the folder specified by the supervisor (simulation_input_agent).\n\n"
        "Available tools: read_file, write_file, list_directory, "
        "count_atom_type_in_cif, list_example_simulation_inputs, copy_file, read_plan, write_plan.\n\n"
        "Instructions:\n"
        "1. Read the plan to understand the simulation type and requirements.\n"
        "2. Browse known formats in `raspa_examples/simulation_input/` using list_example_simulation_inputs and read_file.\n"
        "   - Make sure to read multiple files to get a comprehensive understanding of the format.\n"
        "   - Based on the example descriptions, identify which parameters are required for the current simulation type (Helium void fraction, pressure, temperature etc.).\n"
        "3. Adapt the template to the current simulation:\n"
        "   - Verify the structure and adsorbate files in the target folder using list_directory and read_file.\n"
        "   - Adjust units, file references, component names, etc., as needed.\n"
        "   - Ensure the simulation box is at least 24 Å in each direction, by using 'get_unit_cell_size'.\n"
        "   - Include only placeholders required for this simulation type (e.g., {{pres}} for μVT; omit {{temp}} if not needed).\n"
        "   - The amount of unit cells should be written as UnitCells a b c\n"
        "5. Write the final file using write_file, named `simulation.input` in the correct folder.\n"
        "6. If the same file is reused, write it once and use copy_file to duplicate.\n\n"
        "⚠️ Ensure the file is syntactically valid and consistent with the structure, adsorbate, and simulation type.\n"
        "Do not leave your own comments in the file.\n\n"
        "Return a short summary of the file written, including which example it was based on and what values need to be filled in when templating (<60 words)."
        "Use 'write_summary' to update the plan with your actions (<60 words)."
    ),
    state_schema=AgentState,
    tools=[
        # get_all_example_metadata,
           read_file, 
           write_file, 
           list_directory, 
           count_atom_type_in_cif, 
           get_unit_cell_size,
           list_example_simulation_inputs, 
           copy_file,  
           read_plan, 
           write_summary]
)


    def simulation_input_agent_node(state: AgentState):
        new_messages = list(state.get("messages", [])) + [HumanMessage(content=state["instructions"], name="supervisor")]
        result = simulation_input_agent.invoke({"messages": new_messages})

        new_messages.extend(result["messages"])

        return {"messages": new_messages}

    agent_subgraph = make_agent_subgraph(AgentState, "run", simulation_input_agent_node)

    return agent_subgraph