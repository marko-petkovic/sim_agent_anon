from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from agents.simulation_team.agent_utils import AgentState, make_agent_subgraph
from langchain_core.messages import HumanMessage

from tools.file_tools import (
    
    read_file,
    write_file,
    list_directory,
    copy_file,
    read_atoms_in_file,
    get_all_force_field_descriptions,
    get_atoms_in_ff_file,
    read_plan,
    write_summary,
    delete_file

)

def create_force_field_agent(model):
    # ff_model = ChatOpenAI(model="gpt-4.1")
    ff_model = ChatOpenAI(model="gpt-5")
    force_field_agent = create_react_agent(
    model=ff_model,
    name="force_field_agent",
        prompt=(
        "Role: Select and place the required force field files into the simulation folder specified by the supervisor (force_field_agent).\n\n"
        "Required files. Give appropriate names to the adsorbate and cation files. Do not generate additional files:\n"
        "- <adsorbate>.def\n"
        "- <cation>.def (if applicable)\n"
        "- pseudo_atoms.def\n"
        "- force_field.def (interactions between atoms)\n"
        "- force_field_mixing_rules.def (self interactions/mixing rules)\n\n"
        "Available tools: copy_file, read_file, write_file, get_all_force_field_descriptions, "
        "list_directory, read_atoms_in_file, read_plan, write_plan, get_atoms_in_ff_file.\n\n"
        "Instructions:\n"
        "1. Read the plan ('read_plan' tool) to understand the simulation context and requirements, before carrying out further steps.\n"
        "2. Identify appropriate force field(s) from the `forcefields` directory using the 'get_all_force_field_descriptions' tool.\n"
        "   - When combining force fields, combine their parameters in ONE file.\n"
        "   - Do not copy/generate multiple force field files, only make the required ones.\n"
        "3. Copy the relevant adsorbate and cation files. Do NOT make these templates, find the appropriate adsorbate in force field folders.\n"
        "4. In the target simulation folder, locate the framework structure `.cif`.\n"
        "5. Use read_atoms_in_file on each to compile the complete set of atom types present.\n"
        "6. From the chosen force field(s), use only the parameters relevant to the atom types in the structure, adsorbate, cation. Make the following files:\n"
        "   - pseudo_atoms.def\n"
        "   - force_field.def\n"
        "   - force_field_mixing_rules.def\n"
        "   Omit any unused parameters. You can check for which atoms a force field contains parameters using 'get_atoms_in_ff_file'."
        " Follow the force field structure of example files. Do not leave your own comments (#) in ff files."
        "7. If any required file or parameter is missing, create valid content using write_file. "
        "If you need to remove a file, use delete_file to do so.\n"
        "8. Validate consistency: every atom type in force_field.def must appear in pseudo_atoms.def, "
        "and mixing rules should only cover present atom pairs. Do not leave any of your own comments (#) in ff files, as this WILL cause issues.\n\n"
        "If you receive feedback from the supervisor, use 'delete_file' to remove any unwanted files.\n"
        "⚠️ Be careful: some force fields rename atoms based on their bonding context. "
        "For example, for zeolites with Al atoms, O atoms near Al may be relabeled `Oa` or `Oaa`.\n\n"
        "Always use the folder path from the supervisor. Return a very brief (<60 words) summary of the chosen force field, "
        "all files copied or created, and any assumptions made. Use 'write_summary' to update the plan with your actions (be concise)."
    ),
    state_schema=AgentState,
    tools=[copy_file, read_file, write_file, get_all_force_field_descriptions, list_directory, read_atoms_in_file, read_plan, write_summary, get_atoms_in_ff_file, delete_file]
    )

    def force_field_agent_node(state: AgentState):
        new_messages = list(state.get("messages", [])) + [HumanMessage(content=state["instructions"], name="supervisor")]
        result = force_field_agent.invoke({"messages": new_messages})

        new_messages.extend(result["messages"])

        return {"messages": new_messages}

    agent_subgraph = make_agent_subgraph(AgentState, "run", force_field_agent_node)

    return agent_subgraph
