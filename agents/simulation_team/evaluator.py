from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from agents.simulation_team.agent_utils import AgentState, make_agent_subgraph
from langchain_core.messages import HumanMessage
import json
from typing import Literal
from langgraph.types import Command

from tools.file_tools import (
    list_directory,
    read_file,
    read_plan,
    count_atom_type_in_cif,
    get_unit_cell_size, 
    read_atoms_in_file,
    get_atoms_in_ff_file,
    list_example_simulation_inputs
)

evaluator_message = {

    "structure_agent_node":(
        "Checks:\n"
        "1. .cif in the correct folder.\n"
        "2. Files are correctly named and valid.\n"
        "3. Files match the framework in the task."
    ),

    "force_field_agent_node": (
        "Checks:\n"
        "1. Relevant .def files for cations and adsorbates are present.\n"
        "2. force_field.def, force_field_mixing_rules.def, and pseudo_atoms.def are in the correct folder.\n"
        "3. Files are correctly named and valid.\n"
        "4. Files contain parameters for all relevant atoms (framework, adsorbates, cations).\n"
        "5. Number of defined interactions matches the expected count."
    ),

    "simulation_input_agent_node": (
        "Checks:\n"
        "1. simulation.input is in the correct folder.\n"
        "2. Decision made about placeholders (for example, {{pres}}) are sound.\n"
        "3. Unit cells >= 24 Å in each direction if not replaced by a placeholder (get_unit_cell_size).\n"
        "4. Check that relevant fields are used (for example, for muVT, verify ExternalPressure is a field) \n"
        "5. If applicable, correct number of cations (or placeholder) present based on unit cells and .cif. Use count_atom_type_in_cif to verify.\n"
        "6. Check that adsorbates and cations are named the same in simulation.input and the template folder. (For example, H2O in simulation.input, and H2O.def in the template folder)\n"
        "7. No unnecessary fields are created in the file.\n"

    "Keep in mind, this agent is highly specified. Do not mark things you do not understand as wrong."

    
    ),

    "code_generator_node": (
        "Checks:\n"
        "1. Generated code performed the intended tasks (copying files, filling templates, creating folders).\n"
        "2. If the code requested clarification, ensure it was forwarded appropriately to the supervisor."
    ),
    

}


def get_current_agent_summary(agent_name):
    with open("plan.json", "r") as file:
        plan = json.load(file)
    return plan[agent_name].get("summary", "")


def create_evaluator(model):
    evaluator_model = ChatOpenAI(model="gpt-5")
    evaluator = create_react_agent(
    model=evaluator_model,
    name="evaluator",
    prompt=(
    "Role: You are an evaluator in a multi-agent system for RASPA molecular simulation setup.\n\n"
    "Instructions:\n"
    "1. First, read the instruction the agent as well as the agents summary of their task.\n"
    "2. If the agent performed a trivial task (such as copying a file), acknowledge it and move on.\n"
    "3. Otherwise, evaluate the assigned agent’s execution strictly based on facts:\n"
    "   - Compare the agent’s reported actions to the plan.\n"
    "   - Use list_directory to confirm folders exist.\n"
    "   - Use available tools to check file contents without reading (read_atoms_in_file, count_atom_type_in_cif, get_unit_cell_size, get_atoms_in_ff_file).\n"
    "   - Only use read_file for 'simulation.input'"
    "   - Do not verify CIFs, adsorbates, or the origin of files.\n"
    "4. Follow the specific checks provided in the accompanying message.\n"
    "5. Respond fully factually:\n"
    "   - If correct, reply only: good execution by \"agent_name\".\n"
    "   - If incorrect, state exactly what is wrong or missing.\n"
    ),
    tools=[list_directory, read_file, read_atoms_in_file, count_atom_type_in_cif,read_plan,get_unit_cell_size,list_example_simulation_inputs, get_atoms_in_ff_file],
    state_schema=AgentState
)
    
    def evaluator_node(state: AgentState) -> Command[Literal["supervisor"]]:
        agent_name = state["current_agent"]

        summary = get_current_agent_summary(agent_name[:-5])

        messages = {"messages": [HumanMessage(content=evaluator_message[agent_name], name="instructions"), HumanMessage(content=summary, name=agent_name[:-5])]}

        result = evaluator.invoke(messages)
        return {"messages": [
            HumanMessage(content=result["messages"][-1].content, name="evaluator")
        ]}
    return evaluator_node