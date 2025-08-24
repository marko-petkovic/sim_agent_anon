from agents.simulation_team.supervisor import create_supervisor_agent
from agents.simulation_team.structure_agent import create_structure_agent
from agents.simulation_team.force_field_agent import create_force_field_agent
from agents.simulation_team.simulation_input_agent import create_simulation_input_agent
from agents.simulation_team.code_generator import create_code_generator_agent
from agents.simulation_team.evaluator import create_evaluator

from tools.handoff_tools import create_handoff_tool
from agents.simulation_team.agent_utils import AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, MessagesState, END


transfer_to_structure_agent = create_handoff_tool(
    agent_name="structure_agent_node",
    sender_name="supervisor",
    description="Transfer to the Structure Agent to handle placement of the framework (.cif) file."
)

transfer_to_force_field_agent = create_handoff_tool(
    agent_name="force_field_agent_node",
    sender_name="supervisor",
    description="Transfer to the Force Field Agent to handle, cations (optional), adsorbates, force field and pseudo atoms definitions."
)

transfer_to_simulation_input_agent = create_handoff_tool(
    agent_name="simulation_input_agent_node",
    sender_name="supervisor",
    description="Transfer to the Simulation Input Agent to create simulation input template."
)

transfer_to_code_generator = create_handoff_tool(
    agent_name="code_generator_node",
    sender_name="supervisor",
    description="Transfer to the Code Generator to handle code generation and adaptation.\n" \
    "Provide clear instructions for what the code should copy or modify, including the target folder and any specific parameters.\n" \
)

transfer_tools = [transfer_to_structure_agent, transfer_to_force_field_agent, transfer_to_simulation_input_agent, transfer_to_code_generator]

def create_simulation_team():
    supervisor = create_supervisor_agent(transfer_tools)
    structure_graph = create_structure_agent(None)
    ff_graph = create_force_field_agent(None)
    si_graph = create_simulation_input_agent(None)
    cg_graph = create_code_generator_agent(None)
    evaluator_node = create_evaluator(None)

    supervisor_memory = InMemorySaver()
    supervisor_graph = (StateGraph(AgentState)
                    .add_node(supervisor, destinations=("structure_agent_node", "force_field_agent_node", "simulation_input_agent_node", "code_generator_node"))
                    .add_node("structure_agent_node", structure_graph)
                    .add_node("force_field_agent_node", ff_graph)
                    .add_node("simulation_input_agent_node", si_graph)
                    .add_node("code_generator_node", cg_graph)
                    .add_node("evaluator_node", evaluator_node)
                    .add_edge("structure_agent_node", "evaluator_node")
                    .add_edge("force_field_agent_node", "evaluator_node")
                    .add_edge("simulation_input_agent_node", "evaluator_node")
                    .add_edge("code_generator_node", "evaluator_node")
                    .add_edge("evaluator_node", "supervisor")
                    .add_edge(START, "supervisor")
                    .compile(checkpointer=supervisor_memory))
    
    return supervisor_graph