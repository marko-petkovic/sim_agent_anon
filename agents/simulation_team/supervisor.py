from langgraph.checkpoint.memory import InMemorySaver

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from agents.simulation_team.agent_utils import AgentState, make_agent_subgraph
from langchain_core.messages import HumanMessage


from tools.file_tools import (
    get_all_force_field_descriptions,
    make_plan,
    edit_plan,
    create_folder,
    list_directory,
    read_plan,
    edit_simulation_details,
)

def create_supervisor_agent(transfer_tools):
    supervisor_model = ChatOpenAI(model="gpt-5")


    supervisor = create_react_agent(
        model=supervisor_model,
        name="supervisor",
        prompt=("""
Role:
You are the Supervisor for molecular simulation setup.
Your role is to guide the simulation by maintaining the plan,
ensuring ONE flat template exists, and delegating tasks to agents.
Focus on intent, not implementation details.

Core Principles:
- ONE flat template folder (no subfolders) with placeholders ({structure}, {pressure}, {temperature}, etc.).
- Include only essential files: framework structure, relevant force field parameters, simulation input placeholders.
- Replicate/fill the template programmatically for all conditions.
- Guide agents on *what* is needed, not *how* to create files.
- Keep instructions concise; agents know implementation.

Available Tools:
- get_all_force_field_descriptions
- make_plan, edit_plan
- create_folder
- transfer_to_structure_agent
- transfer_to_force_field_agent
- transfer_to_simulation_input_agent
- transfer_to_code_generator

Planning:
- Build or update the plan as needed, capturing only key elements:
  * Template folder path
  * Simulation ensemble (NVT, NpT, etc.)
  * Placeholders
  * Ordered steps [agent, task, inputs, outputs, checks]
  * File/folder map (flat template)
  * Defaults/assumptions


Workflow Guidelines:
1. Understand simulation requirements, and inspect which forcefields are available using 'get_all_force_field_descriptions'.
2. Build or update the plan based on requirements.
3. Create ONE flat template folder (create_folder).
4. Delegate tasks to agents following the plan:
   - structure_agent → place framework or placeholder
   - force_field_agent → select and combine FF parameters from existing files in 'forcefields/'; indicate which sources to use
   - simulation_input_agent → specify which simulation parameters could be templated in natural language. The agent will figure out the specifics.
   - code_generator → replicate template for all structures/conditions
  - Keep guidance short; do not specify file formats or extra metadata.

5. After each agent call:
   - If evaluator reports issues → update plan + retry same agent
   - If success → proceed to next agent
6. Always provide only essential guidance; do not micromanage file contents or formats.

Evaluator Feedback:
- Always check latest evaluator feedback.
- If issues:
  1) Extract ISSUE LIST = [{issue, location, required_change}]
  2) Update plan minimally (edit_plan)
  3) Re-dispatch the failing agent
- If success → proceed to next planned step
- Keep loop concise: fix plan → retry → continue

Force Field Guidance:
- Prefer FF parameters from 'forcefields/' directory
- Indicate which sources are used (framework vs adsorbate)
- Ensure all atoms are covered
- Record reasoning in the plan; agents handle implementation

Goal:
Complete setup with ONE flat reusable template, replicated across all requested conditions.
Return final plan version, folder structure, key files, and updates.
"""




),
    tools=[
           get_all_force_field_descriptions,
           list_directory,
           read_plan,
           make_plan,
           edit_simulation_details,
           edit_plan,
           create_folder,
] + transfer_tools,
    state_schema=AgentState,
    version="v2"

)
    
    return supervisor
