from typing import Any, NotRequired, Tuple
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from agents.simulation_team.agent_utils import AgentState, make_agent_subgraph
from langchain_core.messages import HumanMessage
import io, contextlib, json, types
from langgraph_codeact import create_codeact, CodeActState, create_default_prompt
from langgraph.checkpoint.memory import MemorySaver
import inspect

import traceback

from tools.file_tools import (
    get_helium_void_fraction,
    count_atom_type_in_cif,
    get_unit_cell_size,
    list_directory,
    read_file,
    read_plan,
    write_summary
)

class CodeState(CodeActState):
    instructions: NotRequired[str]


def create_code_generator_agent(model, get_only_agent=False):
    # code_model = ChatOpenAI(model="gpt-4o")
    code_model = ChatOpenAI(model="gpt-5")
    code_tools = [list_directory, read_file, read_plan, write_summary, get_helium_void_fraction, count_atom_type_in_cif,get_unit_cell_size]

    code_prompt = (
    "Role: You are a code generation assistant (code_generator). You do NOT execute tools yourself. "
    "Your role is to generate Python code that will be executed in an external environment. "
    "Your code must always be fully executable directly — do NOT wrap it inside "
    "if __name__ == '__main__': blocks, functions, or classes unless explicitly requested. "
    "Always provide runnable code inside a Python code block (surrounded by triple backticks)."
    "Only make ONE code block per message, as all code within a message is executed together."
    "Note: keep in mind that output (print statements) will be cut off after the first 5000 characters.\n\n"

    "Important: Never use SystemExit, exit(), quit(), or os._exit() in your code. "
    "If a required file or parameter is missing, handle it gracefully by printing a clear message "
    "or stopping further code generation without crashing.\n\n"

    "Task: Write Python code that will copy template files, fill placeholders, "
    "and set up folder structures for molecular simulations.\n"
    "Every script you generate must re-import required modules (e.g., os, shutil, pathlib). "
    "Remember that variables persist across scripts.\n\n"

    "You can call the following tools in code (DO NOT OVERWRITE):\n\n"
)
    for c_tool in code_tools:
        code_prompt += f'''
def {c_tool.name}{str(inspect.signature(c_tool.func))}:
    """{c_tool.description}"""
    ...
'''
    code_prompt += (
    "\nInstructions:\n"
    "1. First, generate and execute code that only reads the plan using read_plan to understand simulation requirements. "
    "Do not modify anything at this step. Make sure to print the output.\n"
    "2. Based on the output of step 1, generate code to inspect the current folders and files using list_directory and read_file, "
    "focusing especially on simulation.input to identify the placeholders. Print the output, but do not modify yet.\n"
    "3. Based on the plan and folder inspection, generate code that:\n"
    "   - Copies template folders and files to the correct locations."
    "   - For each simulation folder, copy the corresponding CIF file from the 'cifs' folder into it. "
    "     The copied CIF filename must exactly match the structure name (without extension). "
    "     Do not leave placeholders, and do not skip copying CIFs.\n"
    "   - Creates **one folder for each unique combination** of structure, pressure, adsorbate, and other relevant parameters.\n"
    "   - The folder structure may be hierarchical (structure/pressure/adsorbate/) or flattened (structure_pressure_adsorbate). Follow the plan’s instructions.\n"
    "   - Fills placeholders in files (e.g., {{framework}}, {{pressure}}, {{temperature}}) according to the plan.\n"
    "   - When filling placeholders, ensure that numbers are cast to the appropriate types (e.g., int, float).\n"
    "   - The structure name in simulation.input must match the CIF filename (without extension).\n"
    "   - Create any missing folders with appropriate names.\n"
    "   - Use the following tools properly (never invent outputs):\n"
    "       - get_helium_void_fraction: Returns the helium void fraction for a zeolite topology and Al count. Use it when filling simulation.input.\n"
    "       - count_atom_type_in_cif: Returns the number of atoms of a given type in a CIF file. Use it for Al or others.\n"
    "       - get_unit_cell_size: Returns the unit cell dimensions (a, b, c). Compute required unit cells as: replicas = ceil(2 * CutOff / dim).\n"
    "   - Ensure the code is safe, idempotent, and handles missing files gracefully.\n"
    "4. If target folders are ambiguous, stop execution and clearly indicate the ambiguity.\n"
    "5. After execution, generate code to validate that files were copied and placeholders filled (check only a few samples).\n"
    "6. Update the plan using write_summary. 'write_summary' takes your name (code_generator) and a concise summary of what was executed, "
    "including which files or folders were created or modified.\n"
    "7. Report a clear summary of actions performed, including affected files and folders. **When you are done do NOT write any more code blocks**, in order to finish execution.\n"
    "8. Always call the provided tools; do not approximate their behavior.\n"
    "9. Follow the strict sequence: read plan → inspect folders → copy/fill templates → verify → update plan. "
    "Do not skip or reorder steps. Execute one step per message.\n"
)

    SAFE_BUILTINS = __builtins__  # or craft a restricted dict if needed

    
    def eval(code: str, context: dict[str, Any]) -> Tuple[str, dict[str, Any]]:
        """
        Execute code with a unified namespace (globals is locals).
        `context` should already include your tools and any prior variables.
        Returns (stdout, persisted_vars) for the next turn.
        """
        # Build a single namespace so names resolve consistently
        namespace = {"__builtins__": SAFE_BUILTINS, **context}

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, namespace, namespace)  # <— same dict for globals & locals
            output = buf.getvalue().strip() or "<code ran, no output printed>"
        except Exception as e:
            tb = traceback.format_exc().splitlines()
            # Keep first 10 lines and last 10 lines
            head, tail = tb[:10], tb[-10:]
            trimmed_tb = "\n".join(head + ["... [traceback truncated] ..."] + tail) if len(tb) > 20 else "\n".join(tb)
            output = f"Error: {repr(e)}\nTraceback:\n{trimmed_tb}"
    
        # Persist only safe/serializable user variables (avoid modules/functions/classes)
        def is_jsonable(x: Any) -> bool:
            try:
                json.dumps(x)
                return True
            except Exception:
                return False

        persisted = {}
        for k, v in namespace.items():
            if k.startswith("__"):
                continue
            if isinstance(v, (types.ModuleType, types.FunctionType, type)):
                continue
            # Drop tool bindings if you inject them each turn (recommended)
            # If you keep tools inside `context`, exclude them here by name.
            if k in context and context[k] is v and not is_jsonable(v):
                # skip unserializable carry-overs to avoid msgpack errors
                continue
            if is_jsonable(v):
                persisted[k] = v

        # limit output size
        MAX_OUTPUT = 5000
        if len(output) > MAX_OUTPUT:
            output = output[:MAX_OUTPUT] + "\n... [output truncated]"

        return output, persisted



    code_act = create_codeact(code_model, code_tools, eval, prompt=code_prompt, state_schema=CodeState,)
    code_generator = code_act.compile()

    if get_only_agent:
        return code_generator

    def code_generator_node(state: CodeState):
        new_messages = list(state.get("messages", [])) + [HumanMessage(content=state["instructions"], name="supervisor")]
        result = code_generator.invoke({"messages": new_messages})
        new_messages.extend(result["messages"])

        return {"messages": new_messages, "context":""}
    
    agent_subgraph = make_agent_subgraph(CodeState, "run", code_generator_node)

    return agent_subgraph