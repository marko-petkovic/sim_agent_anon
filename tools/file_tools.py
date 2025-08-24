import json
import os
import shutil

from pathlib import Path
from typing import Annotated, List, Dict

from langchain.agents import tool


# Define root folders
EXAMPLES_DIR = Path("example_runs")
RUNS_DIR = Path("runs")


def list_example_runs_func() -> List[str]:
    """List all available example run folders."""
    return sorted([f.name for f in EXAMPLES_DIR.iterdir() if f.is_dir()])

def list_force_fields_func() -> List[str]:
    """List all available force field definition files."""
    return sorted([f.name for f in Path("forcefields").iterdir() if f.is_dir()])

@tool
def list_example_runs() -> List[str]:
    """get the folder and file structure of the example runs."""
    startpath = str(EXAMPLES_DIR)
    output = []
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        output.append(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            output.append(f'{subindent}{f}')
    return '\n'.join(output)

@tool
def list_directory(folder: str):
    """Return a string representation of the folder and file structure of a given directory (folder)."""
    
    # if the agent tries to access '.', we need to tell them its not allowed
    if folder in [".", ""]:
        return "Accessing the current directory is not allowed."

    start_path = str(folder)
    output = []
    x = 0
    
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        output.append(f'{indent}{os.path.basename(root)}/')
        x += 1
        if x > 50:
            output.append(f'{indent}... and more')
            return '\n'.join(output)
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            output.append(f'{subindent}{f}')
            x += 1
            if x > 50:
                output.append(f'{indent}... and more')
                return '\n'.join(output)
        
    return '\n'.join(output)


def read_description(run_folder: str) -> str:
    """Read the markdown description of a given example run."""
    path = EXAMPLES_DIR / run_folder / "description.md"
    return path.read_text()

@tool
def read_file(folder_path: str, filename: str) -> str:
    """Read a file (e.g. simulation.input or force_field.def) from the given folder path."""
    path = Path(folder_path) / filename
    return path.read_text()

@tool
def read_atoms_in_file(folder_path: str, filename: str) -> List[str]:
    """Read unique atoms from a structure/adsorbate/cation file (.cif or .def) in the given folder path."""
    path = Path(folder_path) / filename
    
    if filename.endswith('.cif'):
        with open(path, 'r') as file:
            lines = file.readlines()
        atoms = set()
        structure_started = False
        for line in lines:
            if line.startswith("_atom_site"):
                structure_started = True
                continue
            elif structure_started and line.startswith("loop_"):
                structure_started = False
                continue
            elif structure_started:
                parts = line.split()
                if len(parts) > 2:
                    atoms.add(parts[1])
    elif filename.endswith('.def'):
        with open(path, 'r') as file:
            lines = file.readlines()
        atoms = set()
        atoms_started = False
        for line in lines:
            if line.startswith("# atomic positions"):
                atoms_started = True
                continue
            elif atoms_started and line.startswith("#"):
                atoms_started = False
                break
            elif atoms_started:
                parts = line.split()
                if len(parts) > 1:
                    atoms.add(parts[1])

    return atoms

@tool
def delete_file(folder_path: str, filename: str) -> str:
    """Delete a file from the given folder path. WARNING: This action is irreversible."""
    path = Path(folder_path) / filename
    if path.exists():
        path.unlink()
        return f"File {filename} deleted from {folder_path}."
    return f"File {filename} not found in {folder_path}."

@tool
def write_file(folder_path: str, filename: str, content: str) -> str:
    """Write a file to the given folder path."""
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_text(content)
    return f"File {filename} written to {folder_path}."

@tool
def create_folder(folder_path: str):
    """Create a new folder for a run."""
    run_path = folder_path
    os.makedirs(run_path, exist_ok=True)
    return f"Folder {folder_path} created."

@tool
def delete_folder(folder_path: str):
    """Delete a folder and all its contents."""
    shutil.rmtree(folder_path, ignore_errors=True)
    return f"Folder {folder_path} deleted."

@tool
def get_all_example_metadata() -> List[Dict]:
    """Return metadata from all descriptions (can later be extended with embeddings)."""
    all_data = []
    for run_name in list_example_runs_func():
        desc = read_description(run_name)
        all_data.append({
            "name": run_name,
            "description": desc
        })
    return all_data

@tool
def get_all_force_field_descriptions() -> List[Dict]:
    """Return metadata from all force field description files."""
    all_data = []
    for force_field_name in list_force_fields_func():
        desc = (Path("forcefields") / force_field_name / "description.md").read_text()
        all_data.append({
            "name": force_field_name,
            "description": desc
        })
    return all_data

@tool
def copy_file(src: str, dst_folder: str, dst_name: str = None) -> str:
    """Copy a file to a run folder, optionally renaming it."""
    src = Path(src)
    dst_folder = Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)
    dst_path = dst_folder / (dst_name or src.name)
    shutil.copy(src, dst_path)
    return f"Copied {src} to {dst_path}"

@tool
def list_example_simulation_inputs() -> List[str]:
    """List all example simulation input files."""
    startpath = Path("raspa_examples/simulation_input")
    names = sorted([f.name for f in startpath.iterdir() if f.is_file()])

    out_str = ""

    for name in names:
        if name.endswith(".md"):
            continue

        md = name.replace(".input", ".md")
        # check if description exists
        md_content = Path("raspa_examples/simulation_input") / md
        if md_content.exists():
            with open(md_content, 'r') as file:
                md_description = file.read()
                out_str += f"- {name}: {md_description}\n"
        else:
            out_str += f"- {name}: No description\n"

    return out_str



@tool
def count_atom_type_in_cif(cif_path: str, atom_type: str) -> int:
    """Count occurrences of a specific atom type in a CIF file."""
    path = Path(cif_path)
    if not path.exists():
        return 0
    with open(path, 'r') as file:
        lines = file.readlines()
    
    cnt = 0
    structure_started = False
    for line in lines:
        if line.startswith("_atom_site"):
            structure_started = True
            continue
        elif structure_started and line.startswith("loop_"):
            structure_started = False
            continue
        elif structure_started and line.startswith(atom_type):
            cnt += 1

    return cnt


@tool
def make_plan(simulation_details: Annotated[str, "Details about the simulation."],
              agent_list: Annotated[List[str], "List of agent names to include in the plan."],
              task_list: Annotated[List[str], "List of tasks for each agent."]):
    """Create a plan for the agents to follow."""
    plan = {agent: {"task": task, "summary": ""} for agent, task in zip(agent_list, task_list)}
    plan["simulation_details"] = simulation_details
    with open("plan.json", "w") as file:
        json.dump(plan, file, indent=2)

    #plan_string = [f"{agent}: {info['task']}" for agent, info in plan.items()]
    #return "\n\n".join(plan_string)
    return "Plan created."


from ast import literal_eval

@tool
def get_helium_void_fraction(zeolite_code: str, n_al: Annotated[int, "Number of Al atoms"]) -> float:
    """Get the helium void fraction for a zeolite topology with a set number of Al atoms."""
    with open(f"HVF/HVF_{zeolite_code}/hvf_{zeolite_code.lower()}.dat") as f:
        data = literal_eval(f.read())
    
    if n_al in data:
        return data[n_al]
    else:
        x0 = min(data.keys())
        x1 = max(data.keys())
        y0 = data[x0]
        y1 = data[x1]

        # calculate slope and intercept
        slope = (y1 - y0) / (x1 - x0)
        intercept = y0 - slope * x0

        # linear interpolation
        return slope * n_al + intercept


@tool
def read_plan() -> str:
    """Read the current plan."""

    if not Path("plan.json").exists():
        return "No plan found."

    with open("plan.json", "r") as file:
        plan_dict = json.load(file)
    agent_string = [f"{agent}:\n Task description: {info['task']} \n Summary: {info['summary']}" for agent, info in plan_dict.items() if agent!= "simulation_details"]
    plan_string = f"Simulation Details: {plan_dict.get('simulation_details', 'No details provided')}\n\n"
    plan_string += "\n\n".join(agent_string)
    return plan_string

@tool
def write_summary(agent_name: str, task_summary: str):
    """Write a summary of the task carried out by a specific agent."""
    with open("plan.json", "r") as file:
        plan_dict = json.load(file)
    
    if agent_name in plan_dict:
        plan_dict[agent_name]["summary"] = task_summary
    else:
        raise ValueError(f"Agent {agent_name} not found in the plan.")
    
    with open("plan.json", "w") as file:
        json.dump(plan_dict, file, indent=2)
    
    return f"Summary updated."

@tool
def get_unit_cell_size(cif_path: str):
    """Get the size of the unit cell from a CIF file."""
    path = Path(cif_path)
    if not path.exists():
        return None
    with open(path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if line.startswith("_cell_length_a"):
            a = float(line.split()[1])
        elif line.startswith("_cell_length_b"):
            b = float(line.split()[1])
        elif line.startswith("_cell_length_c"):
            c = float(line.split()[1])
    return (a, b, c)

@tool
def edit_plan(agent_name: str, new_task: str):
    """Edit the task description for a specific agent."""
    with open("plan.json", "r") as file:
        plan_dict = json.load(file)

    if agent_name in plan_dict:
        plan_dict[agent_name]["task"] = new_task
        plan_dict[agent_name]["summary"] = ""
    else:
        raise ValueError(f"Agent {agent_name} not found in the plan.")

    with open("plan.json", "w") as file:
        json.dump(plan_dict, file, indent=2)

    return f"Task for {agent_name} updated."

@tool
def edit_simulation_details(new_details):
    """Edit the simulation details in the plan."""
    with open("plan.json", "r") as file:
        plan_dict = json.load(file)

    plan_dict["simulation_details"] = new_details

    with open("plan.json", "w") as file:
        json.dump(plan_dict, file, indent=2)

    return "Simulation details updated."

def get_force_field_atoms(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    interactions = []
    started = False
    for line in lines:
        if line.startswith("# type"):
            started = True
            continue
        if started and line.startswith("#"):
            break
        if started:
            interactions.extend(line.split()[:2])
    return list(set(interactions))

def get_force_field_mixing_atoms(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    interactions = []
    started = False
    for line in lines:
        if line.startswith("# type"):
            started = True
            continue
        if started and line.startswith("#"):
            break
        if started:
            interactions.append(line.split()[0])
    return interactions

def get_pseudo_atoms(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    interactions = []
    started = False
    for line in lines:
        if line.startswith("#type"):
            started = True
            continue
        if started and len(line) < 2:
            break
        if started:
            interactions.append(line.split()[0])
    return interactions

@tool
def get_atoms_in_ff_file(folder_path: str, file_name: str) -> List[str]:
    """Gets the atoms defined in a force field file (force_field.def, pseudo_atoms.def, force_field_mixing_rules.def)."""
    assert file_name in ["force_field.def", "pseudo_atoms.def", "force_field_mixing_rules.def"]
    if file_name == "force_field.def":
        return get_force_field_atoms(f"{folder_path}/{file_name}")
    elif file_name == "pseudo_atoms.def":
        return get_pseudo_atoms(f"{folder_path}/{file_name}")
    elif file_name == "force_field_mixing_rules.def":
        return get_force_field_mixing_atoms(f"{folder_path}/{file_name}")

    else:
        raise ValueError(f"Unknown file name: {file_name}")

