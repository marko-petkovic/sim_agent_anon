import requests
import time
import subprocess
from pathlib import Path
import os
import shutil
import pymupdf

import json
from rapidfuzz import fuzz

from langchain.agents import tool

@tool
def semantic_scholar_search(query, limit=5, fields="title,authors,url,abstract,year,externalIds"):
    """Perform a semantic search using the Semantic Scholar API."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": fields
    }

    tries = 0
    while tries < 10:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            break
        except Exception as e:
            tries += 1
            print("sleeping for 1 second before retrying due to:", e)
            time.sleep(1)  # Wait for a second before retrying
    if tries == 10:
        return "Search timed out, please try again later."

    
    data = response.json()
    return data.get("data", [])



def download_paper(doi: str, paper_name: str):
    """
    Uses PyPaperBot to download a paper by DOI.

    Parameters:
        doi (str): DOI of the paper to download.
        download_dir (str): Directory where PDF will be saved.
    """
    download_dir = f"./papers/{paper_name}"
    # Ensure download directory exists
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    # Build the command as a list
    cmd = [
        "python",
        "-m", "PyPaperBot",
        f"--doi={doi}",
        f"--dwn-dir={download_dir}"
    ]

    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running PyPaperBot:", result.stderr)
    else:
        print("PyPaperBot output:", result.stdout)
        print(f"Paper downloaded to {download_dir}")



def alphabetic_ratio(text):
    letters = sum(c.isalpha() for c in text)
    return letters / max(len(text), 1)

def is_header(block_text: str, max_length=50, similarity_threshold=80):
    """
    Determine if a text block is a section header.
    
    Args:
        block_text (str): The text block to check.
        common_headers (list): List of canonical headers (e.g., Introduction, Methods).
        max_length (int): Maximum number of characters for a block to be considered a header.
        similarity_threshold (int): Minimum similarity % to match a known header.
    
    Returns:
        bool: True if block is likely a header, False otherwise.
    """
    block_text = block_text.strip()
    
    # Skip long blocks
    if len(block_text) > max_length:
        return False
    
    if alphabetic_ratio(block_text) < 0.75:
        return False
    
    common_headers = ["Introduction", "Methodology", "Methods", "Results", "Experiments",
                      "Discussion", "Conclusion", "Abstract", "References", "Supplementary"]
    # Check similarity against common headers
    if common_headers:
        for header in common_headers:
            score = fuzz.partial_ratio(block_text.lower(), header.lower())
            if score >= similarity_threshold:
                return True
    
    
    # Otherwise, treat short block as potential header
    return True


def filter_headers(text):

    content_dict = {}

    curr_hdr = None

    for block in text:
        if is_header(block):
            curr_hdr = block.strip()
            content_dict[curr_hdr] = ""
        elif curr_hdr is not None:
            content_dict[curr_hdr] += block

    # remove entries with no content
    content_dict = {k: v for k, v in content_dict.items() if len(v.strip()) > 0}
    return content_dict



def chunk_paper_sections(paper_dict, max_words=2000, overlap=100):
    chunks = {}
    for section, content in paper_dict.items():
        chunks[section] = {}
        words = content.split()
        for idx, i in enumerate(range(0, len(words), max_words - overlap)):
            chunk = ' '.join(words[i:i + max_words])
            chunks[section][f"chunk_{idx}"] = chunk

    return chunks

def parse_paper(paper_path: str):
    text = []

    doc = pymupdf.open(paper_path)
    for page in doc:
        blocks = page.get_text("blocks")
        blocks = [i[4] for i in blocks if i[6] == 0]  # Extract text from blocks
        text.extend(blocks)

    text_dict = filter_headers(text)
    return chunk_paper_sections(text_dict)


@tool
def download_paper_tool(doi: str, paper_name: str, paper_year: int):
    """
    Download a paper using PyPaperBot.

    Parameters:
        doi (str): DOI of the paper to download.
        paper_name (str): Name of the paper.
        paper_year (int): Year of publication.
    """
    if os.path.exists(f"./papers/{paper_name}"):
        return f"Paper already downloaded to ./papers/{paper_name}"

    download_paper(doi, paper_name)

    # check if the paper was downloaded successfully
    download_dir = f"./papers/{paper_name}"
    for file in os.listdir(download_dir):
        if file.endswith(".pdf"):
            try:
                # Try to parse the PDF to ensure it's valid
                parsed_paper = parse_paper(os.path.join(download_dir, file))
                with open(os.path.join(download_dir, "parsed_paper.json"), "w") as f:
                    json.dump(parsed_paper, f)

                return f"Paper downloaded and parsed successfully to {download_dir}/{file}"
            except Exception as e:
                print(f"Failed to parse {file}: {e}")

    # delete the directory if download failed
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir, ignore_errors=True)

    return "Paper unavailable, download failed"

@tool
def read_paper_names():
    """Read the names of all downloaded papers, located in the ./papers directory."""
    download_dir = "./papers"
    paper_names = []
    for folder in os.listdir(download_dir):
        if os.path.isdir(os.path.join(download_dir, folder)):
            paper_names.append(folder)
    return paper_names

@tool
def read_paper_headers(paper_folder: str):
    """
    Read the headers from a parsed paper JSON file.

    Args:
        paper_folder (str): The path to the paper folder. NOT the pdf file.

    Returns:
        list: A dictionary mapping headers to the amount of chunks they contain
    """
    with open(os.path.join(paper_folder, "parsed_paper.json"), "r") as f:
        parsed_paper = json.load(f)

    # headers = []
    # for section in parsed_paper:
    #     chunks = []
    #     for chunk in parsed_paper[section]:
    #         chunks.append(chunk)
    #     headers.append(f"{section}: {', '.join(chunks)}")
    headers = {}
    for section in parsed_paper:
        chunks = []
        for chunk in parsed_paper[section]:
            chunks.append(chunk)
        headers[section] = chunks

    return headers

@tool
def read_whole_paper(paper_folder: str):
    """
    Read the entire content of a parsed paper JSON file.
    """
    with open(os.path.join(paper_folder, "parsed_paper.json"), "r") as f:
        parsed_paper = json.load(f)

    # Combine all sections and chunks into a single string
    full_text = ""
    for section in parsed_paper:
        full_text += f"=== {section} ===\n"
        for chunk in parsed_paper[section]:
            full_text += parsed_paper[section][chunk] + "\n"

    MAX_CHARS = 50_000

    return full_text[:MAX_CHARS]

@tool
def read_paper_section(paper_folder: str, section: str, chunk: int=0):
    """
    Read a specific section from a parsed paper JSON file.

    Args:
        paper_folder (str): The path to the paper folder. NOT the pdf file.
        section (str): The section to read.
        chunk (int): The chunk number to read.

    Returns:
        str: The content of the specified section and chunk.
    """
    with open(os.path.join(paper_folder, "parsed_paper.json"), "r") as f:
        parsed_paper = json.load(f)

    if section in parsed_paper:
        if chunk < len(parsed_paper[section]):
            return parsed_paper[section][f"chunk_{chunk}"]
        else:
            return "Chunk not found"
    else:
        return "Section not found"

@tool
def read_finding(paper_folder: str):
    """
    Read force field parameters from a text file in the specified paper folder.

    Args:
        paper_folder (str): The path to the paper folder.  NOT the finding file.

    Returns:
        list: A list of force field parameters found in the file.
    """
    finding_file = os.path.join(paper_folder, "findings.txt")
    if not os.path.exists(finding_file):
        return []

    with open(finding_file, "r") as f:
        findings = f.readlines()

    return [finding.strip() for finding in findings]

from typing import List

@tool
def write_finding(paper_folder: str, findings: List[str]):
    """
    Write force field parameters to a text file in the specified paper folder.

    Args:
        paper_folder (str): The path to the paper folder.
        findings (List[str]): The findings to write.
    """
    finding_file = os.path.join(paper_folder, "findings.txt")
    with open(finding_file, "a") as f:
        for finding in findings:
            f.write(finding + "\n")
    return f"Successfully written to {finding_file}"

@tool
def write_file(folder: str, filename: str, content: str):
    """
    Write content to a file in the specified folder.

    Args:
        folder (str): The path to the folder.
        filename (str): The name of the file.
        content (str): The content to write.
    """
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, filename), "w") as f:
        f.write(content)
    return f"Successfully written to {os.path.join(folder, filename)}"

@tool
def read_file(folder: str, filename: str):
    """
    Read content from a file in the specified folder.

    Args:
        folder (str): The path to the folder.
        filename (str): The name of the file.

    Returns:
        str: The content of the file.
    """
    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        return f"File {filename} does not exist in {folder}."

    with open(file_path, "r") as f:
        content = f.read()
    return content

@tool
def list_directory(folder: str):
    """List the folder and file structure of a given directory (folder)."""
    
    # if the agent tries to access '.', we need to tell them its not allowed
    if folder in [".", ""]:
        return "Accessing the current directory is not allowed."

    start_path = str(folder)
    output = []
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        output.append(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            output.append(f'{subindent}{f}')
    return '\n'.join(output)


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