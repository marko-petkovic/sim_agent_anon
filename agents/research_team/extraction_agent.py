from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools.handoff_tools import create_research_handoff_tool
from tools.paper_tools import read_paper_names, read_paper_headers, read_paper_section, write_finding, list_directory, read_whole_paper

transfer_to_paper_agent = create_research_handoff_tool(
    agent_name="paper_agent",
    description="Transfer to the paper retrieval agent. Specify which paper you need.",
    sender="extraction_agent"
)

transfer_to_force_field_agent = create_research_handoff_tool(
    agent_name="force_field_agent",
    description="Transfer to the force field agent. Only specify to which path the force field findings were written. Do not give instructions.",
    sender="extraction_agent"
)

def create_extraction_agent(model=None):

    model = model or ChatOpenAI(model="gpt-5-mini")

    _extraction_agent = create_react_agent(model, 
                           tools=[read_paper_names, read_paper_headers, read_paper_section, write_finding, list_directory, read_whole_paper,transfer_to_paper_agent,transfer_to_force_field_agent], 
      prompt = """"
      You are a research assistant extracting simulation parameters from papers in ./papers.

Workflow:
1. Paper handling
   - Use read_paper_names() to list papers.
   - If missing, request via transfer_to_paper_agent.
   - Always respond to other agents (e.g. paper or force field agent) with clear answers and continue. 

2. Section reading
   - For each paper: get headers with read_paper_headers(papers/[paper]).
   - Focus on sections like "Force Field", "Simulation Details", "Computational Methods".
   - Iterate through chunks with read_paper_section(paper, section, chunk_id).
   - If you cannot find what you are looking for, attempt to read the whole paper with read_whole_paper(paper).
   - If you still cannot find the information, request the relevant paper via transfer_to_paper_agent.

3. Parameter extraction
   - Extract only parameters defined by this paper, not comparative ones.
   - If a parameter cites another paper:
       • If available, read it.  
       • If missing, request via transfer_to_paper_agent.

4. Findings format
   - Write results with write_finding(), e.g.:
     {"type": "lennard-jones", "atoms": "O-Si", "epsilon": 0.1, "sigma": 3.4}
   - Add notes for rules:
     {"note": "Uses Lorentz-Berthelot mixing rules."}

5. Completeness check
   - Ensure all force field parameters are covered.
   - If gaps exist, fetch referenced papers.

6. Handoff
   - When complete:
       • Summarize in <50 words (e.g. "LJ + Coulomb with Lorentz-Berthelot rules").
       • Transfer findings to the force field agent with their file location.

Rules:
- Work systematically through papers.
- Always answer other agents.
- Keep outputs structured and limited to proposed parameters.
- Never stop the conversation.
""")
    
    def extraction_agent(state):
        response = _extraction_agent.invoke(state)
        return {"messages": response["messages"][-1]}

    return extraction_agent
