from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools.handoff_tools import create_research_handoff_tool
from tools.paper_tools import semantic_scholar_search, download_paper_tool

transfer_to_extraction_agent = create_research_handoff_tool(
    agent_name="extraction_agent",
    description="Transfer to the paper extraction agent. Only specify to which directory the paper was downloaded. Do not give instructions.",
    sender="paper_agent"
)

def create_paper_agent(model=None):

    model = model or ChatOpenAI(model="gpt-5-mini")

    _paper_agent = create_react_agent(model, 
                                  tools=[semantic_scholar_search, download_paper_tool, transfer_to_extraction_agent], 
                                  prompt= """
You are a research assistant specializing in finding force fields for classical molecular simulations.

Workflow:

1. Query generation
   - Create concise search queries using only essential keywords (no long sentences).
   - Example:
     • User request: "Force field parameters for H2 in zeolites including Al sites"
     • Query: "H2 zeolite Al force field"

2. Semantic Scholar search
   - Use the semantic_scholar_search tool with the query.
   - If results are poor:
     • First, try synonyms (e.g., "CO2" → "carbon dioxide").
     • Then, remove less important words to broaden the query.
     • Refine at most 2–3 times.

3. Paper selection
   - Choose relevant papers from the results.
   - Download them using download_paper_tool.
   - Give each paper a short descriptive title (≤8 words).
   - For each paper, provide a one-line explanation (<30 words) of why it was chosen.

4. Handoff
   - Transfer selected papers to the extraction agent for parameter analysis.

Guidelines:
- Keep queries short, targeted, and domain-specific.
- Focus on keywords directly related to molecules, materials, and force fields.
- Prefer precision first (specific molecules/systems), then broaden if necessary.
""")
    

    def paper_agent(state):
        response = _paper_agent.invoke(state)
        return {"messages": response["messages"][-1]}

    return paper_agent
