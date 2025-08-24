from agents.research_team.extraction_agent import create_extraction_agent
from agents.research_team.paper_agent import create_paper_agent
from agents.research_team.writer_agent import create_force_field_agent
from langgraph.graph import StateGraph, START, MessagesState, END

def create_research_team(model=None):
    extraction_agent = create_extraction_agent(model)
    paper_agent = create_paper_agent(model)
    force_field_agent = create_force_field_agent(model)

    graph = (
    StateGraph(MessagesState)
    .add_node("paper_agent", paper_agent)
    .add_node("extraction_agent", extraction_agent)
    .add_node("force_field_agent", force_field_agent)
    .add_edge(START, "paper_agent")
    .add_edge("force_field_agent", END)
    .compile()
    )
    return graph