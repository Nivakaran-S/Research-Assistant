
from langgraph.graph import StateGraph, START, END 
from src.states.scopeState import AgentState, AgentInputState
from src.nodes.scopeNode import clarify_with_user, write_research_brief

graph_builder = StateGraph(AgentState, input_schema=AgentInputState)

graph_builder.add_node("clarify_with_user", clarify_with_user)
graph_builder.add_node("write_research_brief", write_research_brief)

graph_builder.add_edge(START, "clarify_with_user")
graph_builder.add_edge("write_research_brief", END)

scope_research=graph_builder.compile()