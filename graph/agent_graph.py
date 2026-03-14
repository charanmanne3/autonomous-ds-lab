"""LangGraph workflow for LLM multi-agent chatbot."""

from __future__ import annotations

from typing import Any, Callable, Dict, TypedDict

from langgraph.graph import END, StateGraph

from agents.chatbot import (
    AnalysisAgent,
    ChatReportAgent,
    CodeAgent,
    ResearchAgent,
    VisualizationAgent,
)


class AgentState(TypedDict, total=False):
    message: str
    similar_context: str
    research: str
    analysis: str
    code: str
    visualization: str
    report: str
    final_response: str


research_agent = ResearchAgent()
analysis_agent = AnalysisAgent()
code_agent = CodeAgent()
visualization_agent = VisualizationAgent()
report_agent = ChatReportAgent()


def research_node(state: AgentState) -> AgentState:
    return {
        "research": research_agent.run(
            query=state.get("message", ""),
            similar_context=state.get("similar_context", ""),
        )
    }


def analysis_node(state: AgentState) -> AgentState:
    return {
        "analysis": analysis_agent.run(
            query=state.get("message", ""),
            research_notes=state.get("research", ""),
        )
    }


def code_node(state: AgentState) -> AgentState:
    return {
        "code": code_agent.run(
            query=state.get("message", ""),
            analysis=state.get("analysis", ""),
        )
    }


def visualization_node(state: AgentState) -> AgentState:
    return {
        "visualization": visualization_agent.run(
            query=state.get("message", ""),
            analysis=state.get("analysis", ""),
        )
    }


def report_node(state: AgentState) -> AgentState:
    final = report_agent.run(
        query=state.get("message", ""),
        research=state.get("research", ""),
        analysis=state.get("analysis", ""),
        code=state.get("code", ""),
        visualization=state.get("visualization", ""),
        memory_hint=state.get("similar_context", ""),
    )
    return {"report": final, "final_response": final}


def build_chat_graph():
    """Build and compile LangGraph workflow."""
    workflow = StateGraph(AgentState)
    workflow.add_node("research", research_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("code", code_node)
    workflow.add_node("visualization", visualization_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("research")
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "code")
    workflow.add_edge("code", "visualization")
    workflow.add_edge("visualization", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


chat_graph = build_chat_graph()


def run_chat_workflow(message: str, similar_context: str = "") -> Dict[str, Any]:
    """Execute graph and return final response payload."""
    result = chat_graph.invoke({"message": message, "similar_context": similar_context})
    return dict(result)


def run_chat_workflow_with_progress(
    message: str,
    similar_context: str = "",
    progress_callback: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    """Run agent workflow with explicit per-agent progress callbacks."""
    state: AgentState = {"message": message, "similar_context": similar_context}

    if progress_callback:
        progress_callback("Research Agent")
    state.update(research_node(state))

    if progress_callback:
        progress_callback("Analysis Agent")
    state.update(analysis_node(state))

    if progress_callback:
        progress_callback("Code Agent")
    state.update(code_node(state))

    if progress_callback:
        progress_callback("Visualization Agent")
    state.update(visualization_node(state))

    if progress_callback:
        progress_callback("Report Agent")
    state.update(report_node(state))

    return dict(state)

