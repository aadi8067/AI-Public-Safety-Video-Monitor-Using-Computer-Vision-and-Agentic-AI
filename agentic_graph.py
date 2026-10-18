"""LangChain + LangGraph orchestration for the Agentic AI layer.

This module changes *how* an already-scored incident is routed through
the Risk Agent and Response Agent -- it does not change what either of
them computes. RiskAgent.assess() and ResponseAgent.determine() (in
agentic_ai.py) and EventCorrelator (in event_correlator.py) are untouched.

  Supervisor -> [ LangGraph: assess_risk -> determine_response ] -> incident

Each agent is wrapped as a LangChain Runnable (RunnableLambda) -- the
atomic "tool" unit -- and the two are composed as nodes of a compiled
LangGraph StateGraph -- the stateful orchestrator -- exactly mirroring
the "Agentic Supervisor -> Risk Agent -> Response Agent" pipeline this
codebase has documented since the original agentic_ai.py was written.

If langchain-core / langgraph are not installed (or fail to import for
any other reason), `build_agentic_pipeline` transparently falls back to
a plain-Python pipeline that runs the exact same two calls, in the exact
same order, with the exact same merge -- so behaviour and output are
identical either way, and the rest of the app is never affected by
whether these two optional dependencies happen to be installed.
"""
from typing import Any, Dict, TypedDict

try:
    from langchain_core.runnables import RunnableLambda
    from langgraph.graph import StateGraph, START, END
    _LANGGRAPH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only when the
    # optional deps are missing/broken; see build_agentic_pipeline().
    RunnableLambda = None
    StateGraph = START = END = None
    _LANGGRAPH_IMPORT_ERROR = exc


class AgenticState(TypedDict, total=False):
    """State threaded through the LangGraph pipeline for one incident."""
    incident: Dict[str, Any]
    risk: Dict[str, Any]
    response: Dict[str, Any]


class _FallbackPipeline:
    """Plain-Python Supervisor -> Risk -> Response pipeline.

    Used automatically when langchain-core / langgraph are not available.
    This is intentionally just the original two-call sequence the
    Supervisor used before the graph existed, so nothing about scoring or
    response selection changes if these optional packages are missing.
    """

    backend = "fallback"

    def __init__(self, risk_agent, response_agent):
        self._risk_agent = risk_agent
        self._response_agent = response_agent

    def run(self, incident_dict):
        risk = self._risk_agent.assess(incident_dict)
        response = self._response_agent.determine(risk, incident_dict)
        incident_dict.update(risk)
        incident_dict.update(response)
        return incident_dict


class _LangGraphPipeline:
    """Supervisor -> Risk Agent -> Response Agent as a compiled LangGraph
    StateGraph of two nodes, each backed by a LangChain RunnableLambda
    wrapping the existing (unmodified) RiskAgent/ResponseAgent methods."""

    backend = "langgraph"

    def __init__(self, risk_agent, response_agent):
        self._risk_runnable = RunnableLambda(risk_agent.assess)
        self._response_runnable = RunnableLambda(
            lambda inp: response_agent.determine(inp["risk"], inp["incident"])
        )
        self._graph = self._build()

    def _assess_risk(self, state: "AgenticState") -> dict:
        risk = self._risk_runnable.invoke(state["incident"])
        return {"risk": risk}

    def _determine_response(self, state: "AgenticState") -> dict:
        response = self._response_runnable.invoke(
            {"risk": state["risk"], "incident": state["incident"]}
        )
        return {"response": response}

    def _build(self):
        graph = StateGraph(AgenticState)
        graph.add_node("assess_risk", self._assess_risk)
        graph.add_node("determine_response", self._determine_response)
        graph.add_edge(START, "assess_risk")
        graph.add_edge("assess_risk", "determine_response")
        graph.add_edge("determine_response", END)
        return graph.compile()

    def run(self, incident_dict):
        result = self._graph.invoke({"incident": incident_dict})
        # Same merge as the original Supervisor._score(): risk fields
        # first, then response fields, onto the incident dict in place.
        incident_dict.update(result["risk"])
        incident_dict.update(result["response"])
        return incident_dict


def build_agentic_pipeline(risk_agent, response_agent):
    """Build the Supervisor -> Risk -> Response pipeline.

    Returns a _LangGraphPipeline when langchain-core + langgraph import
    and build cleanly; otherwise returns the output-identical
    _FallbackPipeline so the caller's behaviour never depends on whether
    these optional dependencies are installed.
    """
    if _LANGGRAPH_IMPORT_ERROR is None:
        try:
            return _LangGraphPipeline(risk_agent, response_agent)
        except Exception as exc:
            print(f"[agentic_graph] LangGraph pipeline failed to build ({exc!r}); "
                  f"falling back to the plain-Python pipeline.")
    return _FallbackPipeline(risk_agent, response_agent)


def backend_status():
    """Human-readable string describing which pipeline backend is active,
    for logging / the JSON response -- never affects behaviour."""
    if _LANGGRAPH_IMPORT_ERROR is None:
        return "langgraph"
    return f"fallback (langchain-core/langgraph not available: {_LANGGRAPH_IMPORT_ERROR})"
