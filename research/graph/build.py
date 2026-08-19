"""Wire the research loop as a LangGraph StateGraph.

  plan -> execute -> reflect -> (insufficient AND within budget? -> execute)
                             -> synthesize -> END

Synthesis is now real: per-claim citations with an explicit unverified path, then a
mechanical overlap check that demotes claims whose cited passages don't back them.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from research.graph.budget import stop_reason
from research.graph.hops import execute_batch
from research.graph.state import ResearchState, new_state
from research.graph.sufficiency import reflect_node
from research.planning.decompose import decompose
from research.synthesis.synthesize import _dedup_evidence, synthesize
from research.synthesis.verify import verify


def plan_node(state: ResearchState) -> dict:
    plan, _usage = decompose(state["question"])
    return {"plan": plan}


def synthesize_node(state: ResearchState) -> dict:
    final = verify(synthesize(state), _dedup_evidence(state))
    return {
        "final": {
            "answer": final.render(),
            "claims": [
                {"text": c.text, "passage_ids": c.passage_ids, "supported": c.supported}
                for c in final.claims
            ],
            "unverified_count": len(final.unverified),
            "prompt_tokens": final.prompt_tokens,
            "completion_tokens": final.completion_tokens,
        },
        "done": True,
    }


def after_execute(state: ResearchState) -> str:
    """Budget beats plan: even unrun hops don't justify blowing the cap."""
    if stop_reason(state):
        return "synthesize"
    return "execute" if state["plan"].ready() else "reflect"


def after_reflect(state: ResearchState) -> str:
    """Reflection can only ADD work; it can never override the budget."""
    if state.get("done") or stop_reason(state):
        return "synthesize"
    return "execute" if state["plan"].ready() else "synthesize"


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_batch)
    graph.add_node("reflect", reflect_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_conditional_edges("execute", after_execute, {"execute": "execute", "reflect": "reflect", "synthesize": "synthesize"})
    graph.add_conditional_edges("reflect", after_reflect, {"execute": "execute", "synthesize": "synthesize"})
    graph.add_edge("synthesize", END)
    return graph.compile()


def research(question: str) -> ResearchState:
    return build_graph().invoke(new_state(question))
