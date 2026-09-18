"""
LangGraph Workflow Definition.

Assembles the end-to-end conversation graph:
extract_variables
  -> check_completeness
       ├── insufficient -> return clarifying question -> END
       └── sufficient -> retrieve_knowledge -> reason -> validate_grounding
                            ├── pass -> compute_confidence -> END
                            └── fail -> retry reason (max 1) -> fallback -> compute_confidence -> END
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from app.graph.state import GraphState
from app.graph.nodes import (
    extract_variables,
    check_completeness,
    retrieve_knowledge,
    reason,
    validate_grounding,
    compute_confidence
)


# In-memory session store for MVP multi-turn state persistence
SESSION_STORE: Dict[str, GraphState] = {}


def ask_clarification(state: GraphState) -> Dict[str, Any]:
    """
    Formats the clarifying question as the final response when context is incomplete.
    """
    question = state.get("clarifying_question", "Could you provide more environmental context for your land?")
    return {
        "final_response": question,
        "confidence": "LOW"
    }


def route_completeness(state: GraphState) -> Literal["retrieve_knowledge", "ask_clarification"]:
    """
    Conditional routing based on context completeness.
    """
    if state.get("is_complete", False):
        return "retrieve_knowledge"
    return "ask_clarification"


def route_validation(state: GraphState) -> Literal["compute_confidence", "reason"]:
    """
    Conditional routing based on grounding validation:
    - If valid (or second retry completed), proceed to compute_confidence.
    - If invalid and retry_count <= 1, retry reasoning.
    """
    if state.get("validation_passed", False):
        return "compute_confidence"
    return "reason"


def build_graph():
    """
    Builds and compiles the LangGraph StateGraph.
    """
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("extract_variables", extract_variables)
    workflow.add_node("check_completeness", check_completeness)
    workflow.add_node("ask_clarification", ask_clarification)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("reason", reason)
    workflow.add_node("validate_grounding", validate_grounding)
    workflow.add_node("compute_confidence", compute_confidence)

    # Set Entry Point
    workflow.set_entry_point("extract_variables")

    # Connect Edges
    workflow.add_edge("extract_variables", "check_completeness")

    # Conditional Branch: Completeness Check
    workflow.add_conditional_edges(
        "check_completeness",
        route_completeness,
        {
            "retrieve_knowledge": "retrieve_knowledge",
            "ask_clarification": "ask_clarification"
        }
    )

    workflow.add_edge("ask_clarification", END)
    workflow.add_edge("retrieve_knowledge", "reason")
    workflow.add_edge("reason", "validate_grounding")

    # Conditional Branch: Grounding Validation (Retry or Proceed)
    workflow.add_conditional_edges(
        "validate_grounding",
        route_validation,
        {
            "compute_confidence": "compute_confidence",
            "reason": "reason"
        }
    )

    workflow.add_edge("compute_confidence", END)

    return workflow.compile()


# Compile the global workflow graph instance
graph_app = build_graph()


def execute_chat_turn(session_id: str, user_message: str) -> Dict[str, Any]:
    """
    Executes a single chat turn within an active or new conversation session.
    Persists variables and message history across turns.
    """
    # Retrieve existing state or initialize fresh session
    if session_id in SESSION_STORE:
        prev_state = SESSION_STORE[session_id]
        current_state: GraphState = {
            "session_id": session_id,
            "current_user_message": user_message,
            "messages": prev_state.get("messages", []) + [{"role": "user", "content": user_message}],
            "variables": prev_state.get("variables", {}),
            "is_complete": False,
            "clarifying_question": None,
            "retrieved_context": {},
            "draft_response": None,
            "validation_passed": False,
            "retry_count": 0,
            "confidence": "LOW",
            "final_response": None
        }
    else:
        current_state: GraphState = {
            "session_id": session_id,
            "current_user_message": user_message,
            "messages": [{"role": "user", "content": user_message}],
            "variables": {},
            "is_complete": False,
            "clarifying_question": None,
            "retrieved_context": {},
            "draft_response": None,
            "validation_passed": False,
            "retry_count": 0,
            "confidence": "LOW",
            "final_response": None
        }

    # Run the state machine
    result = graph_app.invoke(current_state)

    # Append assistant response to message history
    assistant_msg = result.get("final_response", "")
    result["messages"].append({"role": "assistant", "content": assistant_msg})

    # Save state for subsequent turns
    SESSION_STORE[session_id] = result

    return result
