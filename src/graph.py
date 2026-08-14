"""
LangGraph Multi-Agent Orchestration (Plan-and-Execute).

Flow:
    Task → Planner → [Agent 1] → [Agent 2] → ... → END
"""
import sys
import argparse
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from langgraph.graph import StateGraph, END
from state import MultiAgentState
from agents import (
    planner_node,
    researcher_node,
    coder_node,
    reviewer_node,
)


# ==========================================
# Deterministic Routing
# ==========================================

def route_to_next_step(state):
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    iteration = state.get("iteration_count", 0)

    if iteration >= 8 or current_step >= len(plan):
        return "END"

    next_agent = plan[current_step]
    if next_agent in ["researcher", "coder", "reviewer"]:
        return next_agent

    return "END"


# ==========================================
# Build Graph
# ==========================================

def build_multi_agent_graph():
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        route_to_next_step,
        {
            "researcher": "researcher",
            "coder": "coder",
            "reviewer": "reviewer",
            "END": END,
        },
    )

    for agent in ["researcher", "coder", "reviewer"]:
        workflow.add_conditional_edges(
            agent,
            route_to_next_step,
            {
                "researcher": "researcher",
                "coder": "coder",
                "reviewer": "reviewer",
                "END": END,
            },
        )

    return workflow.compile()


# ==========================================
# Test Function
# ==========================================

def test_agent(agent_name: str):
    print(f"\n{'=' * 60}")
    print(f"📝 {agent_name.capitalize()} Test")
    print(f"{'=' * 60}")

    tasks = {
        "researcher": "Search the web for 3 best practices of Laplace transform in Python and return summary with sources.",
        "coder": "Write a Python function to compute Laplace transform using sympy.",
        "reviewer": "Review the Python function for Laplace transform.",
        "all": "Write a Python function that calculates Laplace transform based on best research practices.",
    }

    task = tasks.get(agent_name, tasks["all"])
    print(f"Task: {task}\n")

    initial_state = {
        "messages": [],
        "current_task": task,
        "task_status": "planning",
        "iteration_count": 0,
        "plan": [agent_name] if agent_name != "all" else [],
        "current_step": 0,
        "last_output": "",
        "researcher_output": "",
        "coder_output": "",
    }

    # 1. Single Agent Execution
    if agent_name in ["researcher", "coder", "reviewer"]:
        print(f"⚡ Running single agent directly: [{agent_name}]\n")

        if agent_name == "reviewer":
            try:
                with open("last_coder_output.txt", "r", encoding="utf-8") as f:
                    saved_code = f.read().strip()
                if saved_code:
                    initial_state["coder_output"] = saved_code
                    initial_state["last_output"] = saved_code
                    print("📌 Loaded code from 'last_coder_output.txt' for Reviewer.\n")
            except FileNotFoundError:
                initial_state["coder_output"] = "import sympy as sp\ndef laplace(f, t, s):\n    return sp.laplace_transform(f, t, s)"
                initial_state["last_output"] = initial_state["coder_output"]

        nodes_map = {
            "researcher": researcher_node,
            "coder": coder_node,
            "reviewer": reviewer_node,
        }

        try:
            node_func = nodes_map[agent_name]
            result = node_func(initial_state)
            last_output = result.get("last_output", "")

            if agent_name == "coder" and last_output:
                with open("last_coder_output.txt", "w", encoding="utf-8") as f:
                    f.write(last_output)

            print(f"\n{'=' * 60}")
            print("📊 Final Output:")
            print("=" * 60)
            print(f"\n{last_output}\n")
            print("-" * 40)
            print(f"✅ Single agent '{agent_name}' executed successfully.")

        except Exception as e:
            print(f"\n❌ Error during single agent test: {e}")
            traceback.print_exc()

    # 2. Full Graph Execution
    elif agent_name == "all":
        print("🚀 Executing Full Multi-Agent Graph Chain...\n")
        try:
            app = build_multi_agent_graph()
            print("✅ Graph compiled successfully!\n")

            final_state = app.invoke(initial_state)

            print(f"\n{'=' * 60}")
            print("📊 Final Conversation Flow:")
            print("=" * 60 + "\n")

            for msg in final_state.get("messages", []):
                content = msg.content if hasattr(msg, "content") else str(msg)
                print(content)
                print("-" * 40 + "\n")

            print(f"✅ Executed plan: {final_state.get('plan', [])}")
            print(f"✅ Task status: {final_state.get('task_status', 'completed')}")

        except Exception as e:
            print(f"\n❌ Error during full graph execution: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Graph Tests")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["researcher", "coder", "reviewer", "all"],
        help="Which test to run",
    )
    args = parser.parse_args()
    test_agent(args.test)

# =========================================================
# Wrapper Functions for Streamlit / External Calling
# =========================================================


def run_agent_test(agent_name: str, task: str) -> str:
    """Executes a single agent test and returns the output as a string."""
    if agent_name == "researcher":
        from src.agents import researcher_node

        state = {"messages": [("user", task)], "task": task}
        result = researcher_node(state)
        return result.get("researcher_output", "No output generated.")

    elif agent_name == "coder":
        from src.agents import coder_node

        state = {"messages": [("user", task)], "task": task}
        result = coder_node(state)
        return result.get("coder_output", "No output generated.")

    elif agent_name == "reviewer":
        from src.agents import reviewer_node

        state = {"messages": [("user", task)], "task": task}
        result = reviewer_node(state)
        return result.get("reviewer_output", "No output generated.")

    return "Invalid agent name specified."


def run_full_graph(task: str) -> dict:
    """Executes the full LangGraph pipeline and returns the final state dict."""
    # ۱. ابتدا گراف را می‌سازیم و در متغیر app می‌ریزیم
    app = build_multi_agent_graph()

    # ۲. وضعیت اولیه را تعریف می‌کنیم
    initial_state = {
        "task": task,
        "plan": [],
        "current_step": 0,
        "messages": [("user", task)],
    }

    # ۳. گراف را اجرا می‌کنیم
    final_state = app.invoke(initial_state)
    return final_state