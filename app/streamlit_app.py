import os
import sys
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

# Fix Python path so it can locate the 'src' module
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load environment variables
load_dotenv()

# Import graph runner functions
from src.graph import run_agent_test, run_full_graph

# Page Configuration
st.set_page_config(
    page_title="Multi-Agent AI Studio",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Plan-and-Execute System")
st.markdown("""
Welcome to the live interactive demo of the **LangGraph** & **Groq** powered Multi-Agent System.
""")

# ---------------------------------------------------------
# Sidebar: System Constraints & Rules
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚠️ Playground Rules & Guidelines")
    st.info("""
    1. **Single Testing:** Test individual agents with dedicated inputs.
    2. **Full Graph Execution:** Provide an overall Root Task to run the full collaborative pipeline (Researcher ➔ Coder ➔ Reviewer).
    """)

    st.markdown("---")
    st.caption("Powered by LangGraph, Groq & Streamlit")

# ---------------------------------------------------------
# Tabs Separation
# ---------------------------------------------------------
tab_single, tab_full = st.tabs(["🧪 Single Agent Testing", "⚡ Full Graph Execution Pipeline"])

# ==========================================
# TAB 1: Single Agent Testing
# ==========================================
with tab_single:
    st.subheader("📝 Test Individual Agents in Isolation")
    col1, col2, col3 = st.columns(3)

    with col1:
        task_researcher = st.text_area(
            "1. Researcher Task:",
            value="Search the web for 3 best practices of Laplace transform in Python and return summary with sources.",
            height=120,
            key="single_researcher_input"
        )
        if st.button("🔍 Run Researcher Alone", use_container_width=True):
            with st.spinner("Searching the web and aggregating insights..."):
                try:
                    res = run_agent_test("researcher", task_researcher)
                    st.markdown(res)
                except Exception as e:
                    st.error(f"Error executing Researcher agent: {e}")

    with col2:
        task_coder = st.text_area(
            "2. Coder Task:",
            value="Write a Python function to compute Laplace transform using sympy.",
            height=120,
            key="single_coder_input"
        )
        if st.button("💻 Run Coder Alone", use_container_width=True):
            coder_input_lower = task_coder.lower()
            if not any(keyword in coder_input_lower for keyword in ["code", "python", "function", "script", "class"]):
                st.warning("⚠️ Guardrail Triggered: Please enter a prompt that explicitly requests Python code generation!")
            else:
                with st.spinner("Generating Python code solution..."):
                    try:
                        res = run_agent_test("coder", task_coder)
                        st.code(res, language="python")
                    except Exception as e:
                        st.error(f"Error executing Coder agent: {e}")

    with col3:
        task_reviewer = st.text_area(
            "3. Reviewer Task:",
            value="Review the Python function for Laplace transform.",
            height=120,
            key="single_reviewer_input"
        )
        if st.button("🧐 Run Reviewer Alone", use_container_width=True):
            with st.spinner("Analyzing and refactoring code snippet..."):
                try:
                    res = run_agent_test("reviewer", task_reviewer)
                    st.markdown(res)
                except Exception as e:
                    st.error(f"Error executing Reviewer agent: {e}")

# ==========================================
# TAB 2: Full Graph Execution
# ==========================================
with tab_full:
    st.subheader("🚀 Full Multi-Agent Graph Execution Flow")

    root_task = st.text_area(
        "🎯 Root Pipeline Task (Goal for All Agents):",
        value="Research best practices for computing Laplace transforms in Python, write a clean function using SymPy, and review the code for accuracy and performance.",
        height=100,
        help="This main prompt is passed to the Planner to generate the execution graph."
    )

    if st.button("⚡ Execute Full Pipeline (All Agents)", type="primary", use_container_width=True):
        output_container = st.container()
        with output_container:
            st.info(f"**Root Pipeline Task:** {root_task}")
            with st.spinner("Executing Plan-and-Execute Graph (Planner ➔ Researcher ➔ Coder ➔ Reviewer)..."):
                try:
                    final_state = run_full_graph(root_task)

                    if final_state.get("researcher_output"):
                        with st.expander("🔎 Step 1: Researcher Findings", expanded=True):
                            st.markdown(final_state["researcher_output"])

                    if final_state.get("coder_output"):
                        with st.expander("💻 Step 2: Coder Implementation", expanded=True):
                            st.code(final_state["coder_output"], language="python")

                    if final_state.get("reviewer_output"):
                        with st.expander("🧐 Step 3: Reviewer Audit & Refactor", expanded=True):
                            st.markdown(final_state["reviewer_output"])

                    st.success(f"✅ Multi-Agent Graph executed successfully! Plan: {' ➔ '.join(final_state.get('plan', []))}")
                except Exception as e:
                    st.error(f"Error during full graph execution: {e}")