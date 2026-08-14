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
Welcome to the live interactive demo of the **LangGraph** & **Groq (Llama 3.1)** powered Multi-Agent System.
You can test each agent individually in isolation or run the full collaborative execution pipeline across all agents.
""")

# ---------------------------------------------------------
# Sidebar: System Constraints & Rules
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚠️ Playground Rules & Guidelines")
    st.info("""
    To get the best results and prevent execution errors, please adhere to these guidelines:
    
    1. **Researcher Test:** Queries requiring live web search, technical documentation, or math formulations.
    2. **Coder Test:** Explicit requests for Python code generation only (e.g., avoid off-topic queries like *"What is the price of Bitcoin?"*).
    3. **Reviewer Test:** Evaluates, audits, and refactors previously generated code snippets.
    4. **Full Graph Execution:** Combined tasks must focus on **Python programming, mathematical modeling, or computer science concepts** so the pipeline chain (*Research ➔ Code ➔ Review*) executes logically.
    """)
    
    st.markdown("---")
    st.caption("Powered by LangGraph, Groq Llama-3.1 & Streamlit")

# ---------------------------------------------------------
# Main UI: Input Tasks
# ---------------------------------------------------------
st.subheader("📝 Define Agent Tasks")

col1, col2, col3 = st.columns(3)

with col1:
    task_researcher = st.text_area(
        "1. Researcher Task:",
        value="Search the web for 3 best practices of Laplace transform in Python and return summary with sources.",
        height=120,
        help="Specify a task that requires web research or scientific guidelines."
    )

with col2:
    task_coder = st.text_area(
        "2. Coder Task:",
        value="Write a Python function to compute Laplace transform using sympy.",
        height=120,
        help="Specify a task strictly requesting Python code generation."
    )

with col3:
    task_reviewer = st.text_area(
        "3. Reviewer Task:",
        value="Review the Python function for Laplace transform.",
        height=120,
        help="Specify criteria for code analysis, bug fixing, or refactoring."
    )

st.markdown("---")

# ---------------------------------------------------------
# Execution Buttons
# ---------------------------------------------------------
st.subheader("🚀 Execute Agents")

btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

run_researcher = btn_col1.button("🔍 Run Researcher", use_container_width=True)
run_coder = btn_col2.button("💻 Run Coder", use_container_width=True)
run_reviewer = btn_col3.button("🧐 Run Reviewer", use_container_width=True)
run_all = btn_col4.button("⚡ Run All Agents (Full Graph)", type="primary", use_container_width=True)

# Container for rendering output
output_container = st.container()

# ---------------------------------------------------------
# Logic Handlers
# ---------------------------------------------------------
if run_researcher:
    with output_container:
        st.subheader("📊 Researcher Output")
        with st.spinner("Searching the web and aggregating insights..."):
            try:
                res = run_agent_test("researcher", task_researcher)
                st.markdown(res)
            except Exception as e:
                st.error(f"Error executing Researcher agent: {e}")

elif run_coder:
    # Client-side input validation guardrail
    coder_input_lower = task_coder.lower()
    if not any(keyword in coder_input_lower for keyword in ["code", "python", "function", "script", "class"]):
        st.warning("⚠️ Guardrail Triggered: Please enter a prompt that explicitly requests Python code generation!")
    else:
        with output_container:
            st.subheader("💻 Coder Output")
            with st.spinner("Generating Python code solution..."):
                try:
                    res = run_agent_test("coder", task_coder)
                    st.code(res, language="python")
                except Exception as e:
                    st.error(f"Error executing Coder agent: {e}")

elif run_reviewer:
    with output_container:
        st.subheader("🧐 Reviewer Output")
        with st.spinner("Analyzing and refactoring code snippet..."):
            try:
                res = run_agent_test("reviewer", task_reviewer)
                st.markdown(res)
            except Exception as e:
                st.error(f"Error executing Reviewer agent: {e}")

elif run_all:
    with output_container:
        st.subheader("📊 Full Multi-Agent Graph Execution Flow")
        st.info(f"**Root Pipeline Task:** {task_researcher}")
        
        with st.spinner("Executing Plan-and-Execute Graph (Planner ➔ Researcher ➔ Coder ➔ Reviewer)..."):
            try:
                final_state = run_full_graph(task_researcher)
                
                if "researcher_output" in final_state:
                    with st.expander("🔎 Step 1: Researcher Findings", expanded=True):
                        st.markdown(final_state["researcher_output"])
                        
                if "coder_output" in final_state:
                    with st.expander("💻 Step 2: Coder Implementation", expanded=True):
                        st.code(final_state["coder_output"], language="python")
                        
                if "reviewer_output" in final_state:
                    with st.expander("🧐 Step 3: Reviewer Audit & Refactor", expanded=True):
                        st.markdown(final_state["reviewer_output"])
                        
                st.success("✅ Multi-Agent Graph executed successfully!")
            except Exception as e:
                st.error(f"Error during full graph execution: {e}")