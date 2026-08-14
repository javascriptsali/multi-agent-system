# 🤖 Smart Multi-Agent Orchestration System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-Powered-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-red.svg)](https://groq.com/)

> 🚀 **An autonomous multi-agent system where specialized AI agents (Researcher, Coder, Reviewer) collaborate to solve complex software engineering tasks.**

---

## 🌟 Overview

A robust, production-ready **Multi-Agent AI Pipeline** built with **LangGraph**, **Groq (Llama 3.1)**, and **DuckDuckGo Search**. The system uses a **Plan-and-Execute** architecture where an AI Planner dynamically breaks down tasks and delegates them to specialized agents (Researcher, Coder, and Reviewer) to produce high-quality, peer-reviewed python code.

---

## 🌟 Key Features

* **🧠 Dynamic Planning:** An explicit Planner agent analyzes complex tasks and builds optimal execution chains (e.g., `researcher → coder → reviewer`).
* **🌐 Web-Search Integrated Research:** Researcher agent queries the live web to fetch current best practices, documentation, and sources.
* **💻 Clean Code Generation:** Coder agent focuses exclusively on generating pure, executable, and documented Python code.
* **🔍 Automated Code Review:** Reviewer agent performs AST/logic analysis, identifies edge-case bugs, and outputs refactored production code.
* **🛡️ Isolated State Management:** Modular LangGraph state design preventing context-bleed and hallucination between pipeline steps.
* **⚡ Isolated Agent Unit Testing:** Built-in CLI runner for testing each agent independently or running the entire multi-agent graph chain.

---

## 🧠 How It Works (The Agent Graph)

The system is powered by a deterministic state-graph architecture managed by **LangGraph**. Here is the step-by-step execution flow:

1. **Planner Node:**
   - Evaluates the user's initial prompt.
   - Generates a dynamic, ordered execution sequence of agents (e.g., `["researcher", "coder", "reviewer"]`).
   - Updates the shared `MultiAgentState` with the step sequence and execution roadmap.

2. **Researcher Node:**
   - Consumes the prompt task and queries web search engines using DuckDuckGo tools.
   - Aggregates best practices, documentation links, and key concepts.
   - Stores the pure summary into `researcher_output` inside the shared state.

3. **Coder Node:**
   - Receives the clean research findings from `researcher_output` (ignoring metadata/junk).
   - Generates clean, production-grade, documented Python code matching the user's constraints.
   - Saves the raw Python output strictly into `coder_output` for downstream isolation.

4. **Reviewer Node:**
   - Reads *only* the output from `coder_output` to prevent context pollution.
   - Analyzes syntax validity, edge cases, type safety, and runtime security issues.
   - Produces a structured code review breakdown alongside a single, fully refactored Python block.

5. **Deterministic Routing (`route_to_next_step`):**
   - After each agent completes its step, the conditional router inspects `current_step` in `MultiAgentState`.
   - Directs control to the next scheduled node in the plan without extra LLM routing overhead.
   - Terminates gracefully at `END` when the final step completes or max iterations (safety guard) are reached.

---

## 🏗️ System Architecture

```text
                       ┌─────────────────────────┐
                       │      User Task          │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │     Planner Node        │
                       └────────────┬────────────┘
                                    │ Generates Sequence
                                    ▼
     ┌─────────────────────────────────────────────────────────────┐
     │                       LangGraph State                       │
     └───────┬──────────────────────┬──────────────────────┬───────┘
             │                      │                      │
             ▼                      ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │Researcher Agent │───>│   Coder Agent   │───>│ Reviewer Agent  │
    │(DuckDuckGo Tool)│    │ (Pure Code Gen) │    |(CodeRefactoring)│
    └─────────────────┘    └─────────────────┘    └─────────────────┘

```

## 🛠️ Tech Stack

|       Component      |           Technology           |
|----------------------|--------------------------------|
| **Orchestration**    | LangGraph (StateGraph)         |
| **LLM Provider**     | Groq API (Llama 3.3 70B)       |
| **Agent Framework**  | LangChain Core                 |
| **Web Search**       | DuckDuckGo (ddgs)              |
| **Code Analysis**    | Python `ast` module            |
| **State Management** | TypedDict & LangGraph Messages |

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/javascriptsali/multi-agent-system.git
cd multi-agent-system
```
### 2. Set up environment

```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
or: source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```
### 3. Configure API key

- Create a .env file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```
- 🔑 Get your free API key from console.groq.com

### 4. Run the system

```bash
# Test the full pipeline (Research -> Code -> Review)
python graph.py --test all

# Test individual agents
python graph.py --test researcher
python graph.py --test coder
python graph.py --test reviewer
```
### 📁 Project Structure

```tree
multi-agent-system/
├── data/
├── docs/
├── logs/
├── multi-agent-system/
├── src/
│   ├── __init__.py
│   ├── agents.py           # Agent definitions (Planner, Researcher, Coder, Reviewer)
│   ├── graph.py            # LangGraph workflow orchestration & CLI test setup
│   ├── main.py             # Main entry point for application execution
│   ├── state.py            # LangGraph shared state schemas (MultiAgentState)
│   └── tools.py            # Custom agent tools (DuckDuckGo search, analysis, etc.)
├── tests/
│   └── __init__.py         # Test suites module
├── .Dockerignore
├── .env                    # Environment secrets (GROQ_API_KEY)
├── .gitignore
├── Dockerfile
├── last_coder_output.txt   # Cached output for reviewer standalone testing
├── README.md               # System documentation
└── requirements.txt        # Project dependencies
```
### 💡 Key Architectural Lessons Learned

- **Building production-grade Multi-Agent systems requires strict control boundaries:**

- **1. Limit Tool Scope: Giving execution/analysis tools to code generation nodes often leads to infinite tool-calling loops. Restrict external tools solely to agents that require real-world data (e.g., Researcher).**

- **2. Isolate State Variables: Avoid passing full raw chat history to all nodes. Storing individual state variables (researcher_output, coder_output) keeps downstream nodes focused.**

- **3. Network Resilience: LLM APIs can drop connections under high loads (WinError 10054). Always configure timeouts and automatic retries at model instantiation:**

```bash
ChatGroq(model="llama-3.1-8b-instant", request_timeout=60.0, max_retries=3)
```
# 🤝 Contributing
- **Contributions are welcome! Please feel free to submit a Pull Request.**

# 📝 License

- **This project is licensed under the MIT License — see the LICENSE file for details.**

# 📧 Contact

- **Saleh Bakhtiyari-[javascriptsali@gmail.com](javascriptsali@gmail.com)**
- **[Project Link]:(https://github.com/javascriptsali/multi-agent-system.git)**

# ⭐ Support

- **If you find this project useful, consider giving it a star!**
- **Your support helps motivate me to create more open-source AI engineering projects.** 🚀

