"""
Agent Definitions for Plan-and-Execute Multi-Agent System.
"""
import os
import re
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from tools import (
    search_web,
    summarize_text,
    analyze_code,
    review_code,
    check_dependencies,
)

load_dotenv()

PLANNER_MODEL = "llama-3.1-8b-instant"
AGENT_MODEL = "llama-3.1-8b-instant"


def get_planner_llm() -> ChatGroq:
    return ChatGroq(
        model=PLANNER_MODEL,
        temperature=0.0,
        max_tokens=500,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def get_agent_llm() -> ChatGroq:
    return ChatGroq(
        model=AGENT_MODEL,
        temperature=0.1,
        max_tokens=3000,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ==========================================
# Strict Prompts
# ==========================================

PLANNER_PROMPT = """You are a Task Planner. Given a user request, decide which agents should run in sequence.
Available agents: researcher, coder, reviewer.

RULES:
1. Unless the request is exclusively text search, ALWAYS include all 3 agents in order: researcher,coder,reviewer.
2. Output ONLY a comma-separated list of agent names (e.g., researcher,coder,reviewer).
3. No extra text, markdown, or explanations."""

RESEARCHER_PROMPT = """You are a Research Specialist.
Your task is to search the web and summarize key best practices based on search findings.

STRICT INSTRUCTIONS:
1. Call the search_web tool to fetch accurate information.
2. Summarize findings in EXACTLY this format:

### Summary of Best Practices:
• [Point 1]
• [Point 2]
• [Point 3]

### Sources:
1. [Source Domain/URL 1]
2. [Source Domain/URL 2]

RULES:
- NEVER leave 'Summary of Best Practices' empty.
- ALWAYS include the '### Sources:' section with web domains/URLs found.
- NEVER write Python code."""

CODER_PROMPT = """You are an Expert Python Developer.
Your job is to write high-quality, executable Python code based strictly on the user request and research context provided.

STRICT INSTRUCTIONS:
1. Output ONLY executable Python code inside a markdown code block (```python ... ```).
2. DO NOT include any conversational intro/outro or explanations outside the code block.
3. DO NOT write research findings or web scraper scripts unless explicitly requested."""

REVIEWER_PROMPT = """You are a Senior Code Reviewer.
Your task is to review the provided Python code for bugs, type safety, performance, and style issues.

STRICT INSTRUCTIONS:
1. DO NOT summarize research findings or invent sources.
2. Focus strictly on the Python code provided in the context.
3. Provide EXACTLY one code review analysis and one refactored python code block.

FORMAT REQUIRED:
### Code Review Analysis
- **Strengths:** [Short summary]
- **Key Issues Found:** [Max 3 bullet points]

### Refactored Code
```python
# Refactored python code here
```"""

TOOL_MAP = {
    "search_web": search_web,
    "summarize_text": summarize_text,
    "analyze_code": analyze_code,
    "review_code": review_code,
    "check_dependencies": check_dependencies,
}


def create_researcher_llm():
    return get_agent_llm().bind_tools([search_web, summarize_text])


def create_coder_llm():
    return get_agent_llm()


def create_reviewer_llm():
    return get_agent_llm()


# ==========================================
# Planner Node
# ==========================================

import re

def planner_node(state):
    llm = get_planner_llm()
    task = state.get("current_task") or state.get("task", "")

    response = llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"Task: {task}"),
    ])

    plan_text = str(response.content).lower()
    
    # استخراج کلمات کلیدی با Regex (حتی اگر مدل فاصله یا کاراکتر اضافی داده باشد)
    valid_agents = ["researcher", "coder", "reviewer"]
    raw_found = re.findall(r"\b(researcher|coder|reviewer)\b", plan_text)
    
    # حذف موارد تکراری با حفظ ترتیب
    plan = []
    for agent in raw_found:
        if agent not in plan:
            plan.append(agent)

    # اگر مدل لیست ناقص داد (مثلاً کمتر از ۳ ایجنت) یا هیچ‌کدام را نیافت، لیست کامل را جایگزین کن
    if len(plan) < 3:
        plan = ["researcher", "coder", "reviewer"]

    print(f"📋 Generated Plan: {' → '.join(plan)}")

    return {
        "plan": plan,
        "current_step": 0,
        "iteration_count": state.get("iteration_count", 0),
        "task_status": "executing",
    }


# ==========================================
# Agent Execution Helpers
# ==========================================

def run_agent_with_context(llm, system_prompt, agent_specific_task, context="", max_iterations=3):
    user_content = f"### Main Task:\n{agent_specific_task}"

    if context:
        user_content += f"\n\n### Additional Context / Previous Output:\n```text\n{context.strip()}\n```"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]

    for _ in range(max_iterations):
        try:
            response = llm.invoke(messages)
        except Exception as e:
            return f"Agent error: {str(e)}"

        tool_calls = getattr(response, "tool_calls", None)

        if tool_calls and len(tool_calls) > 0:
            messages.append(response)

            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                if tool_name in TOOL_MAP:
                    try:
                        result = TOOL_MAP[tool_name].invoke(tool_args)
                    except Exception as e:
                        result = f"Error executing tool {tool_name}: {str(e)}"
                else:
                    result = f"Tool '{tool_name}' is not available."

                messages.append(ToolMessage(
                    content=str(result)[:2000],
                    tool_call_id=tool_call_id if tool_call_id else "call_1",
                    name=tool_name,
                ))
        else:
            return response.content if hasattr(response, "content") else str(response)

    last = messages[-1]
    return last.content if hasattr(last, "content") else str(last)


# ==========================================
# Agent Nodes
# ==========================================

def researcher_node(state):
    main_task = state.get("current_task") or state.get("task", "")

    response_text = run_agent_with_context(
        create_researcher_llm(),
        RESEARCHER_PROMPT,
        agent_specific_task=f"Research best practices and sources for: {main_task}",
        context="",
    )

    formatted_message = f"[Researcher]\n{response_text}"

    return {
        "messages": [AIMessage(content=formatted_message)],
        "last_output": response_text,
        "researcher_output": response_text,
        "current_step": state.get("current_step", 0) + 1,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def coder_node(state):
    print("\n--------------------------------------------------")
    print(f"⚙️ [DEBUG CODER] Starting node. Initial current_step: {state.get('current_step', 0)}")
    
    main_task = state.get("current_task") or state.get("task", "")
    research_context = state.get("researcher_output", "")

    coder_instruction = (
        f"Implement a complete Python script or solution that fulfills the requirement: '{main_task}'."
    )

    response_text = run_agent_with_context(
        create_coder_llm(),
        CODER_PROMPT,
        agent_specific_task=coder_instruction,
        context=research_context,
    )

    formatted_message = f"[Coder]\n{response_text}"

    print(f"✅ [DEBUG CODER] Execution finished.")
    print(f"📊 [DEBUG CODER] Plan in State: {state.get('plan', [])}")
    print(f"➡️ [DEBUG CODER] Step after increment will be: {state.get('current_step', 0) + 1}")
    print("--------------------------------------------------\n")

    return {
        "messages": [AIMessage(content=formatted_message)],
        "last_output": response_text,
        "coder_output": response_text,
        "current_step": state.get("current_step", 0) + 1,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def reviewer_node(state):
    main_task = state.get("current_task") or state.get("task", "")
    code_context = state.get("coder_output") or state.get("last_output", "")
    research_context = state.get("researcher_output", "")

    # ترکیب بافت برای ارزیابی همه‌جانبه توسط Reviewer
    full_context = f"--- RESEARCH FINDINGS ---\n{research_context}\n\n--- CODE TO REVIEW ---\n{code_context}"

    response_text = run_agent_with_context(
        create_reviewer_llm(),
        REVIEWER_PROMPT,
        agent_specific_task=(
            f"Review the provided Python code for the task: '{main_task}'. "
            f"Audit for syntax errors (e.g. missing symbols/imports), correctness, and optimization."
        ),
        context=full_context,
    )

    formatted_message = f"[Reviewer]\n{response_text}"

    return {
        "messages": [AIMessage(content=formatted_message)],
        "last_output": response_text,
        "reviewer_output": response_text,
        "current_step": state.get("current_step", 0) + 1,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "task_status": "executing",  # وضعیت توسط Router تعیین شود نه هاردکد در نود
    }