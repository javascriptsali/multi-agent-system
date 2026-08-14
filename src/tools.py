"""
Specialized Tools Module for the Multi-Agent System.

Each agent has its own set of tools:
- Researcher: web search for information gathering
- Coder: code analysis and generation helpers
- Reviewer: code quality assessment
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import re
import ast
from langchain_core.tools import tool


# ==========================================
# RESEARCHER AGENT TOOLS
# ==========================================

@tool
def search_web(query: str, max_results: int = 3) -> str:
    """Search the web for information."""
    try:
        max_results = int(max_results)
    except (ValueError, TypeError):
        max_results = 3

    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=max_results))
        if not results:
            return "No results found."
        
        formatted = []
        for r in results:
            formatted.append(f"Title: {r.get('title')}\nSnippet: {r.get('body', '')[:400]}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Error executing search: {str(e)}"


@tool
def summarize_text(text: str, max_sentences: int = 5) -> str:
    """
    Summarize a given text into a concise format.
    """
    if not text or not text.strip():
        return "No text provided to summarize."

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= max_sentences:
        return text

    summary = ". ".join(sentences[:max_sentences]) + "."
    return f"Summary ({max_sentences} key points):\n{summary}"


# ==========================================
# CODER & REVIEWER TOOLS
# ==========================================

@tool
def analyze_code(code: str) -> str:
    """
    Analyze Python code for syntax validity, complexity, and structure.
    """
    report = []

    try:
        tree = ast.parse(code)
        report.append("✅ Syntax: Valid Python code")
    except SyntaxError as e:
        report.append(f"❌ Syntax Error: {e}")
        return "\n".join(report)

    lines = code.split("\n")
    report.append(f"📏 Total Lines: {len(lines)}")
    report.append(f"📏 Non-empty Lines: {len([l for l in lines if l.strip()])}")

    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    report.append(f"🔧 Functions ({len(functions)}): {', '.join(functions) if functions else 'None'}")
    report.append(f"🏗️ Classes ({len(classes)}): {', '.join(classes) if classes else 'None'}")

    return "\n".join(report)


@tool
def review_code(code: str = "") -> str:
    """
    Analyzes python code for potential syntax errors and basic security flaws.
    """
    if not code:
        return "No code provided for review."

    issues = []
    
    try:
        ast.parse(code)
        issues.append("✅ Syntax is valid.")
    except Exception as e:
        issues.append(f"❌ Syntax Error: {e}")

    if "eval(" in code or "exec(" in code:
        issues.append("⚠️ Security Warning: Dangerous eval/exec detected.")
    if "open(" in code and "encoding=" not in code:
        issues.append("💡 Style Suggestion: Missing explicit encoding='utf-8' in open().")

    return "\n".join(issues)


@tool
def check_dependencies(code: str = "") -> str:
    """
    Identify external library imports in the code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "Cannot analyze: Syntax error in code."

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

    if not imports:
        return "No imports found in the code."

    std_lib = {
        "os", "sys", "re", "ast", "json", "math", "datetime",
        "collections", "itertools", "functools", "pathlib",
        "typing", "logging", "unittest", "io", "time", "random"
    }

    external = imports - std_lib
    standard = imports & std_lib

    report = ["📦 DEPENDENCY ANALYSIS", "=" * 30]
    if standard:
        report.append(f"📚 Standard Library: {', '.join(sorted(standard))}")
    if external:
        report.append(f"🔗 External Libraries: {', '.join(sorted(external))}")

    return "\n".join(report)


if __name__ == "__main__":
    print("🔍 Testing search_web...")
    print(search_web.invoke({"query": "Python best practices", "max_results": 2}))