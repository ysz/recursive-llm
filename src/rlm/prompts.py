"""System prompt templates for RLM."""


def build_system_prompt(context_size: int, depth: int = 0) -> str:
    """
    Build system prompt for RLM.

    Args:
        context_size: Size of context in characters
        depth: Current recursion depth

    Returns:
        System prompt string
    """
    # Minimal prompt (paper-style)
    prompt = f"""You are a Recursive Language Model. You interact with context through a Python REPL environment.

The context is stored in variable `context` (not in this prompt). Size: {context_size:,} characters.

Available in environment:
- context: str (the document to analyze)
- query: str (the question: "{"{"}query{"}"}")
- recursive_llm(sub_query, sub_context) -> str (recursively process sub-context)
- re: already imported regex module (use re.findall, re.search, etc.)

Write Python code to answer the query. The last expression or print() output will be shown to you.

Examples:
- print(context[:100])  # See first 100 chars
- errors = re.findall(r'ERROR', context)  # Find all ERROR
- count = len(errors); print(count)  # Count and show

When you have the answer, use FINAL("answer") - this is NOT a function, just write it as text.

Depth: {depth}"""

    return prompt


def build_user_prompt(query: str) -> str:
    """
    Build user prompt.

    Args:
        query: User's question

    Returns:
        User prompt string
    """
    return query
