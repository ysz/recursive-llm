"""Type definitions for RLM."""

from typing import TypedDict, Optional, Any, Callable, Awaitable


class Message(TypedDict):
    """LLM message format."""
    role: str
    content: str


class RLMConfig(TypedDict, total=False):
    """Configuration for RLM instance."""
    model: str
    recursive_model: Optional[str]
    api_base: Optional[str]
    api_key: Optional[str]
    max_depth: int
    max_iterations: int
    temperature: float
    timeout: int


class REPLEnvironment(TypedDict, total=False):
    """REPL execution environment."""
    context: str
    query: str
    recursive_llm: Callable[[str, str], Awaitable[str]]
    re: Any  # re module


class CompletionResult(TypedDict):
    """Result from RLM completion."""
    answer: str
    iterations: int
    depth: int
    llm_calls: int
