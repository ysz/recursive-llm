"""Parse FINAL() and FINAL_VAR() statements from LLM responses."""

import re
from typing import Optional, Dict, Any


def extract_final(response: str) -> Optional[str]:
    """
    Extract answer from FINAL() statement.

    Args:
        response: LLM response text

    Returns:
        Extracted answer or None if not found
    """
    # Look for FINAL("answer") or FINAL('answer')
    patterns = [
        r'FINAL\s*\(\s*"""(.*)"""',  # FINAL("""answer""") - triple double quotes
        r"FINAL\s*\(\s*'''(.*)'''",  # FINAL('''answer''') - triple single quotes
        r'FINAL\s*\(\s*"([^"]*)"',  # FINAL("answer") - double quotes
        r"FINAL\s*\(\s*'([^']*)'",  # FINAL('answer') - single quotes
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    return None


def extract_final_var(response: str, env: Dict[str, Any]) -> Optional[str]:
    """
    Extract answer from FINAL_VAR() statement.

    Args:
        response: LLM response text
        env: REPL environment with variables

    Returns:
        Variable value as string or None if not found
    """
    # Look for FINAL_VAR(var_name)
    match = re.search(r'FINAL_VAR\s*\(\s*(\w+)\s*\)', response)
    if not match:
        return None

    var_name = match.group(1)

    # Get variable from environment
    if var_name in env:
        value = env[var_name]
        return str(value)

    return None


def is_final(response: str) -> bool:
    """
    Check if response contains FINAL() or FINAL_VAR().

    Args:
        response: LLM response text

    Returns:
        True if response contains final statement
    """
    return 'FINAL(' in response or 'FINAL_VAR(' in response


def parse_response(response: str, env: Dict[str, Any]) -> Optional[str]:
    """
    Parse response for any final statement.

    Args:
        response: LLM response text
        env: REPL environment

    Returns:
        Final answer or None
    """
    # Try FINAL() first
    answer = extract_final(response)
    if answer is not None:
        return answer

    # Try FINAL_VAR()
    answer = extract_final_var(response, env)
    if answer is not None:
        return answer

    return None
