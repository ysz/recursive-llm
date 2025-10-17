"""Tests for REPL executor."""

import pytest
import re
from rlm.repl import REPLExecutor, REPLError


@pytest.fixture
def repl():
    """Create REPL executor."""
    return REPLExecutor()


def test_simple_expression(repl):
    """Test simple expression."""
    env = {}
    result = repl.execute("x = 5 + 3", env)
    assert env['x'] == 8


def test_string_operations(repl):
    """Test string operations on context."""
    env = {'context': 'Hello World'}
    result = repl.execute("result = context[:5]", env)
    assert env['result'] == 'Hello'


def test_regex_operations(repl):
    """Test regex operations."""
    env = {
        'context': 'The year is 2025',
        're': re
    }
    result = repl.execute("matches = re.findall(r'\\d+', context)", env)
    assert env['matches'] == ['2025']


def test_print_output(repl):
    """Test capturing print output."""
    env = {}
    result = repl.execute("print('Hello')", env)
    assert 'Hello' in result


def test_multiline_code(repl):
    """Test multiline code."""
    code = """
x = 10
y = 20
z = x + y
print(z)
"""
    env = {}
    result = repl.execute(code, env)
    assert '30' in result


def test_code_block_extraction(repl):
    """Test extracting code from markdown blocks."""
    text = """
Here's some code:
```python
x = 5
print(x)
```
"""
    env = {}
    result = repl.execute(text, env)
    assert env['x'] == 5


def test_list_operations(repl):
    """Test list operations."""
    env = {}
    result = repl.execute("items = [1, 2, 3, 4, 5]", env)
    assert env['items'] == [1, 2, 3, 4, 5]


def test_forbidden_import(repl):
    """Test that arbitrary imports are forbidden."""
    env = {}
    with pytest.raises(REPLError):
        repl.execute("import os", env)


def test_safe_builtins(repl):
    """Test safe built-in functions."""
    env = {}
    result = repl.execute("result = len([1, 2, 3])", env)
    assert env['result'] == 3


def test_comprehension(repl):
    """Test list comprehension."""
    env = {'context': 'Hello World'}
    result = repl.execute("chars = [c for c in context if c.isupper()]", env)
    assert env['chars'] == ['H', 'W']


def test_empty_code(repl):
    """Test empty code."""
    env = {}
    result = repl.execute("", env)
    assert "No code" in result


def test_syntax_error(repl):
    """Test syntax error handling."""
    env = {}
    with pytest.raises(REPLError):
        repl.execute("x = ", env)


def test_runtime_error(repl):
    """Test runtime error handling."""
    env = {}
    with pytest.raises(REPLError):
        repl.execute("x = 1 / 0", env)
