"""Tests for parser module."""

import pytest
from rlm.parser import extract_final, extract_final_var, is_final, parse_response


def test_extract_final_double_quotes():
    """Test extracting FINAL with double quotes."""
    response = 'Some text\nFINAL("The answer is 42")\nMore text'
    assert extract_final(response) == "The answer is 42"


def test_extract_final_single_quotes():
    """Test extracting FINAL with single quotes."""
    response = "FINAL('Hello world')"
    assert extract_final(response) == "Hello world"


def test_extract_final_triple_quotes():
    """Test extracting FINAL with triple quotes."""
    response = '''FINAL("""This is a
multiline
answer""")'''
    assert "multiline" in extract_final(response)


def test_extract_final_not_found():
    """Test when FINAL not found."""
    response = "Just some text without final"
    assert extract_final(response) is None


def test_extract_final_var():
    """Test extracting FINAL_VAR."""
    response = "result = 'test'\nFINAL_VAR(result)"
    env = {'result': 'test value'}
    assert extract_final_var(response, env) == "test value"


def test_extract_final_var_not_found():
    """Test when FINAL_VAR not found."""
    response = "Just some code"
    env = {}
    assert extract_final_var(response, env) is None


def test_extract_final_var_missing_variable():
    """Test when variable doesn't exist in env."""
    response = "FINAL_VAR(missing)"
    env = {}
    assert extract_final_var(response, env) is None


def test_is_final():
    """Test is_final detection."""
    assert is_final('FINAL("answer")')
    assert is_final('FINAL_VAR(result)')
    assert not is_final('Just text')


def test_parse_response_final():
    """Test parse_response with FINAL."""
    response = 'FINAL("answer")'
    env = {}
    assert parse_response(response, env) == "answer"


def test_parse_response_final_var():
    """Test parse_response with FINAL_VAR."""
    response = 'FINAL_VAR(x)'
    env = {'x': 'value'}
    assert parse_response(response, env) == "value"


def test_parse_response_none():
    """Test parse_response with no final."""
    response = 'Just code'
    env = {}
    assert parse_response(response, env) is None
