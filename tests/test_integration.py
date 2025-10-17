"""Integration tests for RLM."""

import pytest
from unittest.mock import patch, MagicMock
from rlm import RLM


class MockResponse:
    """Mock LLM response."""
    def __init__(self, content):
        self.choices = [MagicMock(message=MagicMock(content=content))]




@pytest.mark.asyncio
async def test_peek_strategy():
    """Test peeking at context start."""
    responses = [
        MockResponse('peek = context[:50]'),
        MockResponse('FINAL_VAR(peek)'),
    ]

    with patch('rlm.core.litellm.acompletion') as mock:
        mock.side_effect = responses

        rlm = RLM(model="test-model")
        result = await rlm.acompletion(
            "What does the context start with?",
            "This is a long document that starts with this sentence..."
        )

        assert "This is a long document" in result


@pytest.mark.asyncio
async def test_search_strategy():
    """Test regex search strategy."""
    responses = [
        MockResponse('matches = re.findall(r"\\d{4}", context)'),
        MockResponse('FINAL_VAR(matches)'),
    ]

    with patch('rlm.core.litellm.acompletion') as mock:
        mock.side_effect = responses

        rlm = RLM(model="test-model")
        result = await rlm.acompletion(
            "Find all years",
            "The years 2020, 2021, and 2022 were important."
        )

        assert "2020" in result


@pytest.mark.asyncio
async def test_chunk_strategy():
    """Test chunking context."""
    responses = [
        MockResponse('chunks = [context[i:i+10] for i in range(0, len(context), 10)]\nnum_chunks = len(chunks)'),
        MockResponse('FINAL_VAR(num_chunks)'),
    ]

    with patch('rlm.core.litellm.acompletion') as mock:
        mock.side_effect = responses
        rlm = RLM(model="test-model")
        result = await rlm.acompletion(
            "Chunk the context",
            "A" * 50  # 50 chars -> 5 chunks of 10
        )

        assert "5" in result


@pytest.mark.asyncio
async def test_extraction_strategy():
    """Test data extraction."""
    responses = [
        MockResponse('lines = context.split("\\n")\nnames = [l for l in lines if "Name:" in l]\nprint(names)'),
        MockResponse('FINAL_VAR(names)'),
    ]

    with patch('rlm.core.litellm.acompletion') as mock:
        mock.side_effect = responses
        rlm = RLM(model="test-model")
        context = """
Name: Alice
Age: 30
Name: Bob
Age: 25
"""
        result = await rlm.acompletion("Extract names", context)

        assert "Alice" in result or "Bob" in result


@pytest.mark.asyncio
async def test_error_recovery():
    """Test recovery from REPL errors."""
    responses = [
        MockResponse('x = undefined_variable'),  # Will cause error
        MockResponse('x = "recovered"\nprint(x)'),
        MockResponse('FINAL("Error recovered")'),
    ]

    with patch('rlm.core.litellm.acompletion') as mock:
        mock.side_effect = responses
        rlm = RLM(model="test-model")
        result = await rlm.acompletion("Test", "Context")

        assert result == "Error recovered"


@pytest.mark.asyncio
async def test_long_context():
    """Test with long context."""
    responses = [
        MockResponse('length = len(context)'),
        MockResponse('FINAL_VAR(length)'),
    ]

    with patch('rlm.core.litellm.acompletion') as mock:
        mock.side_effect = responses
        rlm = RLM(model="test-model")
        long_context = "A" * 100000  # 100k chars
        result = await rlm.acompletion("How long is this?", long_context)

        assert "100000" in result


@pytest.mark.asyncio
async def test_multiline_answer():
    """Test multiline final answer."""
    responses = [
        MockResponse('FINAL("""Line 1\nLine 2\nLine 3""")'),
    ]

    with patch('rlm.core.litellm.acompletion') as mock:
        mock.side_effect = responses
        rlm = RLM(model="test-model")
        result = await rlm.acompletion("Test", "Context")

        assert "Line 1" in result
        assert "Line 2" in result


@pytest.mark.asyncio
async def test_context_not_in_prompt():
    """Test that context is not passed in messages."""
    with patch('rlm.core.litellm.acompletion') as mock:
        mock.return_value = MockResponse('FINAL("Done")')

        rlm = RLM(model="test-model")
        context = "Very long context " * 1000
        await rlm.acompletion("Test", context)

        # Check that context is not in any message
        call_args = mock.call_args[1]
        messages = call_args['messages']

        for msg in messages:
            # Context should not be in the message content
            assert context not in msg['content']
