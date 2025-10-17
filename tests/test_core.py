"""Tests for core RLM."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from rlm import RLM, MaxIterationsError, MaxDepthError


class MockResponse:
    """Mock LLM response."""
    def __init__(self, content):
        self.choices = [MagicMock(message=MagicMock(content=content))]


@pytest.fixture
def mock_litellm():
    """Mock litellm.acompletion."""
    with patch('rlm.core.litellm.acompletion') as mock:
        yield mock


@pytest.mark.asyncio
async def test_simple_completion(mock_litellm):
    """Test simple completion with FINAL."""
    mock_litellm.return_value = MockResponse('FINAL("The answer")')

    rlm = RLM(model="test-model")
    result = await rlm.acompletion("What is the answer?", "Some context")

    assert result == "The answer"
    assert mock_litellm.called


@pytest.mark.asyncio
async def test_multi_step_completion(mock_litellm):
    """Test multi-step completion."""
    responses = [
        MockResponse('x = context[:10]\nprint(x)'),
        MockResponse('FINAL("Done")'),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    result = await rlm.acompletion("Test", "Hello World Test")

    assert result == "Done"
    assert mock_litellm.call_count == 2


@pytest.mark.asyncio
async def test_max_iterations_error(mock_litellm):
    """Test max iterations exceeded."""
    mock_litellm.return_value = MockResponse('x = 1')  # Never returns FINAL

    rlm = RLM(model="test-model", max_iterations=3)

    with pytest.raises(MaxIterationsError):
        await rlm.acompletion("Test", "Context")


@pytest.mark.asyncio
async def test_max_depth_error(mock_litellm):
    """Test max depth exceeded."""
    rlm = RLM(model="test-model", max_depth=2, _current_depth=2)

    with pytest.raises(MaxDepthError):
        await rlm.acompletion("Test", "Context")


@pytest.mark.asyncio
async def test_final_var(mock_litellm):
    """Test FINAL_VAR extraction."""
    responses = [
        MockResponse('result = "Test Answer"\nprint(result)'),
        MockResponse('FINAL_VAR(result)'),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    result = await rlm.acompletion("Test", "Context")

    assert result == "Test Answer"


@pytest.mark.asyncio
async def test_repl_error_handling(mock_litellm):
    """Test REPL error handling."""
    responses = [
        MockResponse('x = 1 / 0'),  # This will cause error
        MockResponse('FINAL("Recovered")'),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    result = await rlm.acompletion("Test", "Context")

    assert result == "Recovered"


@pytest.mark.asyncio
async def test_context_operations(mock_litellm):
    """Test context operations in REPL."""
    responses = [
        MockResponse('first_10 = context[:10]'),
        MockResponse('FINAL_VAR(first_10)'),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    result = await rlm.acompletion("Get first 10 chars", "Hello World Example")

    assert result == "Hello Worl"


def test_sync_completion():
    """Test sync wrapper."""
    with patch('rlm.core.litellm.acompletion') as mock:
        mock.return_value = MockResponse('FINAL("Sync result")')

        rlm = RLM(model="test-model")
        result = rlm.completion("Test", "Context")

        assert result == "Sync result"


@pytest.mark.asyncio
async def test_two_models(mock_litellm):
    """Test using different models for root and recursive."""
    mock_litellm.return_value = MockResponse('FINAL("Answer")')

    rlm = RLM(
        model="expensive-model",
        recursive_model="cheap-model",
        _current_depth=0
    )

    await rlm.acompletion("Test", "Context")

    # First call should use expensive model
    call_args = mock_litellm.call_args_list[0]
    assert call_args[1]['model'] == "expensive-model"


@pytest.mark.asyncio
async def test_stats(mock_litellm):
    """Test statistics tracking."""
    responses = [
        MockResponse('x = 1'),
        MockResponse('y = 2'),
        MockResponse('FINAL("Done")'),
    ]
    mock_litellm.side_effect = responses

    rlm = RLM(model="test-model")
    await rlm.acompletion("Test", "Context")

    stats = rlm.stats
    assert stats['llm_calls'] == 3
    assert stats['iterations'] == 3
    assert stats['depth'] == 0


@pytest.mark.asyncio
async def test_api_base_and_key(mock_litellm):
    """Test API base and key passing."""
    mock_litellm.return_value = MockResponse('FINAL("Answer")')

    rlm = RLM(
        model="test-model",
        api_base="http://localhost:8000",
        api_key="test-key"
    )

    await rlm.acompletion("Test", "Context")

    call_kwargs = mock_litellm.call_args[1]
    assert call_kwargs['api_base'] == "http://localhost:8000"
    assert call_kwargs['api_key'] == "test-key"
