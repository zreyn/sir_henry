"""Unit tests for llm_worker.py module."""

import pytest
from unittest.mock import patch, MagicMock, Mock
import json


@patch("llm_worker.requests.post")
@patch("llm_worker.prompt_queue")
@patch("llm_worker.interrupt_event")
@patch("llm_worker.sentence_queue")
@patch("llm_worker.logger")
@patch("llm_worker.SYSTEM_PROMPT", "You are a test character.")
def test_llm_worker_success(
    mock_logger,
    mock_sentence_queue,
    mock_interrupt,
    mock_prompt_queue,
    mock_requests_post,
):
    """Test llm_worker() with successful response."""
    # Setup mock response
    mock_response = MagicMock()
    response_lines = [
        b'{"response": "Hello"}',
        b'{"response": " world"}',
        b'{"response": "[SEP]"}',
    ]
    mock_response.iter_lines.return_value = iter(response_lines)
    mock_requests_post.return_value = mock_response

    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return "user input"
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return "more input"

    mock_prompt_queue.get.side_effect = queue_get
    mock_interrupt.is_set.return_value = False

    from llm_worker import llm_worker

    with pytest.raises(KeyboardInterrupt):
        llm_worker()

    mock_requests_post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": "user input",
            "system": "You are a test character.",
            "stream": True,
        },
        stream=True,
    )
    mock_interrupt.clear.assert_called_once()
    mock_sentence_queue.put.assert_called_once_with("Hello world")
    mock_logger.info.assert_called()


@patch("llm_worker.requests.post")
@patch("llm_worker.prompt_queue")
@patch("llm_worker.interrupt_event")
@patch("llm_worker.sentence_queue")
@patch("llm_worker.logger")
@patch("llm_worker.SYSTEM_PROMPT", "You are a test character.")
def test_llm_worker_multiple_sentences(
    mock_logger,
    mock_sentence_queue,
    mock_interrupt,
    mock_prompt_queue,
    mock_requests_post,
):
    """Test llm_worker() with multiple sentences separated by [SEP]."""
    mock_response = MagicMock()
    response_lines = [
        b'{"response": "First"}',
        b'{"response": " sentence[SEP]"}',
        b'{"response": "Second"}',
        b'{"response": " sentence[SEP]"}',
        b'{"response": "Third"}',
    ]
    mock_response.iter_lines.return_value = iter(response_lines)
    mock_requests_post.return_value = mock_response

    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return "user input"
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return "more input"

    mock_prompt_queue.get.side_effect = queue_get
    mock_interrupt.is_set.return_value = False

    from llm_worker import llm_worker

    with pytest.raises(KeyboardInterrupt):
        llm_worker()

    # Should queue two sentences
    assert mock_sentence_queue.put.call_count == 2
    mock_sentence_queue.put.assert_any_call("First sentence")
    mock_sentence_queue.put.assert_any_call("Second sentence")


@patch("llm_worker.requests.post")
@patch("llm_worker.prompt_queue")
@patch("llm_worker.interrupt_event")
@patch("llm_worker.sentence_queue")
@patch("llm_worker.logger")
@patch("llm_worker.SYSTEM_PROMPT", "You are a test character.")
def test_llm_worker_interrupt_during_stream(
    mock_logger,
    mock_sentence_queue,
    mock_interrupt,
    mock_prompt_queue,
    mock_requests_post,
):
    """Test llm_worker() when interrupted during streaming."""
    mock_response = MagicMock()
    response_lines = [
        b'{"response": "Hello"}',
        b'{"response": " world"}',
    ]

    call_count = [0]

    def iter_lines():
        for line in response_lines:
            call_count[0] += 1
            if call_count[0] == 2:
                mock_interrupt.is_set.return_value = True
            yield line

    mock_response.iter_lines.return_value = iter_lines()
    mock_requests_post.return_value = mock_response

    queue_call_count = [0]

    def queue_get():
        queue_call_count[0] += 1
        if queue_call_count[0] == 1:
            return "user input"
        elif queue_call_count[0] == 2:
            raise KeyboardInterrupt()
        return "more input"

    mock_prompt_queue.get.side_effect = queue_get
    mock_interrupt.is_set.side_effect = lambda: call_count[0] >= 2

    from llm_worker import llm_worker

    with pytest.raises(KeyboardInterrupt):
        llm_worker()

    # Should break out of loop when interrupted
    mock_sentence_queue.put.assert_not_called()


@patch("llm_worker.requests.post")
@patch("llm_worker.prompt_queue")
@patch("llm_worker.interrupt_event")
@patch("llm_worker.sentence_queue")
@patch("llm_worker.logger")
@patch("llm_worker.SYSTEM_PROMPT", "You are a test character.")
def test_llm_worker_empty_lines(
    mock_logger,
    mock_sentence_queue,
    mock_interrupt,
    mock_prompt_queue,
    mock_requests_post,
):
    """Test llm_worker() handles empty lines."""
    mock_response = MagicMock()
    response_lines = [
        b"",
        b'{"response": "Hello"}',
        b"",
        b'{"response": " world[SEP]"}',
    ]
    mock_response.iter_lines.return_value = iter(response_lines)
    mock_requests_post.return_value = mock_response

    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return "user input"
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return "more input"

    mock_prompt_queue.get.side_effect = queue_get
    mock_interrupt.is_set.return_value = False

    from llm_worker import llm_worker

    with pytest.raises(KeyboardInterrupt):
        llm_worker()

    # Should skip empty lines and process valid responses
    mock_sentence_queue.put.assert_called_once_with("Hello world")


@patch("llm_worker.requests.post")
@patch("llm_worker.prompt_queue")
@patch("llm_worker.interrupt_event")
@patch("llm_worker.sentence_queue")
@patch("llm_worker.logger")
@patch("llm_worker.SYSTEM_PROMPT", "You are a test character.")
def test_llm_worker_empty_sentence(
    mock_logger,
    mock_sentence_queue,
    mock_interrupt,
    mock_prompt_queue,
    mock_requests_post,
):
    """Test llm_worker() skips empty sentences."""
    mock_response = MagicMock()
    response_lines = [
        b'{"response": "   [SEP]"}',  # Empty sentence
        b'{"response": "Valid[SEP]"}',
        b'{"response": "   "}',  # Trailing whitespace only
    ]
    mock_response.iter_lines.return_value = iter(response_lines)
    mock_requests_post.return_value = mock_response

    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return "user input"
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return "more input"

    mock_prompt_queue.get.side_effect = queue_get
    mock_interrupt.is_set.return_value = False

    from llm_worker import llm_worker

    with pytest.raises(KeyboardInterrupt):
        llm_worker()

    # Should only queue non-empty sentence
    mock_sentence_queue.put.assert_called_once_with("Valid")


@patch("llm_worker.requests.post")
@patch("llm_worker.prompt_queue")
@patch("llm_worker.interrupt_event")
@patch("llm_worker.sentence_queue")
@patch("llm_worker.logger")
@patch("llm_worker.SYSTEM_PROMPT", "You are a test character.")
def test_llm_worker_exception(
    mock_logger,
    mock_sentence_queue,
    mock_interrupt,
    mock_prompt_queue,
    mock_requests_post,
):
    """Test llm_worker() handles exceptions."""
    mock_requests_post.side_effect = Exception("Connection error")

    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return "user input"
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return "more input"

    mock_prompt_queue.get.side_effect = queue_get
    mock_interrupt.is_set.return_value = False

    from llm_worker import llm_worker

    with pytest.raises(KeyboardInterrupt):
        llm_worker()

    mock_logger.error.assert_called_once_with("LLM Error: Connection error")


@patch("llm_worker.requests.post")
@patch("llm_worker.prompt_queue")
@patch("llm_worker.interrupt_event")
@patch("llm_worker.sentence_queue")
@patch("llm_worker.logger")
@patch("llm_worker.SYSTEM_PROMPT", "You are a test character.")
def test_llm_worker_no_sep_in_response(
    mock_logger,
    mock_sentence_queue,
    mock_interrupt,
    mock_prompt_queue,
    mock_requests_post,
):
    """Test llm_worker() when response doesn't contain [SEP]."""
    mock_response = MagicMock()
    response_lines = [
        b'{"response": "Hello"}',
        b'{"response": " world"}',
    ]
    mock_response.iter_lines.return_value = iter(response_lines)
    mock_requests_post.return_value = mock_response

    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return "user input"
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return "more input"

    mock_prompt_queue.get.side_effect = queue_get
    mock_interrupt.is_set.return_value = False

    from llm_worker import llm_worker

    with pytest.raises(KeyboardInterrupt):
        llm_worker()

    # Should not queue anything if no [SEP] found
    mock_sentence_queue.put.assert_not_called()


@patch("llm_worker.requests.post")
@patch("llm_worker.prompt_queue")
@patch("llm_worker.interrupt_event")
@patch("llm_worker.sentence_queue")
@patch("llm_worker.logger")
@patch("llm_worker.SYSTEM_PROMPT", "You are a test character.")
def test_llm_worker_partial_buffer(
    mock_logger,
    mock_sentence_queue,
    mock_interrupt,
    mock_prompt_queue,
    mock_requests_post,
):
    """Test llm_worker() handles partial buffer correctly."""
    mock_response = MagicMock()
    response_lines = [
        b'{"response": "First[SEP]Second"}',  # Partial second sentence
    ]
    mock_response.iter_lines.return_value = iter(response_lines)
    mock_requests_post.return_value = mock_response

    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return "user input"
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return "more input"

    mock_prompt_queue.get.side_effect = queue_get
    mock_interrupt.is_set.return_value = False

    from llm_worker import llm_worker

    with pytest.raises(KeyboardInterrupt):
        llm_worker()

    # Should queue first sentence, keep "Second" in buffer
    mock_sentence_queue.put.assert_called_once_with("First")
