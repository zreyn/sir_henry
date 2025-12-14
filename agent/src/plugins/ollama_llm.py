"""
Custom Ollama LLM plugin for livekit-agents.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterator

import requests

from livekit.agents import llm


@dataclass
class OllamaOptions:
    model: str = "llama3.2:3b"
    host: str = "localhost:11434"
    temperature: float = 0.7


class OllamaLLM(llm.LLM):
    """
    Ollama LLM plugin for LiveKit Agents.
    Uses local Ollama server for language model inference.
    """

    def __init__(
        self,
        *,
        model: str = "llama3.2:3b",
        host: str = "localhost:11434",
        temperature: float = 0.7,
    ):
        super().__init__()
        self._opts = OllamaOptions(
            model=model,
            host=host,
            temperature=temperature,
        )

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool] | None = None,
        tool_choice: llm.ToolChoice | None = None,
        conn_options: llm.LLMConnOptions | None = None,
    ) -> "OllamaLLMStream":
        return OllamaLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )


class OllamaLLMStream(llm.LLMStream):
    """Streaming response handler for Ollama LLM."""

    def __init__(
        self,
        *,
        llm: OllamaLLM,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool] | None,
        conn_options: llm.LLMConnOptions | None,
    ):
        super().__init__(
            llm=llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options
        )
        self._llm = llm

    async def _run(self) -> None:
        """Stream the LLM response."""
        opts = self._llm._opts

        # Build the prompt from chat context
        system_prompt = ""
        messages = []

        for msg in self._chat_ctx.items:
            if isinstance(msg, llm.ChatMessage):
                if msg.role == llm.ChatRole.SYSTEM:
                    system_prompt = self._get_text_content(msg)
                elif msg.role == llm.ChatRole.USER:
                    messages.append(
                        {"role": "user", "content": self._get_text_content(msg)}
                    )
                elif msg.role == llm.ChatRole.ASSISTANT:
                    messages.append(
                        {"role": "assistant", "content": self._get_text_content(msg)}
                    )

        # Get the most recent user message as the prompt
        prompt = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                prompt = msg["content"]
                break

        # Make streaming request to Ollama
        loop = asyncio.get_event_loop()

        def _stream_request():
            resp = requests.post(
                f"http://{opts.host}/api/generate",
                json={
                    "model": opts.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": True,
                    "options": {
                        "temperature": opts.temperature,
                    },
                },
                stream=True,
            )
            for line in resp.iter_lines():
                if line:
                    yield line

        # Process the stream
        request_id = "ollama-" + str(id(self))
        full_response = ""

        try:
            for line in await loop.run_in_executor(
                None, lambda: list(_stream_request())
            ):
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    full_response += token
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            request_id=request_id,
                            choices=[
                                llm.Choice(
                                    delta=llm.ChoiceDelta(
                                        role=llm.ChatRole.ASSISTANT,
                                        content=token,
                                    ),
                                    index=0,
                                )
                            ],
                        )
                    )
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

    def _get_text_content(self, msg: llm.ChatMessage) -> str:
        """Extract text content from a chat message."""
        if isinstance(msg.content, str):
            return msg.content
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, str):
                    return item
        return ""
