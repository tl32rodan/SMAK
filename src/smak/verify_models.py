"""Shared helpers for verifying SMAK internal models."""

from __future__ import annotations

from typing import Callable

from smak.bridge.models import InternalNomicEmbedding, build_internal_llm


def verify_models(
    text: str,
    provider: str,
    llm_model: str | None,
    llm_api_base: str | None,
    embedding_model: str | None,
    embedding_api_base: str | None,
    *,
    embedder_factory: Callable[..., InternalNomicEmbedding] = InternalNomicEmbedding,
    llm_factory: Callable[..., object] = build_internal_llm,
    chat_message_factory: Callable[..., object] | None = None,
) -> tuple[int, str]:
    """Run a single embedding + LLM round-trip for quick validation."""
    if chat_message_factory is None:
        from llama_index.core.llms import ChatMessage  # local import for optional dependency

        chat_message_factory = ChatMessage
    embedder = embedder_factory(
        api_base=embedding_api_base,
        model=embedding_model,
    )
    embedding = embedder.get_text_embedding(text)

    llm = llm_factory(
        provider=provider,
        model=llm_model,
        api_base=llm_api_base,
    )
    response = llm.chat([chat_message_factory(role="user", content=text)])
    return len(embedding), response.message.content
