"""Quick verification script for internal embedding and LLM adapters."""

from __future__ import annotations

import argparse
from typing import Sequence

from llama_index.core.llms import ChatMessage

from smak.bridge.models import InternalNomicEmbedding, build_internal_llm


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text",
        default="Hello from SMAK",
        help="Text to embed and send to the LLM.",
    )
    parser.add_argument(
        "--provider",
        default="qwen",
        choices=["qwen", "gpt-oss", "gpt_oss"],
        help="Internal LLM provider.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override the LLM model name.",
    )
    parser.add_argument(
        "--llm-api-base",
        default=None,
        help="Override the internal LLM API base URL.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override the Nomic embedding model name.",
    )
    parser.add_argument(
        "--embedding-api-base",
        default=None,
        help="Override the Nomic embedding API base URL.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    embedder = InternalNomicEmbedding(
        api_base=args.embedding_api_base,
        model=args.embedding_model,
    )
    embedding = embedder.get_text_embedding(args.text)
    print("Embedding vector length:", len(embedding))

    llm = build_internal_llm(
        provider=args.provider,
        model=args.llm_model,
        api_base=args.llm_api_base,
    )
    response = llm.chat([ChatMessage(role="user", content=args.text)])
    print("LLM response:", response.message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
