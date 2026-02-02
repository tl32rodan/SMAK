"""Quick verification script for internal embedding and LLM adapters."""

from __future__ import annotations

import argparse
from typing import Sequence

from smak.verify_models import verify_models


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
    embedding_length, response_text = verify_models(
        text=args.text,
        provider=args.provider,
        llm_model=args.llm_model,
        llm_api_base=args.llm_api_base,
        embedding_model=args.embedding_model,
        embedding_api_base=args.embedding_api_base,
    )
    print("Embedding vector length:", embedding_length)
    print("LLM response:", response_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
