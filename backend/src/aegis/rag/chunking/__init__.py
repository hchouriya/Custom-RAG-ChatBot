"""Chunking: parsed blocks in, token-budgeted chunks with locators out.

Five strategies over one packer. See :mod:`aegis.rag.chunking.base` for why the split is
"decide boundaries" versus "fill to budget", and :mod:`aegis.rag.chunking.router` for how a
single document uses several strategies at once.
"""

from aegis.rag.chunking.base import Segment, build_context_header, pack, split_sentences
from aegis.rag.chunking.code import CodeChunker
from aegis.rag.chunking.markdown import MarkdownChunker
from aegis.rag.chunking.recursive import RecursiveChunker
from aegis.rag.chunking.router import DefaultChunkRouter, build_chunk_router
from aegis.rag.chunking.semantic import SemanticChunker
from aegis.rag.chunking.table import TableChunker
from aegis.rag.chunking.tokenizer import TokenCounter, get_token_counter

__all__ = [
    "CodeChunker",
    "DefaultChunkRouter",
    "MarkdownChunker",
    "RecursiveChunker",
    "Segment",
    "SemanticChunker",
    "TableChunker",
    "TokenCounter",
    "build_chunk_router",
    "build_context_header",
    "get_token_counter",
    "pack",
    "split_sentences",
]
