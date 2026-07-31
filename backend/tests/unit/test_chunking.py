"""Chunking behaviour.

The assertions here are the ones with retrieval consequences: budgets that hold, boundaries
that land on sentences, tables that keep their headers, offsets that point back at the source,
and no chunk that is a duplicate of its neighbour's tail.
"""

from __future__ import annotations

import itertools

import pytest

from aegis.domain.enums import ChunkType
from aegis.domain.ports.chunker import ChunkingConfig, ProtoChunk
from aegis.domain.ports.parser import DocumentBlock, PageInfo, ParsedDocument
from aegis.domain.values import EmbeddingVector
from aegis.rag.chunking import (
    CodeChunker,
    MarkdownChunker,
    RecursiveChunker,
    SemanticChunker,
    TableChunker,
    build_chunk_router,
    get_token_counter,
    split_sentences,
)
from aegis.rag.chunking.tokenizer import HeuristicCounter

COUNTER = get_token_counter()

SMALL = ChunkingConfig(target_tokens=60, overlap_pct=10, min_tokens=20, max_tokens=120)


def _prose(sentences: int, *, word: str = "policy") -> str:
    return " ".join(
        f"The {word} requires that all staff complete step {i} before the deadline."
        for i in range(sentences)
    )


def _block(text: str, **kwargs: object) -> DocumentBlock:
    block = DocumentBlock(text=text, **kwargs)  # type: ignore[arg-type]
    block.char_end = block.char_start + len(text)
    return block


class TestTokenizer:
    def test_counts_are_positive_and_monotonic(self) -> None:
        short = COUNTER.count("one two three")
        longer = COUNTER.count("one two three four five six seven eight")
        assert 0 < short < longer

    def test_truncate_respects_budget(self) -> None:
        text = _prose(20)
        cut = COUNTER.truncate(text, 25)
        assert COUNTER.count(cut) <= 25
        assert text.startswith(cut[: len(cut) - 1])

    def test_heuristic_does_not_undercount_code(self) -> None:
        """The chars/4 rule these budgets replace underestimates punctuation-dense text."""
        code = "for (let i=0; i<n; i++) { total += values[i].amount; }"
        assert HeuristicCounter().count(code) > len(code) / 4

    def test_heuristic_handles_cjk(self) -> None:
        assert HeuristicCounter().count("会社の方針は毎年見直されます") >= 10


class TestSentenceSplitting:
    def test_offsets_point_into_the_source(self) -> None:
        text = "First sentence here. Second sentence follows."
        for sentence, start, end in split_sentences(text):
            assert text[start:end] == sentence

    @pytest.mark.parametrize(
        "text",
        [
            "See Dr. Smith for approval. Then file the form.",
            "Refer to Section 4.2. The rule applies to contractors.",
            "Submit by Jan. 5 each year. Late filings are rejected.",
        ],
    )
    def test_abbreviations_do_not_split(self, text: str) -> None:
        assert len(split_sentences(text)) == 2


class TestRecursive:
    def test_chunks_stay_within_max_tokens(self) -> None:
        blocks = [_block(_prose(40))]
        chunks = RecursiveChunker(COUNTER).split(blocks, SMALL)
        assert chunks
        assert all(c.token_count <= SMALL.max_tokens for c in chunks)

    def test_boundaries_land_on_sentence_ends(self) -> None:
        chunks = RecursiveChunker(COUNTER).split([_block(_prose(40))], SMALL)
        assert all(c.content.rstrip().endswith(".") for c in chunks)

    def test_overlap_is_shared_text_not_duplicate_chunks(self) -> None:
        chunks = RecursiveChunker(COUNTER).split([_block(_prose(40))], SMALL)
        contents = [c.content for c in chunks]
        assert len(set(contents)) == len(contents)
        first_tail = chunks[0].content.split(". ")[-1]
        assert first_tail[:20] in chunks[1].content

    def test_ordinals_are_dense_and_ordered(self) -> None:
        chunks = RecursiveChunker(COUNTER).split([_block(_prose(40))], SMALL)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))

    def test_offsets_are_non_decreasing(self) -> None:
        chunks = RecursiveChunker(COUNTER).split([_block(_prose(40))], SMALL)
        assert all(a.char_start <= b.char_start for a, b in itertools.pairwise(chunks))
        assert all(c.char_end > c.char_start for c in chunks)

    def test_heading_opens_the_next_chunk(self) -> None:
        blocks = [
            _block("Section 1 Overview", block_type=ChunkType.HEADING, heading_level=1),
            _block(_prose(6)),
            _block("Section 2 Termination", block_type=ChunkType.HEADING, heading_level=1),
            _block(_prose(6, word="notice")),
        ]
        chunks = RecursiveChunker(COUNTER).split(blocks, SMALL)
        assert any(c.content.startswith("Section 2 Termination") for c in chunks)
        assert not any(c.content.strip() == "Section 2 Termination" for c in chunks)

    def test_short_paragraph_is_not_its_own_chunk(self) -> None:
        blocks = [_block("Short note."), _block(_prose(10))]
        chunks = RecursiveChunker(COUNTER).split(blocks, SMALL)
        assert chunks[0].content.startswith("Short note.")
        assert chunks[0].token_count >= SMALL.min_tokens

    def test_unpunctuated_wall_of_text_is_still_split(self) -> None:
        blocks = [_block(" ".join(f"word{i}" for i in range(600)))]
        chunks = RecursiveChunker(COUNTER).split(blocks, SMALL)
        assert len(chunks) > 1
        assert all(c.token_count <= SMALL.max_tokens for c in chunks)


class TestContextHeaders:
    def test_heading_path_is_prepended_for_embedding(self) -> None:
        config = ChunkingConfig(
            target_tokens=60,
            min_tokens=20,
            max_tokens=120,
            document_title="Employee Handbook",
        )
        blocks = [_block(_prose(8), heading_path=("Leave", "Parental Leave"))]
        chunk = RecursiveChunker(COUNTER).split(blocks, config)[0]
        assert chunk.context_header == "Employee Handbook > Leave > Parental Leave"
        assert chunk.embed_text.startswith("Employee Handbook > Leave > Parental Leave")
        assert chunk.content not in chunk.context_header

    def test_headers_can_be_disabled(self) -> None:
        config = ChunkingConfig(contextual_headers=False, document_title="Handbook")
        chunk = RecursiveChunker(COUNTER).split([_block(_prose(8))], config)[0]
        assert chunk.context_header is None
        assert chunk.embed_text == chunk.content


class TestMarkdown:
    def _blocks(self) -> list[DocumentBlock]:
        return [
            _block("Onboarding", block_type=ChunkType.HEADING, heading_level=1),
            _block(_prose(4, word="onboarding"), heading_path=("Onboarding",)),
            _block("Offboarding", block_type=ChunkType.HEADING, heading_level=1),
            _block(_prose(4, word="offboarding"), heading_path=("Offboarding",)),
        ]

    def test_sections_do_not_bleed_into_each_other(self) -> None:
        chunks = MarkdownChunker(COUNTER).split(self._blocks(), SMALL)
        assert not any("onboarding" in c.content and "offboarding" in c.content for c in chunks)

    def test_section_is_recorded_on_the_chunk(self) -> None:
        chunks = MarkdownChunker(COUNTER).split(self._blocks(), SMALL)
        assert {c.section for c in chunks} == {"Onboarding", "Offboarding"}

    def test_oversized_section_is_split_and_keeps_its_path(self) -> None:
        blocks = [
            _block("Retention", block_type=ChunkType.HEADING, heading_level=1),
            _block(_prose(40), heading_path=("Retention",)),
        ]
        chunks = MarkdownChunker(COUNTER).split(blocks, SMALL)
        assert len(chunks) > 1
        assert all(c.heading_path == ("Retention",) for c in chunks[1:])


class TestTables:
    TABLE = "\n".join(
        [
            "| region | quarter | revenue |",
            "| --- | --- | --- |",
            *[f"| region-{i} | Q{i % 4 + 1} | {i * 1000} |" for i in range(60)],
        ]
    )

    def _chunks(self, config: ChunkingConfig = SMALL) -> list[ProtoChunk]:
        block = _block(self.TABLE, block_type=ChunkType.TABLE)
        return TableChunker(COUNTER).split([block], config)

    def test_every_part_repeats_the_header(self) -> None:
        chunks = self._chunks()
        assert len(chunks) > 1
        assert all(c.content.startswith("| region | quarter | revenue |") for c in chunks)

    def test_rows_are_never_split(self) -> None:
        for chunk in self._chunks():
            for line in chunk.content.split("\n"):
                assert line.startswith("|") and line.rstrip().endswith("|")

    def test_no_row_is_lost_or_duplicated(self) -> None:
        rows = [
            line
            for chunk in self._chunks()
            for line in chunk.content.split("\n")[2:]
            if line.strip()
        ]
        assert len(rows) == 60
        assert len(set(rows)) == 60

    def test_small_table_is_one_chunk(self) -> None:
        block = _block("| a | b |\n| --- | --- |\n| 1 | 2 |", block_type=ChunkType.TABLE)
        chunks = TableChunker(COUNTER).split([block], SMALL)
        assert len(chunks) == 1
        assert chunks[0].chunk_type is ChunkType.TABLE

    def test_parts_are_labelled(self) -> None:
        chunks = self._chunks()
        assert chunks[0].metadata["part"] == 1
        assert chunks[0].metadata["parts"] == len(chunks)


class TestCode:
    SOURCE = "\n".join(
        [
            "def load_config(path):",
            '    """Read the config."""',
            "    with open(path) as handle:",
            "        return json.load(handle)",
            "",
            "@retry",
            "def fetch(url):",
            "    response = client.get(url)",
            "    response.raise_for_status()",
            "    return response.json()",
        ]
    )

    def test_definitions_are_not_split_mid_body(self) -> None:
        block = _block(self.SOURCE, block_type=ChunkType.CODE)
        chunks = CodeChunker(COUNTER).split([block], ChunkingConfig(target_tokens=30, min_tokens=5))
        bodies = "\n".join(c.content for c in chunks)
        assert "def fetch(url):" in bodies
        for chunk in chunks:
            if "def fetch" in chunk.content:
                assert "return response.json()" in chunk.content

    def test_decorator_stays_with_its_function(self) -> None:
        block = _block(self.SOURCE, block_type=ChunkType.CODE)
        chunks = CodeChunker(COUNTER).split([block], ChunkingConfig(target_tokens=30, min_tokens=5))
        for chunk in chunks:
            if "@retry" in chunk.content:
                assert "def fetch(url):" in chunk.content

    def test_chunk_type_is_code(self) -> None:
        block = _block(self.SOURCE, block_type=ChunkType.CODE)
        chunks = CodeChunker(COUNTER).split([block], SMALL)
        assert all(c.chunk_type is ChunkType.CODE for c in chunks)


class TestSemantic:
    def test_breaks_where_the_topic_changes(self) -> None:
        first = _prose(6, word="firewall")
        second = _prose(6, word="parental leave")
        chunks = SemanticChunker(COUNTER).split(
            [_block(f"{first} {second}")],
            ChunkingConfig(target_tokens=400, min_tokens=10, max_tokens=800),
        )
        assert len(chunks) >= 2
        assert not any("firewall" in c.content and "parental leave" in c.content for c in chunks)

    @pytest.mark.asyncio
    async def test_embedding_failure_falls_back_to_lexical(self) -> None:
        class Broken:
            name = "broken"
            model = "broken"
            dim = 3

            async def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
                raise RuntimeError("provider down")

            async def embed_query(self, text: str) -> None: ...
            async def health(self) -> bool:
                return False

            async def close(self) -> None: ...

        chunker = SemanticChunker(COUNTER, Broken())  # type: ignore[arg-type]
        chunks = await chunker.asplit([_block(_prose(20))], SMALL)
        assert chunks


class TestRouter:
    def _document(self) -> ParsedDocument:
        return ParsedDocument(
            blocks=[
                _block("Rates", block_type=ChunkType.HEADING, heading_level=1),
                _block(_prose(6, word="rate"), heading_path=("Rates",)),
                _block(
                    "| tier | price |\n| --- | --- |\n| gold | 100 |",
                    block_type=ChunkType.TABLE,
                    heading_path=("Rates",),
                ),
                _block("Usage", block_type=ChunkType.HEADING, heading_level=1),
                _block("def price(tier):\n    return TIERS[tier]", block_type=ChunkType.CODE),
                _block(_prose(6, word="usage"), heading_path=("Usage",)),
            ],
            pages=[PageInfo(number=1, char_count=1200)],
            title="Pricing Guide",
        )

    @pytest.mark.asyncio
    async def test_mixed_document_uses_several_strategies(self) -> None:
        chunks = await build_chunk_router().chunk(self._document(), SMALL)
        types = {c.chunk_type for c in chunks}
        assert ChunkType.TABLE in types
        assert ChunkType.CODE in types
        assert ChunkType.TEXT in types

    @pytest.mark.asyncio
    async def test_document_order_is_preserved(self) -> None:
        chunks = await build_chunk_router().chunk(self._document(), SMALL)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))
        table_at = next(i for i, c in enumerate(chunks) if c.chunk_type is ChunkType.TABLE)
        code_at = next(i for i, c in enumerate(chunks) if c.chunk_type is ChunkType.CODE)
        assert table_at < code_at

    @pytest.mark.asyncio
    async def test_title_becomes_the_context_header_root(self) -> None:
        chunks = await build_chunk_router().chunk(self._document(), SMALL)
        assert all(
            c.context_header is None or c.context_header.startswith("Pricing Guide") for c in chunks
        )

    @pytest.mark.asyncio
    async def test_repeated_content_is_indexed_once(self) -> None:
        """The same rate table in ten appendices is one piece of information, not ten."""
        table = "| tier | price |\n| --- | --- |\n| gold | 100 |\n| silver | 50 |"
        document = ParsedDocument(
            blocks=[_block(table, block_type=ChunkType.TABLE, page=p) for p in range(1, 11)],
            title="Contract",
        )
        chunks = await build_chunk_router().chunk(document, SMALL)
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_keywords_distinguish_chunks(self) -> None:
        document = ParsedDocument(
            blocks=[
                _block(_prose(8, word="firewall")),
                _block(_prose(8, word="reimbursement")),
            ],
            title="Handbook",
        )
        chunks = await build_chunk_router(strategy="recursive").chunk(document, SMALL)
        joined = {c.ordinal: set(c.keywords) for c in chunks}
        assert any("firewall" in terms for terms in joined.values())

    @pytest.mark.asyncio
    async def test_empty_document_yields_no_chunks(self) -> None:
        assert await build_chunk_router().chunk(ParsedDocument(blocks=[]), SMALL) == []

    @pytest.mark.asyncio
    async def test_ocr_document_gets_smaller_targets(self) -> None:
        blocks = [_block(_prose(30), confidence=0.7)]
        document = ParsedDocument(blocks=blocks, title="Scan", used_ocr=True)
        base = ChunkingConfig(target_tokens=400, min_tokens=100, max_tokens=800)
        chunks = await build_chunk_router().chunk(document, base)
        assert chunks
        assert max(c.token_count for c in chunks) < base.target_tokens
