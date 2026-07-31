"""Embedding providers, the cache, and the sparse encoder.

Every remote provider is exercised through an ``httpx`` mock transport rather than a patched
method, so the assertions cover what actually goes on the wire: the payload shape, the
``input_type`` asymmetry, batching, ordering, and the retry policy.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from aegis.core.errors import ProviderError
from aegis.domain.values import EmbeddingVector
from aegis.rag.embeddings import (
    Bm25SparseEncoder,
    CachedEmbedder,
    CohereEmbedder,
    CorpusStatistics,
    FakeEmbedder,
    OpenAIEmbedder,
    TeiEmbedder,
    VoyageEmbedder,
    l2_normalize,
    tokenize,
)

DIM = 8


def _vector(seed: float = 0.1) -> list[float]:
    return [seed] * DIM


class _Recorder:
    """Mock transport that records requests and replays scripted responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = list(responses)

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if len(self._responses) > 1:
                return self._responses.pop(0)
            return self._responses[0]

        return httpx.MockTransport(handler)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport())

    def payload(self, index: int = 0) -> dict[str, Any]:
        parsed: dict[str, Any] = json.loads(self.requests[index].content)
        return parsed


def _openai_response(count: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": [{"index": i, "embedding": _vector(0.1 * (i + 1))} for i in range(count)]},
    )


class TestOpenAI:
    async def test_sends_model_input_and_dimensions(self) -> None:
        recorder = _Recorder(_openai_response(2))
        embedder = OpenAIEmbedder(
            model="text-embedding-3-large", dim=DIM, api_key="k", client=recorder.client()
        )
        vectors = await embedder.embed_documents(["one", "two"])

        assert len(vectors) == 2
        assert all(v.dim == DIM for v in vectors)
        payload = recorder.payload()
        assert payload["model"] == "text-embedding-3-large"
        assert payload["input"] == ["one", "two"]
        assert payload["dimensions"] == DIM

    async def test_results_are_reordered_by_index(self) -> None:
        """A misordered batch would attach every vector to the wrong chunk, silently."""
        response = httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": _vector(0.9)},
                    {"index": 0, "embedding": _vector(0.1)},
                ]
            },
        )
        embedder = OpenAIEmbedder(
            model="text-embedding-3-large",
            dim=DIM,
            api_key="k",
            client=_Recorder(response).client(),
        )
        vectors = await embedder.embed_documents(["first", "second"])
        assert vectors[0].values[0] == pytest.approx(0.1)
        assert vectors[1].values[0] == pytest.approx(0.9)

    async def test_batches_respect_batch_size(self) -> None:
        recorder = _Recorder(_openai_response(2))
        embedder = OpenAIEmbedder(
            model="text-embedding-3-large",
            dim=DIM,
            api_key="k",
            batch_size=2,
            client=recorder.client(),
        )
        await embedder.embed_documents(["a", "b", "c", "d"])
        assert len(recorder.requests) == 2

    async def test_authorization_header_is_sent(self) -> None:
        recorder = _Recorder(_openai_response(1))
        embedder = OpenAIEmbedder(
            model="text-embedding-3-large", dim=DIM, api_key="secret", client=recorder.client()
        )
        await embedder.embed_query("hello")
        assert recorder.requests[0].headers["authorization"] == "Bearer secret"


class TestErrorHandling:
    async def test_wrong_dimension_is_a_hard_failure(self) -> None:
        response = httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})
        embedder = OpenAIEmbedder(
            model="text-embedding-3-large",
            dim=DIM,
            api_key="k",
            client=_Recorder(response).client(),
        )
        with pytest.raises(ProviderError, match="dimension"):
            await embedder.embed_query("hello")

    async def test_missing_result_is_a_hard_failure(self) -> None:
        embedder = OpenAIEmbedder(
            model="text-embedding-3-large",
            dim=DIM,
            api_key="k",
            client=_Recorder(_openai_response(1)).client(),
        )
        with pytest.raises(ProviderError, match="expected 2 embeddings"):
            await embedder.embed_documents(["one", "two"])

    async def test_client_error_is_not_retried(self) -> None:
        recorder = _Recorder(httpx.Response(401, json={"error": {"message": "bad key"}}))
        embedder = OpenAIEmbedder(model="m", dim=DIM, api_key="k", client=recorder.client())
        with pytest.raises(ProviderError) as caught:
            await embedder.embed_query("hello")
        assert caught.value.retryable is False
        assert len(recorder.requests) == 1

    async def test_server_error_is_retried_then_succeeds(self, monkeypatch: Any) -> None:
        async def no_sleep(_: float) -> None:
            return None

        monkeypatch.setattr("aegis.rag.embeddings.base.asyncio.sleep", no_sleep)
        recorder = _Recorder(httpx.Response(503, json={}), _openai_response(1))
        embedder = OpenAIEmbedder(model="m", dim=DIM, api_key="k", client=recorder.client())
        vector = await embedder.embed_query("hello")
        assert vector.dim == DIM
        assert len(recorder.requests) == 2

    async def test_error_detail_never_contains_the_text(self) -> None:
        recorder = _Recorder(httpx.Response(400, json={"error": {"message": "too long"}}))
        embedder = OpenAIEmbedder(model="m", dim=DIM, api_key="k", client=recorder.client())
        with pytest.raises(ProviderError) as caught:
            await embedder.embed_query("CONFIDENTIAL SALARY DATA")
        assert "CONFIDENTIAL" not in str(caught.value)

    async def test_health_is_false_when_the_provider_is_down(self) -> None:
        recorder = _Recorder(httpx.Response(500, json={}))
        embedder = OpenAIEmbedder(model="m", dim=DIM, api_key="k", client=recorder.client())
        embedder._client = recorder.client()
        assert await embedder.health() is False


class TestAsymmetricProviders:
    async def test_voyage_marks_documents_and_queries_differently(self) -> None:
        recorder = _Recorder(_openai_response(1))
        embedder = VoyageEmbedder(model="voyage-3", dim=DIM, api_key="k", client=recorder.client())
        await embedder.embed_documents(["a"])
        await embedder.embed_query("a")
        assert recorder.payload(0)["input_type"] == "document"
        assert recorder.payload(1)["input_type"] == "query"

    async def test_cohere_uses_its_own_input_type_values(self) -> None:
        response = httpx.Response(200, json={"embeddings": {"float": [_vector()]}})
        recorder = _Recorder(response)
        embedder = CohereEmbedder(
            model="embed-english-v3.0", dim=DIM, api_key="k", client=recorder.client()
        )
        await embedder.embed_documents(["a"])
        await embedder.embed_query("a")
        assert recorder.payload(0)["input_type"] == "search_document"
        assert recorder.payload(1)["input_type"] == "search_query"

    async def test_tei_adds_the_bge_v15_query_instruction(self) -> None:
        recorder = _Recorder(httpx.Response(200, json=[_vector()]))
        embedder = TeiEmbedder(model="BAAI/bge-large-en-v1.5", dim=DIM, client=recorder.client())
        await embedder.embed_documents(["a policy paragraph"])
        await embedder.embed_query("what is the policy")
        assert recorder.payload(0)["inputs"] == ["a policy paragraph"]
        assert recorder.payload(1)["inputs"][0].startswith("Represent this sentence")

    async def test_tei_does_not_add_the_instruction_for_m3(self) -> None:
        recorder = _Recorder(httpx.Response(200, json=[_vector()]))
        embedder = TeiEmbedder(model="BAAI/bge-m3", dim=DIM, client=recorder.client())
        await embedder.embed_query("what is the policy")
        assert recorder.payload(0)["inputs"] == ["what is the policy"]

    async def test_tei_sends_no_authorization_header(self) -> None:
        recorder = _Recorder(httpx.Response(200, json=[_vector()]))
        embedder = TeiEmbedder(model="BAAI/bge-m3", dim=DIM, client=recorder.client())
        await embedder.embed_query("hello")
        assert "authorization" not in recorder.requests[0].headers


class TestFakeEmbedder:
    async def test_is_deterministic(self) -> None:
        embedder = FakeEmbedder(dim=64)
        first = await embedder.embed_query("annual leave policy")
        second = await embedder.embed_query("annual leave policy")
        assert first.values == second.values

    async def test_shared_vocabulary_is_closer_than_unrelated_text(self) -> None:
        """Retrieval tests need ranking to mean something, not just to run."""
        embedder = FakeEmbedder(dim=256)
        query = await embedder.embed_query("annual leave carry over policy")
        related, unrelated = await embedder.embed_documents(
            [
                "the annual leave policy allows carry over of five days",
                "the firewall change request process requires two approvals",
            ]
        )
        assert _cosine(query.values, related.values) > _cosine(query.values, unrelated.values)

    async def test_vectors_are_unit_length(self) -> None:
        vector = await FakeEmbedder(dim=32).embed_query("hello world")
        assert sum(v * v for v in vector.values) == pytest.approx(1.0)


class _MemoryCache:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.reads = 0

    async def get(self, key: str) -> bytes | None:
        return self.store.get(key)

    async def set(self, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
        self.store[key] = value

    async def get_many(self, keys: Any) -> dict[str, bytes]:
        self.reads += 1
        return {k: self.store[k] for k in keys if k in self.store}

    async def set_many(self, items: dict[str, bytes], *, ttl_seconds: int | None = None) -> None:
        self.store.update(items)

    async def delete(self, *keys: str) -> int:
        return len([self.store.pop(k) for k in keys if k in self.store])

    async def incr(self, key: str, *, ttl_seconds: int | None = None) -> int:
        return 1

    async def add(self, key: str, value: bytes, *, ttl_seconds: int) -> bool:
        return self.store.setdefault(key, value) is value

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _CountingEmbedder:
    name = "counting"
    model = "counting-v1"
    dim = 8

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_documents(self, texts: list[str]) -> list[EmbeddingVector]:
        self.calls.append(list(texts))
        return [
            EmbeddingVector(
                values=tuple(l2_normalize([float(len(t))] * self.dim)), model=self.model
            )
            for t in texts
        ]

    async def embed_query(self, text: str) -> EmbeddingVector:
        self.calls.append([text])
        return (await self.embed_documents([text]))[0]

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class TestCache:
    async def test_second_call_hits_the_cache(self) -> None:
        inner = _CountingEmbedder()
        cached = CachedEmbedder(inner, _MemoryCache())
        first = await cached.embed_documents(["alpha", "beta"])
        inner.calls.clear()
        second = await cached.embed_documents(["alpha", "beta"])
        assert not inner.calls
        # Approximate, not exact: cached vectors are stored as float32, which is what the
        # vector stores keep anyway.
        for original, restored in zip(first, second, strict=True):
            assert restored.values == pytest.approx(original.values, rel=1e-6)

    async def test_only_missing_texts_are_embedded(self) -> None:
        inner = _CountingEmbedder()
        cached = CachedEmbedder(inner, _MemoryCache())
        await cached.embed_documents(["alpha"])
        inner.calls.clear()
        await cached.embed_documents(["alpha", "beta"])
        assert inner.calls == [["beta"]]

    async def test_repeats_within_a_batch_are_embedded_once(self) -> None:
        inner = _CountingEmbedder()
        cached = CachedEmbedder(inner, _MemoryCache())
        result = await cached.embed_documents(["same", "same", "same"])
        assert inner.calls == [["same"]]
        assert len({v.values for v in result}) == 1
        assert len(result) == 3

    async def test_queries_and_documents_use_different_keys(self) -> None:
        """Asymmetric models embed the two differently; sharing a key serves a wrong vector."""
        cache = _MemoryCache()
        cached = CachedEmbedder(_CountingEmbedder(), cache)
        await cached.embed_documents(["overlap"])
        await cached.embed_query("overlap")
        assert len(cache.store) == 2

    async def test_model_change_does_not_reuse_old_vectors(self) -> None:
        cache = _MemoryCache()
        first = _CountingEmbedder()
        await CachedEmbedder(first, cache).embed_documents(["text"])

        second = _CountingEmbedder()
        second.model = "counting-v2"
        second.calls.clear()
        await CachedEmbedder(second, cache).embed_documents(["text"])
        assert second.calls == [["text"]]

    async def test_cache_failure_does_not_fail_ingestion(self) -> None:
        class Broken(_MemoryCache):
            async def get_many(self, keys: Any) -> dict[str, bytes]:
                raise ConnectionError("redis is down")

            async def set_many(
                self, items: dict[str, bytes], *, ttl_seconds: int | None = None
            ) -> None:
                raise ConnectionError("redis is down")

        cached = CachedEmbedder(_CountingEmbedder(), Broken())
        assert len(await cached.embed_documents(["a", "b"])) == 2

    async def test_corrupt_entry_is_ignored(self) -> None:
        cache = _MemoryCache()
        inner = _CountingEmbedder()
        cached = CachedEmbedder(inner, cache)
        await cached.embed_documents(["text"])
        cache.store = dict.fromkeys(cache.store, b"truncated")
        inner.calls.clear()
        assert len(await cached.embed_documents(["text"])) == 1
        assert inner.calls == [["text"]]


class TestSparse:
    def test_identifiers_survive_tokenization(self) -> None:
        assert "sku-44921" in tokenize("Order SKU-44921 shipped")
        assert "cve-2024-1234" in tokenize("Patch for CVE-2024-1234 is required")
        assert "policy_hr_014" in tokenize("See policy_hr_014 for details")

    def test_stopwords_are_dropped_but_negations_are_kept(self) -> None:
        tokens = tokenize("this is not a fault of the employee")
        assert "the" not in tokens
        assert "not" in tokens

    def test_document_and_query_vectors_differ(self) -> None:
        encoder = Bm25SparseEncoder()
        document = encoder.encode_document("annual leave carry over rules")
        query = encoder.encode_query("annual leave")
        assert document.nnz > query.nnz
        assert set(query.indices) <= set(document.indices)

    def test_dot_product_ranks_the_matching_document_first(self) -> None:
        encoder = Bm25SparseEncoder()
        relevant = encoder.encode_document("the parental leave entitlement is sixteen weeks")
        other = encoder.encode_document("the firewall change process requires two approvals")
        query = encoder.encode_query("parental leave entitlement")
        assert _sparse_dot(query, relevant) > _sparse_dot(query, other)

    def test_indices_are_stable_across_instances(self) -> None:
        first = Bm25SparseEncoder().encode_document("stable term set", observe=False)
        second = Bm25SparseEncoder().encode_document("stable term set", observe=False)
        assert first.indices == second.indices

    def test_idf_is_never_negative(self) -> None:
        """A negative weight would penalise a chunk for containing the query term."""
        stats = CorpusStatistics()
        encoder = Bm25SparseEncoder(stats)
        for _ in range(10):
            encoder.encode_document("ubiquitous term appears everywhere")
        query = encoder.encode_query("ubiquitous")
        assert all(v >= 0.0 for v in query.values)

    def test_empty_text_yields_an_empty_vector(self) -> None:
        assert Bm25SparseEncoder().encode_document("   ").nnz == 0

    def test_length_normalisation_discounts_repetition(self) -> None:
        encoder = Bm25SparseEncoder()
        once = encoder.encode_document("audit", observe=False)
        many = encoder.encode_document(" ".join(["audit"] * 20), observe=False)
        assert many.values[0] < once.values[0] * 20


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _sparse_dot(query: Any, document: Any) -> float:
    weights: dict[int, float] = dict(zip(document.indices, document.values, strict=True))
    return float(
        sum(
            value * weights.get(index, 0.0)
            for index, value in zip(query.indices, query.values, strict=True)
        )
    )
