"""Object key derivation and the local-disk store.

The key tests are security tests: a filename arrives from a browser, and everything that
makes a path dangerous — traversal, absolute paths, NUL bytes, bidi overrides — arrives with
it. The store tests cover the same ground from the other side, checking that a key which
should never exist is refused even if one is constructed by hand.
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aegis.core.config import Settings
from aegis.core.errors import StorageError
from aegis.infrastructure.storage import (
    LocalObjectStore,
    build_object_store,
    derived_key,
    document_key,
    document_prefix,
    is_safe_key,
    sanitize_filename,
    upload_key,
    version_prefix,
)

CHECKSUM = "9f2c4d1e" * 8


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unpad_b64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Employee Handbook 2026.pdf", "Employee-Handbook-2026.pdf"),
            ("../../../etc/passwd", "passwd"),
            ("..\\..\\windows\\system32\\config", "config"),
            ("/absolute/path/report.docx", "report.docx"),
            ("C:\\Users\\me\\notes.txt", "notes.txt"),
            ("....//....//evil.sh", "evil.sh"),
            ("no-extension", "no-extension"),
            ("......", "file"),
            ("", "file"),
        ],
    )
    def test_dangerous_names_reduce_to_a_basename(self, raw: str, expected: str) -> None:
        assert sanitize_filename(raw) == expected

    def test_nul_and_control_bytes_are_stripped(self) -> None:
        assert sanitize_filename("re\x00port\x1f.pdf") == "report.pdf"

    def test_bidi_override_is_stripped(self) -> None:
        """``exe\u202etxt.`` renders as ``txt.exe`` in a file listing."""
        assert "\u202e" not in sanitize_filename("invoice\u202efdp.exe")

    def test_length_is_bounded(self) -> None:
        result = sanitize_filename("a" * 500 + ".pdf")
        assert len(result) <= 96
        assert result.endswith(".pdf")

    def test_unicode_is_transliterated_not_dropped_entirely(self) -> None:
        assert sanitize_filename("políticas-de-seguridad.pdf") == "pol-ticas-de-seguridad.pdf"


class TestDocumentKey:
    def test_key_is_versioned_and_content_addressed(self) -> None:
        document_id = uuid.uuid4()
        key = document_key(
            document_id=document_id, version_no=3, checksum=CHECKSUM, filename="policy.pdf"
        )
        assert key == f"documents/{document_id}/v3/{CHECKSUM[:16]}/policy.pdf"

    def test_a_new_version_never_overwrites_the_old_bytes(self) -> None:
        """A citation pointing at v1 must still resolve after v2 is uploaded."""
        document_id = uuid.uuid4()
        first = document_key(
            document_id=document_id, version_no=1, checksum=CHECKSUM, filename="p.pdf"
        )
        second = document_key(
            document_id=document_id, version_no=2, checksum="a" * 64, filename="p.pdf"
        )
        assert first != second

    def test_traversal_in_the_filename_cannot_escape_the_prefix(self) -> None:
        key = document_key(
            document_id=uuid.uuid4(),
            version_no=1,
            checksum=CHECKSUM,
            filename="../../../../etc/shadow",
        )
        assert ".." not in key
        assert is_safe_key(key)

    def test_version_zero_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="starts at 1"):
            document_key(
                document_id=uuid.uuid4(), version_no=0, checksum=CHECKSUM, filename="a.pdf"
            )

    def test_short_checksum_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            document_key(document_id=uuid.uuid4(), version_no=1, checksum="ab", filename="a.pdf")

    def test_upload_key_is_date_partitioned(self) -> None:
        upload_id = uuid.uuid4()
        key = upload_key(
            upload_id=upload_id, filename="draft.docx", now=datetime(2026, 7, 5, tzinfo=UTC)
        )
        assert key == f"uploads/2026/07/{upload_id}/draft.docx"

    def test_prefixes_bracket_the_keys_they_are_meant_to_cover(self) -> None:
        document_id = uuid.uuid4()
        key = document_key(
            document_id=document_id, version_no=2, checksum=CHECKSUM, filename="a.pdf"
        )
        assert key.startswith(version_prefix(document_id, 2))
        assert key.startswith(document_prefix(document_id))
        assert not key.startswith(version_prefix(document_id, 3))

    def test_derived_artefacts_sit_beside_the_source(self) -> None:
        key = document_key(
            document_id=uuid.uuid4(), version_no=1, checksum=CHECKSUM, filename="scan.pdf"
        )
        derived = derived_key(key, kind="ocr", suffix=".txt")
        assert derived.startswith(key.rsplit("/", 1)[0])
        assert derived.endswith("/.derived/ocr.txt")


class TestIsSafeKey:
    @pytest.mark.parametrize(
        "key",
        [
            "",
            "/leading-slash",
            "a//b",
            "a/../b",
            "a/./b",
            "..",
            "with\x00nul",
            "x" * 901,
        ],
    )
    def test_rejects(self, key: str) -> None:
        assert not is_safe_key(key)

    @pytest.mark.parametrize("key", ["a", "a/b/c.pdf", "documents/x/v1/abc/file.name.pdf"])
    def test_accepts(self, key: str) -> None:
        assert is_safe_key(key)


class TestLocalObjectStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> LocalObjectStore:
        return LocalObjectStore(tmp_path / "objects", secret="unit-test-secret" * 2)

    async def test_round_trip(self, store: LocalObjectStore) -> None:
        meta = await store.put("a/b/report.pdf", b"%PDF-1.7 hello", content_type="application/pdf")
        assert meta.size_bytes == 14
        assert await store.get("a/b/report.pdf") == b"%PDF-1.7 hello"

        head = await store.head("a/b/report.pdf")
        assert head is not None
        assert head.content_type == "application/pdf"
        assert head.size_bytes == 14

    async def test_head_of_absent_object_is_none_not_an_error(
        self, store: LocalObjectStore
    ) -> None:
        assert await store.head("nothing/here.pdf") is None

    async def test_get_of_absent_object_raises(self, store: LocalObjectStore) -> None:
        with pytest.raises(StorageError, match="not found"):
            await store.get("nothing/here.pdf")

    async def test_delete_is_idempotent(self, store: LocalObjectStore) -> None:
        await store.put("x.txt", b"data", content_type="text/plain")
        await store.delete("x.txt")
        await store.delete("x.txt")
        assert await store.head("x.txt") is None

    async def test_delete_prefix_removes_the_whole_version(self, store: LocalObjectStore) -> None:
        for name in ("v1/a.txt", "v1/b.txt", "v2/c.txt"):
            await store.put(f"doc/{name}", b"x", content_type="text/plain")
        assert await store.delete_prefix("doc/v1") == 2
        assert await store.head("doc/v1/a.txt") is None
        assert await store.head("doc/v2/c.txt") is not None

    @pytest.mark.parametrize("key", ["../escape.txt", "/etc/passwd", "a/../../b.txt"])
    async def test_traversal_is_refused(self, store: LocalObjectStore, key: str) -> None:
        with pytest.raises(StorageError):
            await store.put(key, b"x", content_type="text/plain")

    async def test_a_partial_write_leaves_nothing_behind(
        self, store: LocalObjectStore, tmp_path: Path
    ) -> None:
        await store.put("atomic.bin", b"0123456789", content_type="application/octet-stream")
        leftovers = list((tmp_path / "objects").rglob("*.partial"))
        assert leftovers == []


class TestLocalPresigning:
    @pytest.fixture
    def store(self, tmp_path: Path) -> LocalObjectStore:
        return LocalObjectStore(
            tmp_path / "objects", secret="another-secret-value", base_url="http://api.test"
        )

    async def test_upload_grant_carries_the_limit_it_was_issued_with(
        self, store: LocalObjectStore
    ) -> None:
        grant = await store.presign_upload(
            "uploads/x/file.pdf", content_type="application/pdf", max_bytes=1024
        )
        assert grant.url.startswith("http://api.test")
        assert grant.max_bytes == 1024
        claims = store.verify(grant.fields["token"], operation="put")
        assert claims["k"] == "uploads/x/file.pdf"
        assert claims["max"] == 1024
        assert claims["ct"] == "application/pdf"

    async def test_a_grant_cannot_be_edited_to_name_another_object(
        self, store: LocalObjectStore
    ) -> None:
        grant = await store.presign_upload(
            "uploads/mine/file.pdf", content_type="application/pdf", max_bytes=10
        )
        payload, _, signature = grant.fields["token"].partition(".")
        claims = json.loads(_unpad_b64(payload))
        claims["k"] = "uploads/yours/file.pdf"
        tampered = _b64(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode())
        with pytest.raises(StorageError, match="signature is invalid"):
            store.verify(f"{tampered}.{signature}", operation="put")

    async def test_an_upload_grant_is_not_a_download_grant(self, store: LocalObjectStore) -> None:
        grant = await store.presign_upload(
            "uploads/x/f.pdf", content_type="application/pdf", max_bytes=10
        )
        with pytest.raises(StorageError, match="different operation"):
            store.verify(grant.fields["token"], operation="get")

    async def test_expired_grants_are_refused(self, store: LocalObjectStore) -> None:
        token = store.sign(
            "a/b.pdf",
            operation="get",
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
        with pytest.raises(StorageError, match="expired"):
            store.verify(token, operation="get")

    async def test_a_grant_signed_with_another_secret_is_refused(
        self, store: LocalObjectStore, tmp_path: Path
    ) -> None:
        other = LocalObjectStore(tmp_path / "objects", secret="a-completely-different-secret")
        token = other.sign(
            "a/b.pdf", operation="get", expires_at=datetime.now(UTC) + timedelta(minutes=5)
        )
        with pytest.raises(StorageError, match="signature is invalid"):
            store.verify(token, operation="get")

    @pytest.mark.parametrize("token", ["", "nodot", "a.b.c", "!!!.???"])
    async def test_malformed_grants_raise_rather_than_returning_none(
        self, store: LocalObjectStore, token: str
    ) -> None:
        with pytest.raises(StorageError):
            store.verify(token, operation="get")

    async def test_download_url_is_signed(self, store: LocalObjectStore) -> None:
        url = await store.presign_download("a/b.pdf", filename="Report 2026.pdf")
        assert url.startswith("http://api.test")
        token = url.partition("token=")[2]
        claims = store.verify(token, operation="get")
        assert claims["fn"] == "Report 2026.pdf"


class TestFactory:
    def test_local_backend_is_selected_by_configuration(self, settings: Settings) -> None:
        store = build_object_store(settings.model_copy(update={"storage_backend": "local"}))
        assert isinstance(store, LocalObjectStore)

    def test_unknown_backend_is_a_configuration_error(self, settings: Settings) -> None:
        from aegis.core.errors import ConfigurationError

        with pytest.raises(ConfigurationError, match="unknown storage backend"):
            build_object_store(settings.model_copy(update={"storage_backend": "gridfs"}))
