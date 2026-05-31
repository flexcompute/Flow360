"""Tests for CloudFileCache size-based LRU disk cache."""

import time
from pathlib import Path

import pytest

from flow360.cloud.file_cache import CloudFileCache


@pytest.fixture()
def cache(tmp_path):
    """A small cache (1 KB limit) rooted in a temp directory."""
    return CloudFileCache(cache_root=tmp_path / "cache", max_size_bytes=1024)


class TestGetPut:
    def test_roundtrip(self, cache):
        data = b"hello world"
        cache.put("ns", "res1", "file.bin", data)
        assert cache.get("ns", "res1", "file.bin") == data

    def test_get_miss_returns_none(self, cache):
        assert cache.get("ns", "nonexistent", "file.bin") is None

    def test_multiple_files_per_resource(self, cache):
        cache.put("ns", "res1", "a.bin", b"aaa")
        cache.put("ns", "res1", "b.bin", b"bbb")
        assert cache.get("ns", "res1", "a.bin") == b"aaa"
        assert cache.get("ns", "res1", "b.bin") == b"bbb"

    def test_separate_namespaces(self, cache):
        cache.put("ns1", "res", "f.bin", b"one")
        cache.put("ns2", "res", "f.bin", b"two")
        assert cache.get("ns1", "res", "f.bin") == b"one"
        assert cache.get("ns2", "res", "f.bin") == b"two"


class TestEviction:
    def test_evicts_oldest_when_over_budget(self, tmp_path):
        # 500-byte budget: writing two 300-byte entries should evict the first
        cache = CloudFileCache(cache_root=tmp_path / "cache", max_size_bytes=500)
        cache.put("ns", "old", "f.bin", b"x" * 300)
        time.sleep(0.05)  # ensure distinct mtime
        cache.put("ns", "new", "f.bin", b"y" * 300)

        assert cache.get("ns", "old", "f.bin") is None
        assert cache.get("ns", "new", "f.bin") == b"y" * 300

    def test_lru_order_respected(self, tmp_path):
        cache = CloudFileCache(cache_root=tmp_path / "cache", max_size_bytes=800)

        cache.put("ns", "A", "f.bin", b"a" * 250)
        time.sleep(0.05)
        cache.put("ns", "B", "f.bin", b"b" * 250)
        time.sleep(0.05)

        # Touch A so B becomes oldest
        cache.get("ns", "A", "f.bin")
        time.sleep(0.05)

        # This should evict B (oldest access), not A
        cache.put("ns", "C", "f.bin", b"c" * 400)

        assert cache.get("ns", "A", "f.bin") == b"a" * 250
        assert cache.get("ns", "B", "f.bin") is None
        assert cache.get("ns", "C", "f.bin") == b"c" * 400

    def test_no_eviction_when_within_budget(self, cache):
        cache.put("ns", "r1", "f.bin", b"x" * 100)
        cache.put("ns", "r2", "f.bin", b"y" * 100)
        assert cache.get("ns", "r1", "f.bin") == b"x" * 100
        assert cache.get("ns", "r2", "f.bin") == b"y" * 100

    def test_skip_entry_exceeding_total_budget(self, tmp_path):
        """A single entry larger than the entire cache budget is silently skipped."""
        cache = CloudFileCache(cache_root=tmp_path / "cache", max_size_bytes=500)
        cache.put("ns", "small", "f.bin", b"x" * 100)
        cache.put("ns", "huge", "f.bin", b"y" * 1000)  # exceeds 500-byte budget

        # Oversized entry was not cached
        assert cache.get("ns", "huge", "f.bin") is None
        # Existing entry was not evicted
        assert cache.get("ns", "small", "f.bin") == b"x" * 100

    def test_no_self_eviction_during_multi_file_put(self, tmp_path):
        """Putting a second file for the same resource must not evict the first."""
        # Budget fits one resource with two files (~600 bytes) but not two resources
        cache = CloudFileCache(cache_root=tmp_path / "cache", max_size_bytes=700)
        cache.put("ns", "res", "manifest.json", b"m" * 300)
        time.sleep(0.05)
        # Second file for same resource: should NOT evict its own manifest
        cache.put("ns", "res", "body.bin", b"b" * 300)

        assert cache.get("ns", "res", "manifest.json") == b"m" * 300
        assert cache.get("ns", "res", "body.bin") == b"b" * 300

    def test_overwrite_accounts_for_existing_size(self, tmp_path):
        """Overwriting a file should not over-count size pressure."""
        # Budget = 500. Put 300, then overwrite with 300 again.
        # Net delta is 0, so no eviction should be needed.
        cache = CloudFileCache(cache_root=tmp_path / "cache", max_size_bytes=500)
        cache.put("ns", "other", "f.bin", b"o" * 200)
        time.sleep(0.05)
        cache.put("ns", "res", "f.bin", b"x" * 300)
        time.sleep(0.05)
        # Overwrite with same size — should not evict "other"
        cache.put("ns", "res", "f.bin", b"y" * 300)

        assert cache.get("ns", "other", "f.bin") == b"o" * 200
        assert cache.get("ns", "res", "f.bin") == b"y" * 300


class TestDisabled:
    def test_disabled_cache_returns_none(self, tmp_path):
        cache = CloudFileCache(cache_root=tmp_path / "cache", max_size_bytes=1024)
        cache._disabled = True
        cache.put("ns", "res", "f.bin", b"data")
        assert cache.get("ns", "res", "f.bin") is None

    def test_put_on_write_failure_disables_cache(self, tmp_path, monkeypatch):
        """OSError during write disables the cache (cross-platform)."""
        cache = CloudFileCache(cache_root=tmp_path / "cache", max_size_bytes=1024)

        original_mkdir = Path.mkdir

        def failing_mkdir(self, *args, **kwargs):
            raise OSError("simulated disk write failure")

        monkeypatch.setattr(Path, "mkdir", failing_mkdir)

        cache.put("ns", "res", "f.bin", b"data")
        assert cache._disabled


class TestLastAccess:
    def test_put_creates_last_access_sentinel(self, cache, tmp_path):
        cache.put("ns", "res1", "file.bin", b"data")
        sentinel = tmp_path / "cache" / "ns" / "res1" / ".last_access"
        assert sentinel.exists()

    def test_get_updates_last_access_sentinel(self, cache, tmp_path):
        cache.put("ns", "res1", "file.bin", b"data")
        sentinel = tmp_path / "cache" / "ns" / "res1" / ".last_access"
        mtime_before = sentinel.stat().st_mtime
        time.sleep(0.05)
        cache.get("ns", "res1", "file.bin")
        assert sentinel.stat().st_mtime >= mtime_before
