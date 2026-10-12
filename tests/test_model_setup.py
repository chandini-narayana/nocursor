"""Model download: success, and every way it can fail, without touching the network."""

import hashlib
import runpy
import urllib.error
from pathlib import Path

import pytest

from nocursor.model_setup import (
    CHUNK_BYTES,
    MIN_EXPECTED_BYTES,
    MODEL_FILENAME,
    MODEL_URL,
    ModelDownloadError,
    default_model_path,
    download_model,
    run_download,
)

GOOD = (b"MODEL" * 300_000)[: MIN_EXPECTED_BYTES + 12345]


class FakeResponse:
    def __init__(self, data, content_length="auto", fail_after=None):
        self._data = data
        self._pos = 0
        self._fail_after = fail_after
        self.headers = {}
        if content_length == "auto":
            self.headers["Content-Length"] = str(len(data))
        elif content_length is not None:
            self.headers["Content-Length"] = content_length
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def read(self, n):
        if self._fail_after is not None and self._pos >= self._fail_after:
            raise ConnectionResetError("connection reset by peer")
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


def opener_returning(response, calls=None):
    def opener(url, timeout):
        if calls is not None:
            calls.append((url, timeout))
        return response

    return opener


def opener_raising(exc):
    def opener(url, timeout):
        raise exc

    return opener


class TestSuccess:
    def test_downloads_verifies_and_reports_a_hash(self, tmp_path):
        dest = tmp_path / "models" / MODEL_FILENAME
        calls = []
        result = download_model(dest, opener=opener_returning(FakeResponse(GOOD), calls))
        assert dest.read_bytes() == GOOD
        assert result.sha256 == hashlib.sha256(GOOD).hexdigest()
        assert result.size_bytes == len(GOOD) and result.already_present is False
        assert calls == [(MODEL_URL, 30)]  # a fixed https URL and a timeout
        assert not list(dest.parent.glob("*.part"))

    def test_creates_the_models_directory(self, tmp_path):
        dest = tmp_path / "a" / "b" / MODEL_FILENAME
        download_model(dest, opener=opener_returning(FakeResponse(GOOD)))
        assert dest.is_file()

    def test_data_is_streamed_in_chunks(self, tmp_path):
        class Counting(FakeResponse):
            reads = 0

            def read(self, n):
                Counting.reads += 1
                assert n == CHUNK_BYTES
                return super().read(n)

        download_model(tmp_path / MODEL_FILENAME, opener=opener_returning(Counting(GOOD)))
        assert Counting.reads > 2

    def test_a_missing_content_length_header_is_tolerated(self, tmp_path):
        r = download_model(tmp_path / MODEL_FILENAME,
                           opener=opener_returning(FakeResponse(GOOD, content_length=None)))
        assert r.size_bytes == len(GOOD)

    def test_an_unparseable_content_length_is_ignored(self, tmp_path):
        r = download_model(tmp_path / MODEL_FILENAME,
                           opener=opener_returning(FakeResponse(GOOD, content_length="abc")))
        assert r.size_bytes == len(GOOD)

    def test_default_location_is_the_models_folder(self):
        assert default_model_path().name == MODEL_FILENAME
        assert default_model_path().parent.name == "models"


class TestExistingModel:
    def test_an_existing_model_is_kept_without_any_network_access(self, tmp_path):
        dest = tmp_path / MODEL_FILENAME
        dest.write_bytes(GOOD)

        def never(url, timeout):
            raise AssertionError("must not download when a model exists")

        r = download_model(dest, opener=never)
        assert r.already_present is True and r.sha256 == hashlib.sha256(GOOD).hexdigest()

    def test_a_too_small_existing_file_is_replaced(self, tmp_path):
        dest = tmp_path / MODEL_FILENAME
        dest.write_bytes(b"broken")
        download_model(dest, opener=opener_returning(FakeResponse(GOOD)))
        assert dest.read_bytes() == GOOD

    def test_force_replaces_only_after_a_complete_download(self, tmp_path):
        dest = tmp_path / MODEL_FILENAME
        dest.write_bytes(GOOD)
        newer = GOOD[::-1]
        download_model(dest, force=True, opener=opener_returning(FakeResponse(newer)))
        assert dest.read_bytes() == newer

    def test_a_failed_forced_download_leaves_the_existing_model_untouched(self, tmp_path):
        dest = tmp_path / MODEL_FILENAME
        dest.write_bytes(GOOD)
        with pytest.raises(ModelDownloadError):
            download_model(dest, force=True,
                           opener=opener_raising(urllib.error.URLError("no route")))
        assert dest.read_bytes() == GOOD and not list(tmp_path.glob("*.part"))


class TestFailures:
    def fails(self, tmp_path, opener, match):
        dest = tmp_path / "models" / MODEL_FILENAME
        with pytest.raises(ModelDownloadError, match=match) as info:
            download_model(dest, opener=opener)
        assert not dest.exists() and not list(dest.parent.glob("*.part"))
        return str(info.value)

    def test_no_network(self, tmp_path):
        message = self.fails(tmp_path, opener_raising(urllib.error.URLError("getaddrinfo failed")),
                             "download failed: getaddrinfo failed")
        assert "internet connection" in message and MODEL_URL in message  # tells what to do

    def test_http_error_status(self, tmp_path):
        err = urllib.error.HTTPError(MODEL_URL, 404, "Not Found", {}, None)
        message = self.fails(tmp_path, opener_raising(err), "HTTP 404")
        assert "Not Found" in message and "browser" in message

    def test_timeout(self, tmp_path):
        self.fails(tmp_path, opener_raising(TimeoutError("timed out")), "download failed")

    def test_connection_dropped_midway_cleans_up(self, tmp_path):
        self.fails(tmp_path, opener_returning(FakeResponse(GOOD, fail_after=200_000)),
                   "download failed")

    def test_truncated_download_is_rejected(self, tmp_path):
        response = FakeResponse(GOOD, content_length=str(len(GOOD) + 5000))
        self.fails(tmp_path, opener_returning(response), "cut short")

    def test_an_error_page_instead_of_a_model_is_rejected(self, tmp_path):
        page = b"<html>Access denied</html>"
        self.fails(tmp_path, opener_returning(FakeResponse(page)), "too small")

    def test_cannot_create_the_target_folder(self, tmp_path):
        blocker = tmp_path / "models"
        blocker.write_text("a file where the folder should be")
        with pytest.raises(ModelDownloadError, match="cannot create"):
            download_model(blocker / MODEL_FILENAME, opener=opener_returning(FakeResponse(GOOD)))

    def test_disk_write_failure_cleans_up(self, tmp_path, monkeypatch):
        dest = tmp_path / MODEL_FILENAME
        real_open = Path.open

        def failing_open(self, mode="r", *a, **k):
            if self.suffix == ".part" and "w" in mode:
                raise OSError("disk full")
            return real_open(self, mode, *a, **k)

        monkeypatch.setattr(Path, "open", failing_open)
        with pytest.raises(ModelDownloadError, match="disk full"):
            download_model(dest, opener=opener_returning(FakeResponse(GOOD)))
        assert not dest.exists() and not list(tmp_path.glob("*.part"))


class TestCommand:
    def test_success_prints_the_hash_and_exits_zero(self, tmp_path, monkeypatch, capsys):
        import nocursor.model_setup as ms

        dest = tmp_path / MODEL_FILENAME
        monkeypatch.setattr(ms, "default_model_path", lambda: dest)
        monkeypatch.setattr(ms.urllib.request, "urlopen",
                            lambda url, timeout: FakeResponse(GOOD))
        assert run_download() == 0
        out = capsys.readouterr().out
        assert hashlib.sha256(GOOD).hexdigest() in out and "not enforced" in out
        assert dest.is_file()

    def test_already_present_says_so(self, tmp_path, monkeypatch, capsys):
        import nocursor.model_setup as ms

        dest = tmp_path / MODEL_FILENAME
        dest.write_bytes(GOOD)
        monkeypatch.setattr(ms, "default_model_path", lambda: dest)
        assert run_download() == 0
        assert "already present" in capsys.readouterr().out

    def test_failure_exits_nonzero_with_the_reason_on_stderr(self, tmp_path, monkeypatch, capsys):
        import nocursor.model_setup as ms

        monkeypatch.setattr(ms, "default_model_path", lambda: tmp_path / MODEL_FILENAME)

        def boom(url, timeout):
            raise urllib.error.URLError("network is unreachable")

        monkeypatch.setattr(ms.urllib.request, "urlopen", boom)
        assert run_download() == 1
        assert "network is unreachable" in capsys.readouterr().err

    def test_the_original_script_still_works_as_a_thin_wrapper(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "download_model.py"
        namespace = runpy.run_path(str(script), run_name="not_main")
        assert namespace["run_download"] is run_download
