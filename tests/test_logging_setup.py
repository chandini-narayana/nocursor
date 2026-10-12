"""Logging: levels, rotation, redaction, native-output handling. No camera, no desktop."""

import io
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from nocursor.logging_setup import (
    DIAGNOSTICS_LOGGER,
    LOG_BACKUPS,
    LOG_BYTES,
    MAX_MESSAGE_CHARS,
    SafeFormatter,
    capture_native_stderr,
    configure_logging,
    report_native_output,
    summarize_argument,
)

HOME = r"C:\Users\someone" if os.name == "nt" else "/home/someone"


@pytest.fixture
def stream():
    return io.StringIO()


@pytest.fixture(autouse=True)
def clean_handlers():
    yield
    logging.getLogger("nocursor").handlers.clear()
    for h in list(logging.getLogger().handlers):
        if getattr(h, "_nocursor_handler", False):
            logging.getLogger().removeHandler(h)


def logger(name="nocursor.test"):
    return logging.getLogger(name)


class TestConsoleLevels:
    def test_info_and_above_by_default(self, stream):
        configure_logging(stream=stream)
        logger().debug("hidden detail")
        logger().info("visible info")
        logger().warning("visible warning")
        out = stream.getvalue()
        assert "hidden detail" not in out and "visible info" in out and "visible warning" in out

    def test_verbose_adds_debug_and_logger_names(self, stream):
        configure_logging(verbose=True, stream=stream)
        logger("nocursor.camera").debug("detail")
        assert "detail" in stream.getvalue() and "nocursor.camera" in stream.getvalue()

    def test_default_format_is_short(self, stream):
        configure_logging(stream=stream)
        logger("nocursor.camera").info("camera open")
        line = stream.getvalue().strip()
        assert line.endswith("INFO camera open") and "nocursor.camera" not in line

    def test_diagnostics_are_debug_records_hidden_unless_asked_for(self, stream):
        configure_logging(stream=stream)
        logging.getLogger(DIAGNOSTICS_LOGGER).debug("diag ratio=0.1")
        assert "diag" not in stream.getvalue()

    def test_diagnostics_flag_shows_them_without_showing_other_debug(self, stream):
        configure_logging(diagnostics=True, stream=stream)
        logging.getLogger(DIAGNOSTICS_LOGGER).debug("diag ratio=0.1")
        logger().debug("unrelated debug")
        out = stream.getvalue()
        assert "diag ratio=0.1" in out and "unrelated debug" not in out

    def test_verbose_also_shows_diagnostics(self, stream):
        configure_logging(verbose=True, stream=stream)
        logging.getLogger(DIAGNOSTICS_LOGGER).debug("diag ratio=0.1")
        assert "diag ratio=0.1" in stream.getvalue()

    def test_file_only_records_skip_the_console_but_reach_the_file(self, stream, tmp_path):
        path = tmp_path / "n.log"
        h = configure_logging(stream=stream, log_file=path)
        logger().info("startup summary", extra={"file_only": True})
        logger().info("for everyone")
        for handler in logging.getLogger("nocursor").handlers:
            handler.flush()
        assert "startup summary" not in stream.getvalue()
        assert "for everyone" in stream.getvalue()
        text = path.read_text(encoding="utf-8")
        assert "startup summary" in text and "for everyone" in text
        h.close()

    def test_calling_configure_twice_does_not_duplicate_output(self, stream):
        configure_logging(stream=stream)
        configure_logging(stream=stream)
        logger().info("once")
        assert stream.getvalue().count("once") == 1

    def test_third_party_warnings_are_not_hidden(self, stream):
        configure_logging(stream=stream)
        logging.getLogger("absl").warning("native library problem")
        logging.getLogger("absl").info("native library chatter")
        out = stream.getvalue()
        assert "native library problem" in out and "native library chatter" not in out


class TestLogFile:
    def test_not_created_unless_requested(self, tmp_path, stream):
        configure_logging(stream=stream)
        logger().info("hello")
        assert list(tmp_path.iterdir()) == []

    def test_file_receives_records_with_names_and_full_timestamps(self, tmp_path, stream):
        path = tmp_path / "logs" / "nocursor.log"
        handle = configure_logging(stream=stream, log_file=path)
        logger("nocursor.app").info("started")
        for h in logging.getLogger("nocursor").handlers:
            h.flush()
        line = path.read_text(encoding="utf-8").strip()
        assert "nocursor.app: started" in line and line[:4].isdigit() and "-" in line[:10]
        assert handle.log_file == path and handle.log_file_error is None
        handle.close()

    def test_file_is_rotated_and_bounded(self, tmp_path, stream):
        path = tmp_path / "r.log"
        handle = configure_logging(stream=stream, log_file=path)
        file_handler = next(
            h for h in logging.getLogger("nocursor").handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        )
        assert file_handler.maxBytes == LOG_BYTES and file_handler.backupCount == LOG_BACKUPS
        file_handler.maxBytes = 2000  # shrink so the test is quick
        for i in range(400):
            logger().info("line %d %s", i, "x" * 40)
        file_handler.flush()
        files = sorted(p.name for p in tmp_path.iterdir())
        assert len(files) <= LOG_BACKUPS + 1 and "r.log" in files
        assert all(p.stat().st_size < 4000 for p in tmp_path.iterdir())
        handle.close()

    def test_unwritable_location_degrades_to_a_warning(self, tmp_path, stream):
        blocker = tmp_path / "blocked"
        blocker.write_text("a file, so a directory cannot be created below it")
        handle = configure_logging(stream=stream, log_file=blocker / "sub" / "x.log")
        assert handle.log_file is None and "cannot write log file" in handle.log_file_error
        assert "continuing without a log file" in stream.getvalue()

    def test_close_removes_our_handlers(self, stream):
        handle = configure_logging(stream=stream)
        handle.close()
        logger().info("after close")
        assert stream.getvalue() == ""


class TestRedaction:
    def fmt(self, record_msg, *args, home=HOME):
        record = logging.LogRecord("nocursor.x", logging.INFO, __file__, 1, record_msg, args, None)
        return SafeFormatter("%(message)s", home=home).format(record)

    def test_home_directory_is_replaced(self):
        out = self.fmt("loading %s", HOME + os.sep + "Projects" + os.sep + "nocursor")
        assert HOME not in out and out.startswith("loading ~")

    def test_both_slash_styles_are_replaced(self):
        forward = HOME.replace("\\", "/") + "/x"
        backward = HOME.replace("/", "\\") + "\\y"
        out = self.fmt("a %s and b %s", forward, backward)
        assert "someone" not in out

    def test_case_differences_on_windows_are_still_redacted(self):
        if os.name != "nt":
            pytest.skip("Windows paths are case-insensitive")
        assert "someone" not in self.fmt("p %s", HOME.upper() + r"\file")

    def test_a_short_or_root_home_is_not_used_for_redaction(self):
        assert self.fmt("path /usr/lib", home="/") == "path /usr/lib"

    def test_numpy_arrays_are_summarised_not_printed(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = self.fmt("frame %s", frame)
        assert "ndarray shape=(480, 640, 3) dtype=uint8" in out and "0 0 0" not in out
        assert len(out) < 100

    def test_landmark_sized_arrays_are_summarised_too(self):
        out = self.fmt("landmarks %s", np.random.default_rng(0).random((21, 3)))
        assert "ndarray shape=(21, 3)" in out and "0." not in out.split("dtype")[0]

    def test_large_byte_strings_and_long_sequences_are_summarised(self):
        assert "bytes len=5000" in self.fmt("blob %s", b"\x00" * 5000)
        assert "list len=500" in self.fmt("history %s", list(range(500)))
        assert self.fmt("small %s", [1, 2, 3]) == "small [1, 2, 3]"

    def test_dict_arguments_are_summarised(self):
        args = ({"a": np.zeros(9)},)
        record = logging.LogRecord("n", logging.INFO, __file__, 1, "%(a)s", args, None)
        assert "ndarray shape=(9,)" in SafeFormatter("%(message)s", home=HOME).format(record)

    def test_very_long_messages_are_truncated(self):
        out = self.fmt("x" * 10_000)
        assert len(out) < MAX_MESSAGE_CHARS + 100 and "more characters truncated" in out

    def test_tracebacks_are_redacted_as_well(self):
        try:
            raise ValueError("bad value in " + HOME + os.sep + "secret.txt")
        except ValueError:
            import sys

            info = sys.exc_info()
            record = logging.LogRecord("n", logging.ERROR, __file__, 1, "failed", (), info)
        out = SafeFormatter("%(message)s", home=HOME).format(record)
        assert "someone" not in out and "ValueError" in out

    def test_formatting_does_not_alter_the_record_other_handlers_see(self):
        record = logging.LogRecord("n", logging.INFO, __file__, 1, "a %s", (np.zeros(3),), None)
        SafeFormatter("%(message)s", home=HOME).format(record)
        assert isinstance(record.args[0], np.ndarray)

    def test_summarize_argument_leaves_ordinary_values_alone(self):
        for value in (1, 2.5, "text", None, (1, 2), b"short"):
            assert summarize_argument(value) == value

    def test_redaction_applies_in_a_real_handler(self, stream):
        configure_logging(stream=stream, home=HOME)
        logger().info("model at %s", HOME + os.sep + "m.task")
        logger().info("frame %s", np.ones((10, 10)))
        out = stream.getvalue()
        assert "someone" not in out and "ndarray" in out


class TestNativeOutput:
    def test_stderr_written_by_native_code_is_captured_not_shown(self, capfd):
        with capture_native_stderr() as cap:
            os.write(2, b"INFO: Created TensorFlow Lite XNNPACK delegate for CPU.\n")
        assert "XNNPACK" in cap.text
        assert "XNNPACK" not in capfd.readouterr().err

    def test_stderr_works_normally_again_afterwards(self, capfd):
        with capture_native_stderr():
            os.write(2, b"inside\n")
        os.write(2, b"outside\n")
        err = capfd.readouterr().err
        assert "outside" in err and "inside" not in err

    def test_stderr_is_restored_even_if_the_block_raises(self, capfd):
        with pytest.raises(RuntimeError), capture_native_stderr():
            os.write(2, b"before failure\n")
            raise RuntimeError("boom")
        os.write(2, b"after\n")
        assert "after" in capfd.readouterr().err

    def test_routine_notices_go_to_debug_only(self, caplog):
        text = ("INFO: Created TensorFlow Lite XNNPACK delegate for CPU.\n"
                "W0000 00:00 inference_feedback_manager.cc:121] Feedback manager requires a model "
                "with a single signature inference. Disabling support for feedback tensors.\n")
        with caplog.at_level(logging.DEBUG, logger="t"):
            report_native_output(text, "MediaPipe", failed=False, logger=logging.getLogger("t"))
        assert {r.levelno for r in caplog.records} == {logging.DEBUG}
        assert len(caplog.records) == 2

    def test_lines_that_look_like_errors_are_surfaced_as_warnings(self, caplog):
        text = "INFO: fine\nE0000 00:00 something.cc:1] Failed to allocate tensors\nlast line\n"
        with caplog.at_level(logging.DEBUG, logger="t"):
            report_native_output(text, "MediaPipe", failed=False, logger=logging.getLogger("t"))
        levels = {r.getMessage(): r.levelno for r in caplog.records}
        assert levels["MediaPipe: E0000 00:00 something.cc:1] Failed to allocate tensors"] == (
            logging.WARNING
        )
        assert levels["MediaPipe: INFO: fine"] == logging.DEBUG

    def test_everything_is_an_error_when_the_operation_failed(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="t"):
            report_native_output("INFO: harmless looking\nreal clue\n", "MediaPipe model load",
                                 failed=True, logger=logging.getLogger("t"))
        assert len(caplog.records) == 1 and caplog.records[0].levelno == logging.ERROR
        assert "harmless looking" in caplog.text and "real clue" in caplog.text

    def test_empty_output_logs_nothing(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="t"):
            report_native_output("\n  \n", "MediaPipe", failed=True, logger=logging.getLogger("t"))
        assert caplog.records == []


class TestRealModelLoadNoise:
    def test_model_load_is_quiet_on_stderr_but_kept_for_verbose(self, capfd, caplog):
        from nocursor.hand_tracker import ModelNotFoundError, resolve_model_path

        try:
            model = resolve_model_path(None)
        except ModelNotFoundError:
            pytest.skip("hand_landmarker.task not downloaded")
        from nocursor.hand_tracker import HandTracker

        capfd.readouterr()
        with caplog.at_level(logging.DEBUG, logger="nocursor.hand_tracker"):
            tracker = HandTracker(model)
        tracker.close()
        err = capfd.readouterr().err
        assert "XNNPACK" not in err and "W0000" not in err  # no raw native noise on the console
        assert "MediaPipe model load" in caplog.text  # but it is still available at DEBUG

    def test_a_corrupt_model_raises_instead_of_being_swallowed(self, tmp_path):
        bad = tmp_path / "bad.task"
        bad.write_bytes(b"this is not a model" * 100)
        from nocursor.hand_tracker import HandTracker

        with pytest.raises(RuntimeError, match="zip archive"):
            HandTracker(bad)

    def test_native_output_from_a_failed_load_is_logged_at_error_and_the_error_propagates(
        self, tmp_path, caplog, monkeypatch, capfd
    ):
        from mediapipe.tasks.python import vision

        def failing_loader(options):
            os.write(2, b"E0000 00:00 model.cc:1] Failed to parse flatbuffer" + os.linesep.encode())
            raise RuntimeError("loader exploded")

        monkeypatch.setattr(vision.HandLandmarker, "create_from_options", failing_loader)
        model = tmp_path / "m.task"
        model.write_bytes(b"x")
        from nocursor.hand_tracker import HandTracker

        capfd.readouterr()
        with (
            caplog.at_level(logging.DEBUG, logger="nocursor.hand_tracker"),
            pytest.raises(RuntimeError, match="loader exploded"),
        ):
            HandTracker(model)
        records = [r for r in caplog.records if "MediaPipe model load" in r.getMessage()]
        assert len(records) == 1 and records[0].levelno == logging.ERROR
        assert "Failed to parse flatbuffer" in records[0].getMessage()
        assert "Failed to parse flatbuffer" not in capfd.readouterr().err  # logged, not raw


def test_default_paths_are_not_logged_with_the_real_home_directory(stream):
    configure_logging(stream=stream)
    logger().info("cwd is %s", Path.home() / "Projects")
    assert str(Path.home()) not in stream.getvalue()
