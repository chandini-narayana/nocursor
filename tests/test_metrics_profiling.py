"""Latency statistics, stage timing boundaries, and CPU/memory sampling (all deterministic)."""

import pytest

from nocursor.metrics import (
    P95_MIN_SAMPLES,
    P99_MIN_SAMPLES,
    RollingWindow,
    format_ms,
    nearest_rank,
    summarize,
)
from nocursor.profiling import FrameProfiler, NullProfiler
from nocursor.resources import MemoryInfo, ResourceSampler, process_memory

MS = 1_000_000  # nanoseconds per millisecond


class TestPercentiles:
    def test_nearest_rank_on_a_known_set(self):
        values = list(range(1, 101))  # 1..100
        assert [nearest_rank(values, p) for p in (50, 95, 99, 100)] == [50, 95, 99, 100]

    def test_small_sets_use_the_ceiling_rank(self):
        assert nearest_rank([10, 20, 30, 40], 50) == 20  # rank ceil(2.0) = 2
        assert nearest_rank([10, 20, 30, 40], 75) == 30
        assert nearest_rank([7], 99) == 7

    @pytest.mark.parametrize("bad", [0, 101, -5])
    def test_invalid_percent_rejected(self, bad):
        with pytest.raises(ValueError):
            nearest_rank([1, 2, 3], bad)

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            nearest_rank([], 50)

    def test_summary_of_nothing_is_none(self):
        assert summarize([]) is None
        assert RollingWindow(5).summary() is None

    def test_summary_values(self):
        s = summarize([4.0, 1.0, 3.0, 2.0])
        assert (s.n, s.minimum, s.maximum, s.mean, s.p50) == (4, 1.0, 4.0, 2.5, 2.0)

    def test_percentiles_are_withheld_until_there_are_enough_samples(self):
        few = summarize(range(1, P95_MIN_SAMPLES))  # 19 samples
        assert few.p95 is None and few.p99 is None and few.p50 is not None
        enough95 = summarize(range(1, P95_MIN_SAMPLES + 1))  # 20
        assert enough95.p95 == 19 and enough95.p99 is None
        enough99 = summarize(range(1, P99_MIN_SAMPLES + 1))  # 100
        assert enough99.p95 == 95 and enough99.p99 == 99

    def test_unsorted_input_is_handled(self):
        assert summarize([5, 1, 4, 2, 3]).p50 == 3

    def test_as_dict_keeps_none_and_rounds(self):
        d = summarize([1.23456, 2.0]).as_dict(digits=2)
        assert d["p95"] is None and d["p99"] is None and d["mean"] == 1.62 and d["n"] == 2

    def test_format_ms(self):
        assert format_ms(None) == "n/a" and format_ms(12.345) == "12.3"


class TestRollingWindow:
    def test_only_the_latest_samples_count(self):
        w = RollingWindow(3)
        for v in (100.0, 1.0, 2.0, 3.0):
            w.add(v)
        s = w.summary()
        assert len(w) == 3 and s.maximum == 3.0 and s.minimum == 1.0  # the 100 aged out

    def test_one_slow_frame_ages_out(self):
        w = RollingWindow(50)
        w.add(500.0)
        for _ in range(60):
            w.add(10.0)
        assert w.summary().maximum == 10.0

    def test_clear_and_invalid_size(self):
        w = RollingWindow(2)
        w.add(1.0)
        w.clear()
        assert len(w) == 0
        with pytest.raises(ValueError):
            RollingWindow(0)


class FakeClockNs:
    def __init__(self):
        self.t = 0

    def __call__(self):
        return self.t

    def advance_ms(self, ms):
        self.t += int(ms * MS)


def one_frame(p, clock, *, wait=5, infer=10, gesture=2, dispatch=1, preview=7, tag="hand",
              actions=False):
    p.begin_wait()
    clock.advance_ms(wait)
    p.frame_arrived()
    t = p.mark()
    clock.advance_ms(infer)
    p.record_since("inference", t, tag=tag)
    t = p.mark()
    clock.advance_ms(gesture)
    p.record_since("gesture", t)
    t = p.mark()
    clock.advance_ms(dispatch)
    p.record_since("dispatch", t)
    p.end_dispatch(had_actions=actions)
    t = p.mark()
    clock.advance_ms(preview)
    p.record_since("preview", t)


class TestStageBoundaries:
    def test_each_stage_measures_only_its_own_interval(self):
        clock = FakeClockNs()
        p = FrameProfiler(clock_ns=clock)
        one_frame(p, clock)
        for name, expected in [("frame_wait", 5), ("inference", 10), ("gesture", 2),
                               ("dispatch", 1), ("preview", 7)]:
            assert p.window_summary(name).mean == pytest.approx(expected)

    def test_end_to_end_runs_from_frame_arrival_to_end_of_dispatch(self):
        clock = FakeClockNs()
        p = FrameProfiler(clock_ns=clock)
        one_frame(p, clock, wait=5, infer=10, gesture=2, dispatch=1, preview=7)
        # 10 + 2 + 1: excludes the 5 ms camera wait before arrival and the 7 ms preview after.
        assert p.window_summary("end_to_end").mean == pytest.approx(13)

    def test_preview_time_never_enters_the_latency(self):
        clock = FakeClockNs()
        fast, slow = FrameProfiler(clock_ns=clock), FrameProfiler(clock_ns=clock)
        one_frame(fast, clock, preview=1)
        one_frame(slow, clock, preview=500)
        assert fast.window_summary("end_to_end").mean == slow.window_summary("end_to_end").mean

    def test_action_frames_are_tracked_separately(self):
        clock = FakeClockNs()
        p = FrameProfiler(clock_ns=clock)
        one_frame(p, clock, infer=10, actions=False)
        one_frame(p, clock, infer=20, actions=True)
        assert p.window_summary("end_to_end").n == 2
        action = p.window_summary("end_to_end_action")
        assert action.n == 1 and action.mean == pytest.approx(23)

    def test_inference_is_split_by_whether_a_hand_was_found(self):
        clock = FakeClockNs()
        p = FrameProfiler(clock_ns=clock)
        one_frame(p, clock, infer=12, tag="hand")
        one_frame(p, clock, infer=4, tag="no_hand")
        assert p.window_summary("inference_hand").mean == pytest.approx(12)
        assert p.window_summary("inference_no_hand").mean == pytest.approx(4)
        assert p.window_summary("inference").n == 2

    def test_frame_rate_comes_from_arrival_to_arrival_time(self):
        clock = FakeClockNs()
        p = FrameProfiler(clock_ns=clock)
        for _ in range(11):
            p.begin_wait()
            clock.advance_ms(33.333333)
            p.frame_arrived()
        assert p.window_fps() == pytest.approx(30.0, rel=1e-3)
        assert p.session_fps() == pytest.approx(30.0, rel=1e-3)
        assert p.elapsed_s == pytest.approx(0.33333333, rel=1e-3)

    def test_fps_is_unknown_with_too_little_data(self):
        p = FrameProfiler(clock_ns=FakeClockNs())
        assert p.window_fps() is None and p.session_fps() is None and p.elapsed_s == 0.0
        p.frame_arrived()
        assert p.session_fps() is None

    def test_dropped_frames_are_counted(self):
        p = FrameProfiler(clock_ns=FakeClockNs())
        p.frame_arrived()
        p.frame_arrived(got_frame=False)
        assert (p.frames, p.dropped_frames) == (2, 1)

    def test_end_dispatch_before_any_frame_is_ignored(self):
        p = FrameProfiler(clock_ns=FakeClockNs())
        p.end_dispatch(had_actions=True)
        assert p.window_summary("end_to_end") is None

    def test_session_samples_are_kept_only_when_asked_and_are_capped(self):
        clock = FakeClockNs()
        rolling = FrameProfiler(clock_ns=clock, window=5)
        full = FrameProfiler(clock_ns=clock, window=5, keep_session=True, session_cap=8)
        for p in (rolling, full):
            for _ in range(12):
                one_frame(p, clock)
        assert rolling.session_summary("inference") is None
        assert full.window_summary("inference").n == 5  # the window is bounded
        assert full.session_summary("inference").n == 8  # the session cap is honoured

    def test_overlay_line_needs_data_and_labels_the_figures_as_software(self):
        clock = FakeClockNs()
        p = FrameProfiler(clock_ns=clock)
        assert p.overlay_line() is None
        for _ in range(25):
            one_frame(p, clock)
        line = p.overlay_line()
        assert "fps" in line and "software e2e" in line and "p95" in line and "(window 25)" in line

    def test_overlay_line_shows_na_when_p95_is_not_yet_available(self):
        clock = FakeClockNs()
        p = FrameProfiler(clock_ns=clock)
        for _ in range(3):
            one_frame(p, clock)
        assert "p95 n/a" in p.overlay_line()


class TestTrackingStats:
    def test_loss_events_count_transitions_out_of_stable_tracking(self):
        p = FrameProfiler(clock_ns=FakeClockNs())
        for state in ["acquiring", "tracking", "tracking", "lost", "lost", "tracking",
                      "uncertain", "tracking", "tracking"]:
            p.note_tracking(state)
        assert p.tracking.loss_events == 2  # tracking->lost, tracking->uncertain
        assert p.tracking.frames == 9
        assert p.tracking.non_tracking_fraction == pytest.approx(4 / 9)  # 5 of 9 tracking

    def test_acquiring_to_lost_is_not_a_loss_of_tracking(self):
        p = FrameProfiler(clock_ns=FakeClockNs())
        for state in ["acquiring", "lost", "acquiring", "uncertain"]:
            p.note_tracking(state)
        assert p.tracking.loss_events == 0

    def test_loss_rate_per_minute_uses_wall_time(self):
        clock = FakeClockNs()
        p = FrameProfiler(clock_ns=clock)
        p.frame_arrived()
        for state in ["tracking", "lost", "tracking", "lost"]:
            p.note_tracking(state)
        clock.advance_ms(30_000)  # half a minute later
        p.frame_arrived()
        assert p.loss_events_per_minute() == pytest.approx(2 / 0.5)

    def test_loss_rate_is_unknown_without_elapsed_time(self):
        assert FrameProfiler(clock_ns=FakeClockNs()).loss_events_per_minute() is None


class TestNullProfiler:
    def test_every_call_is_a_harmless_no_op(self):
        p = NullProfiler()
        p.begin_wait()
        p.frame_arrived()
        t = p.mark()
        p.record_since("inference", t, tag="hand")
        p.end_dispatch(True)
        p.note_tracking("tracking")
        assert p.enabled is False and t == 0


class FakeCpu:
    def __init__(self):
        self.wall = 0.0
        self.cpu = 0.0

    def advance(self, wall, cpu):
        self.wall += wall
        self.cpu += cpu


def sampler(fake, cpus=8, memory=None, min_interval=1.0):
    return ResourceSampler(
        min_interval_s=min_interval, wall_clock=lambda: fake.wall, cpu_clock=lambda: fake.cpu,
        memory=memory or (lambda: MemoryInfo(100, 200)), logical_cpus=cpus,
    )


class TestResourceSampler:
    def test_cpu_percent_is_a_ratio_of_deltas_not_absolute_time(self):
        fake = FakeCpu()
        fake.advance(1000.0, 500.0)  # a long-running process with lots of past CPU time
        s = sampler(fake)
        s.start()
        fake.advance(10.0, 5.0)  # in the measured window: 5 CPU-seconds in 10 wall-seconds
        out = s.summary()
        assert out.cpu_time_s == pytest.approx(5.0)  # absolute amount, of the window only
        assert out.cpu_percent_of_one_core == pytest.approx(50.0)
        assert out.wall_s == pytest.approx(10.0)

    def test_machine_percent_divides_by_logical_cpus(self):
        fake = FakeCpu()
        s = sampler(fake, cpus=8)
        s.start()
        fake.advance(10.0, 20.0)  # two cores' worth of work: multi-threaded
        out = s.summary()
        assert out.cpu_percent_of_one_core == pytest.approx(200.0)  # can exceed 100
        assert out.cpu_percent_of_machine == pytest.approx(25.0)  # normalised, cannot
        assert out.logical_cpus == 8

    def test_sampling_is_rate_limited(self):
        fake = FakeCpu()
        s = sampler(fake)
        s.start()
        for _ in range(5):
            fake.advance(0.25, 0.125)  # exactly representable, so the 1.0 s boundary is exact
            s.maybe_sample()  # only every >= 1 s actually samples
        assert s.summary().samples == 2  # one regular sample at 1.0 s + the closing one

    def test_peak_uses_the_busiest_full_interval(self):
        fake = FakeCpu()
        s = sampler(fake)
        s.start()
        fake.advance(1.0, 0.2)
        s.maybe_sample()
        fake.advance(1.0, 0.9)
        s.maybe_sample()
        fake.advance(1.0, 0.1)
        s.maybe_sample()
        assert s.summary().cpu_percent_of_one_core_peak == pytest.approx(90.0)

    def test_a_short_closing_interval_cannot_set_the_peak(self):
        fake = FakeCpu()
        s = sampler(fake)
        s.start()
        fake.advance(1.0, 0.1)
        s.maybe_sample()
        fake.advance(0.05, 0.05)  # 100% over 50 ms: noise from the coarse CPU clock
        out = s.summary()
        assert out.cpu_percent_of_one_core_peak == pytest.approx(10.0)

    def test_summary_before_start_is_none(self):
        assert sampler(FakeCpu()).summary() is None

    def test_first_maybe_sample_starts_the_sampler(self):
        fake = FakeCpu()
        s = sampler(fake)
        s.maybe_sample()
        fake.advance(2.0, 1.0)
        assert s.summary().cpu_percent_of_one_core == pytest.approx(50.0)

    def test_zero_wall_time_gives_no_rate(self):
        fake = FakeCpu()
        s = sampler(fake)
        s.start()
        out = s.summary()
        assert out.cpu_percent_of_one_core is None and out.cpu_percent_of_machine is None

    def test_memory_values_are_passed_through_and_unavailable_stays_none(self):
        fake = FakeCpu()
        s = sampler(fake, memory=lambda: MemoryInfo(None, None))
        s.start()
        fake.advance(1.0, 0.1)
        out = s.summary()
        assert out.rss_bytes is None and out.peak_rss_bytes is None
        s2 = sampler(fake)
        s2.start()
        fake.advance(1.0, 0.1)
        assert (s2.summary().rss_bytes, s2.summary().peak_rss_bytes) == (100, 200)

    def test_as_dict_has_documented_keys(self):
        fake = FakeCpu()
        s = sampler(fake)
        s.start()
        fake.advance(1.0, 0.5)
        keys = set(s.summary().as_dict())
        assert {"cpu_time_s", "cpu_percent_of_one_core", "cpu_percent_of_machine",
                "logical_cpus", "rss_bytes", "peak_rss_bytes"} <= keys


def test_real_process_memory_is_sane_or_unavailable():
    info = process_memory()
    for value in (info.rss_bytes, info.peak_rss_bytes):
        assert value is None or value > 1_000_000  # a Python process is more than 1 MB
    if info.rss_bytes is not None and info.peak_rss_bytes is not None:
        assert info.peak_rss_bytes >= info.rss_bytes * 0.5
