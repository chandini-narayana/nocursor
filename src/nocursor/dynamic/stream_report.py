"""Markdown rendering of the streaming development evaluation."""

from __future__ import annotations

from typing import Any

LIMITATIONS = (
    "Development evidence only. Nothing in this report was chosen, and nothing here may be quoted "
    "as an acceptance result.",
    "Candidates are unreviewed: an activation counts as false because the person was not "
    "performing the target gestures, but nobody has checked each one.",
    "Few minutes of one person's use. Settings found on the same recordings they are scored on "
    "are optimistic; with a single recording no held-out check is possible.",
    "Recall comes from replaying isolated, cued takes with the hand held perfectly still before "
    "and after. Live gestures start and end less cleanly.",
    "The recorder required takes to be an open palm in at least 80% of frames, so a gate at or "
    "below 0.8 costs takes no recall by construction; real recall loss to the gate is "
    "underestimated.",
    "The validation takes also served the GRU's early stopping and rejection threshold.",
    "A window of tracking that is too poor to compute features is skipped, as in live use, and "
    "its time stays in the denominator.",
    "Offline only: nothing is connected to the mouse.",
)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _config_text(row: dict[str, Any]) -> str:
    g, s = row["gate"]["min_palm_fraction"], row["stabilizer"]
    return (f"gate {g:g}, tau {s['tau']:g}, vote {s['k']}/{s['n']}, cooldown {s['cooldown_s']:g}s, "
            f"re-arm {s['rearm_quiet']}, guard {s['opposite_guard_s']:g}s")


def render_eval(report: dict[str, Any]) -> str:
    lines = [
        "# Streaming development evaluation (PRELIMINARY)", "",
        f"Run `{report['run']}`, split `{report['split_id']}`, "
        f"grid hash `{report['grid_hash'][:16]}`.",
        "", "## Read this first", "", *[f"- {text}" for text in LIMITATIONS], "",
    ]  # fmt: skip
    for note in report["notes"]:
        lines.append(f"- Note: {note}.")
    lines += ["", "## Development recordings", "",
              "| recording | minutes | windows | evaluated | conditions |",
              "|---|---|---|---|---|"]
    for r in report["recordings"]:
        lines.append(f"| {r['recording_id']} | {r['minutes']:.2f} | {r['windows']} | "
                     f"{r['evaluated']} | {r['conditions']} |")  # fmt: skip
    lines += ["", f"Replayed takes: {report['replay_takes']}.", "", "## Hashes", ""]
    for name, value in report["hashes"].items():
        lines.append(f"- {name}: `{value}`")
    lines.append("")
    for name, system in report["systems"].items():
        pooled = system["legacy_pooled"]
        lines += [
            f"## {name}", "",
            f"Earlier measure (no gate, no stabilizer): {pooled['flagged_windows']} flagged "
            f"windows, {pooled['events_any']} events, {pooled['events_persistent']} persistent "
            "events.", "",
            "### Fixed ladder (one mechanism added per row, nothing tuned)", "",
            "| stage | candidates | per minute (95% CI; one-sided upper) | pooled recall | "
            "swipe_left | swipe_right | circle | cross | missed by reason |",
            "|---|---|---|---|---|---|---|---|---|",
        ]  # fmt: skip
        for row in system["ladder"]:
            rate = row["rate"]
            cr = row["class_recall"]
            lines.append(
                f"| {row['stage']} | {row['candidates']} | {rate['per_minute']:.2f} "
                f"({rate['ci95_two_sided_per_minute'][0]:.2f} to "
                f"{rate['ci95_two_sided_per_minute'][1]:.2f}; "
                f"<= {rate['upper95_one_sided_per_minute']:.2f}) | "
                f"{_fmt(row['pooled_recall'])} | {_fmt(cr['swipe_left'])} | "
                f"{_fmt(cr['swipe_right'])} | {_fmt(cr['circle_clockwise'])} | "
                f"{row['cross_activations']} | {row['missed_by_reason'] or '-'} |"
            )  # fmt: skip
        best = system["best_in_sample"]
        lines += ["", f"### Trade-off frontier ({system['grid_size']} settings in the declared "
                      "grid; non-dominated rows, in-sample)", "",
                  f"{system['settings_meeting_dev_constraints']} of {system['grid_size']} "
                  "settings keep dev recall (pooled >= 0.90, each gesture >= 0.85, few cross or "
                  "duplicate activations). Fewest candidates among them, **not a selection**: "
                  + ("none." if best is None else f"{best['candidates']} "
                     f"({_fmt(best['candidates_per_minute'], 2)} per minute), recall "
                     f"{_fmt(best['pooled_recall'])}, {_config_text(best)}."), "",
                  "| candidates | per minute | pooled recall | cross | equivalent settings | "
                  "setting |", "|---|---|---|---|---|---|"]  # fmt: skip
        for row in system["pareto_front"][:15]:
            lines.append(f"| {row['candidates']} | {_fmt(row['candidates_per_minute'], 2)} | "
                         f"{_fmt(row['pooled_recall'])} | {row['cross_activations']} | "
                         f"{row['equivalent_settings']} | {_config_text(row)} |")  # fmt: skip
        lines.append("")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------- compare
COMPARE_LIMITATIONS = (
    "Exploratory development screening. Nothing was selected, frozen or accepted, and nothing here "
    "may be quoted as an acceptance result.",
    "One development recording of ordinary use (a couple of minutes). Every candidate count is "
    "tiny, and settings scanned on the same recording they are scored on are optimistic.",
    "Candidates are **unreviewed** unless a person marked them. 'Confirmed false' counts only "
    "marked candidates; an unmarked candidate is never assumed false.",
    "Recall comes from replaying isolated, cued takes. The validation takes also drove the GRU's "
    "(and any candidate's) early stopping and rejection threshold, so recall on them is not "
    "independent generalisation evidence.",
    "The existing takes mostly start moving almost at once (median still time before moving is "
    "about 0.25 s for swipes and 0.1 s for circles), so an arming rule that asks for a deliberate "
    "pause cannot be fairly judged on them: `own_lead_only` shows what they would give, "
    "`rest_padded` assumes a rest that was never recorded.",
    "The settle gate applies to swipes only: recorded circles end in motion (median about 2 hand "
    "sizes per second), so a stillness rule would remove them.",
    "Offline only: nothing is connected to the mouse.",
)


def _cell(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _short(value: str | None) -> str:
    return "none" if value is None else f"`{value[:16]}`"


def _review_source_lines(report: dict[str, Any]) -> list[str]:
    """Where the review marks came from, and how that item list relates to this run's export."""
    source = report.get("review_source")
    lines = ["## Review source", ""]
    if source is None:
        return [*lines, "No review folder was supplied (`--review-dir`), so every candidate "
                        "is unreviewed.", ""]  # fmt: skip
    lines += [
        f"- folder: `{source['folder']}`",
        f"- source `review_items.json` sha256: {_short(source['items_sha256'])}",
        f"- source `review_marks.json` sha256: {_short(source['marks_sha256'])}",
        f"- marked: {source['items_marked']} of {source['items_total']} items",
        f"- this run's own export sha256: {_short(source['current_export_items_sha256'])}",
    ]
    if not source["marks_file_present"]:
        lines += ["", "**Warning: the folder has no `review_marks.json`.** No candidate counts as "
                      "reviewed; every one is unreviewed."]  # fmt: skip
    if source["matches_current_export"]:
        lines += ["", "The source item list is identical to the list this run exports."]
    else:
        lines += [
            "", "**The source item list differs from the list this run exports** (different "
            "sha256). That is allowed: marks are matched to this run's candidates by content "
            "(same recording, same gesture, found by the same variant, and a candidate time within "
            "0.06 s of one recorded in the item), not by file hash or item id. Candidates that "
            "match no item are counted as unreviewed. The export written beside this report has no "
            "marks of its own; marks belong to the source list.", "",
        ]
        for entry in source["differing_items"]:
            for field, values in entry["fields"].items():
                lines.append(f"- item `{entry['item_id']}` differs in `{field}`: source "
                             f"{values['source']} versus current {values['current']}")  # fmt: skip
        if source["items_only_in_source"]:
            lines.append(f"- only in the source list: {', '.join(source['items_only_in_source'])}")
        if source["items_only_in_current"]:
            only = ", ".join(source["items_only_in_current"])
            lines.append(f"- only in this run's export: {only}")
        if not (source["differing_items"] or source["items_only_in_source"]
                or source["items_only_in_current"]):  # fmt: skip
            lines.append("- the items themselves are identical; only the file bytes differ")
    return [*lines, ""]


def render_compare(report: dict[str, Any]) -> str:
    lines = [
        "# Streaming fallback comparison (EXPLORATORY)", "",
        f"Run `{report['run']}`, split `{report['split_id']}`, comparison grid "
        f"`{report['hashes']['compare_grid_hash'][:16]}`.", "",
        "## Read this first", "", *[f"- {text}" for text in COMPARE_LIMITATIONS], "",
    ]  # fmt: skip
    for note in report["notes"]:
        lines.append(f"- Note: {note}.")
    lines += ["", "## Evidence and protocol", "",
              "| recording | minutes | conditions |", "|---|---|---|"]  # fmt: skip
    for r in report["recordings"]:
        lines.append(f"| {r['recording_id']} | {r['minutes']:.2f} | {r['conditions']} |")
    lines += ["", f"Replayed takes by class: {report['replay_takes']}.", ""]
    for name, meaning in report["protocols"].items():
        lines.append(f"- **{name}**: {meaning}.")
    lines += ["", "## Hashes and provenance", ""]
    for name, value in report["hashes"].items():
        if name == "candidates":
            for c in value:
                lines.append(f"- candidate `{c['name']}`: weights `{c['weights_sha256']}`, "
                             f"model.json `{c['model_json_sha256']}`, rejection threshold "
                             f"{c['threshold']}")  # fmt: skip
                for d in c["disclosures"] or []:
                    lines.append(f"  - disclosure: {d}")
        else:
            lines.append(f"- {name}: `{value}`")
    lines += ["", f"Blind review items for these candidates: {report['review_export']['items']} "
                  f"(`{report['review_export']['folder']}/`, sha256 "
                  f"`{report['review_export']['items_sha256'][:16]}`).", ""]  # fmt: skip
    lines += _review_source_lines(report)

    lines += [
        "## Reference settings (declared in advance, nothing tuned)", "",
        "Candidates are activations on the development recording. Review columns count only what a "
        "person marked; everything else is *unreviewed*.", "",
        "| system | mechanism | candidates (per min; one-sided 95% upper) | confirmed false | "
        "genuine | ambiguous | unreviewed | recall (rest-padded) | swipe L / R / circle | cross | "
        "latency p50 / p95 s | recall (own lead only) | circle | cross |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for system, blocks in report["systems"].items():
        for mechanism, block in blocks.items():
            ref, rate, rv = block["reference"], block["reference_rate"], block["review"]
            a, b = ref["protocols"]["rest_padded"], ref["protocols"]["own_lead_only"]
            cr = a["class_recall"]
            lat = a["latency"]
            lines.append(
                f"| {system} | {mechanism} | {ref['candidates']} "
                f"({rate['per_minute']:.1f}; <= {rate['upper95_one_sided_per_minute']:.1f}) | "
                f"{rv['false']} | {rv['genuine']} | {rv['ambiguous']} | {rv['unreviewed']} | "
                f"{_cell(a['pooled_recall'])} | {_cell(cr['swipe_left'])} / "
                f"{_cell(cr['swipe_right'])} / {_cell(cr['circle_clockwise'])} | "
                f"{a['cross_activations']} | {_cell(lat['p50_s'])} / {_cell(lat['p95_s'])} | "
                f"{_cell(b['pooled_recall'])} | {_cell(b['class_recall']['circle_clockwise'])} | "
                f"{b['cross_activations']} |"
            )  # fmt: skip

    unmatched = [(f"{system} / {mechanism}", block["review"]["not_in_review_set"])
                 for system, blocks in report["systems"].items()
                 for mechanism, block in blocks.items() if block["review"]["not_in_review_set"]]
    if unmatched:
        lines += ["", "Candidates that match **no** item of the supplied review folder (counted as "
                      "unreviewed, never as false): "
                  + ", ".join(f"{name}: {count}" for name, count in unmatched) + "."]  # fmt: skip

    lines += ["", "### Why recall was lost (reference settings, rest-padded)", "",
              "| system | mechanism | missed by reason | cross diagnosis |",
              "|---|---|---|---|"]
    for system, blocks in report["systems"].items():
        for mechanism, block in blocks.items():
            a = block["reference"]["protocols"]["rest_padded"]
            lines.append(f"| {system} | {mechanism} | {a['missed_by_reason'] or '-'} | "
                         f"{a['cross_diagnosis']['pairs'] or '-'} |")  # fmt: skip

    lines += ["", "## Exploratory screening over the declared grid (in-sample)", "",
              "Screening: pooled recall >= 0.90, every gesture >= 0.85, at most 2 cross "
              "activations per 45 target takes (raw counts shown; this is not the final "
              "criterion of 2 per 90), few duplicates. 'Best' is the fewest candidates among "
              "settings that pass, **not a selection**.", "",
              "| system | mechanism | protocol | settings passing | best: candidates | recall | "
              "circle | cross | setting |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for system, blocks in report["systems"].items():
        for mechanism, block in blocks.items():
            for protocol, scan in block["screening"].items():
                best = scan["best_in_sample"]
                if best is None:
                    lines.append(f"| {system} | {mechanism} | {protocol} | "
                                 f"{scan['settings_meeting_screening']} of {block['grid_settings']}"
                                 " | none | | | | |")  # fmt: skip
                    continue
                g, st = best["gate"]["min_palm_fraction"], best["stabilizer"]
                extras = best.get("extras") or {}
                settle = extras.get("settle_speed")
                arming = extras.get("arming")
                text = f"gate {g:g}, tau {st['tau']:g}, vote {st['k']}/{st['n']}, re-arm " \
                       f"{st['rearm_quiet']}"
                text += f", settle {settle:g}" if settle else ""
                text += f", arming hold {arming['hold_s']:g}s" if arming else ""
                lines.append(
                    f"| {system} | {mechanism} | {protocol} | "
                    f"{scan['settings_meeting_screening']} of {block['grid_settings']} | "
                    f"{best['candidates']} | {_cell(best['pooled_recall'])} | "
                    f"{_cell(best['class_recall']['circle_clockwise'])} | "
                    f"{best['cross_activations']} | {text} |"
                )

    lines += ["", "## How each window is read, by where it sits on the gesture", "",
              "Window start minus the take's start (negative: the window starts before the "
              "gesture; about 0: it holds the whole gesture; positive: the beginning is cut "
              "off). Counts of windows by what the model said; `rejected` is the model "
              "declining.", ""]
    for system, profiles in report["shift_profiles"].items():
        profile = profiles["circle_clockwise"]
        labels = sorted({label for row in profile.values() for label in row})
        lines += [f"### {system}: circle_clockwise takes", "",
                  "| window start vs take start (s) | " + " | ".join(labels) + " |",
                  "|---|" + "---|" * len(labels)]  # fmt: skip
        for bin_name, row in profile.items():
            lines.append(f"| {bin_name} | " + " | ".join(str(row.get(c, 0)) for c in labels) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"
