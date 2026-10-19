"""Rendering the evaluation: a readable Markdown report and an optional confusion-matrix image.

The report states facts from ``metrics.json`` and never draws conclusions beyond them. Every
report carries the same limitations section, because the data is one person's recordings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nocursor.dynamic.schema import CLASS_NAMES

LIMITATIONS = [
    "One participant, one hand, one camera setup. Results can describe cross-session behaviour "
    "for this person only, not generalisation to other people, hands, cameras or rooms.",
    "One held-out test session. Many takes do not make it many independent trials: takes from one "
    "sitting share lighting, posture and habits. The confidence intervals treat takes as "
    "independent, so the true uncertainty is larger than they show.",
    "One validation session chose the model settings, the early-stopping point and the rejection "
    "threshold, with few negative takes. The threshold in particular is a coarse estimate.",
    "The continuous stream is a single recording of one person's ordinary use, a couple of "
    "minutes long. It is limited evidence about false activations and is not proof of reliable "
    "real-world operation, whatever the count. Overlapping windows are not independent trials.",
    "Takes are deliberate and cued by a countdown, and cut to a fixed window; live use would see "
    "unsegmented, spontaneous movement. Takes in which tracking dropped (mostly hands entering "
    "or leaving view) were excluded from model inputs, so the classifiers never saw them.",
    "Labels come from the performer's intent. Two takes were flagged because their motion looks "
    "weak for the label; they were kept and reported with and without.",
    "Latency figures are in-process software timings on one machine. They are not "
    "camera-to-action latency.",
    "Nothing here is connected to the mouse. This evaluation does not show that the classifier "
    "is safe or useful to run live.",
]


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def with_ci(value: float | None, ci: list[float] | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if ci is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ({ci[0]:.{digits}f} to {ci[1]:.{digits}f})"


def rate_text(count: int, n: int, ci: list[float] | None) -> str:
    if n == 0:
        return "n/a"
    span = "" if ci is None else f" ({pct(ci[0])} to {pct(ci[1])})"
    return f"{count}/{n} = {pct(count / n)}{span}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines + [""]


def _matrix(confusion: dict[str, Any]) -> list[str]:
    columns = confusion["columns_predicted"]
    rows = [[f"**{name}**", *[str(v) for v in row]]
            for name, row in zip(confusion["rows_true"], confusion["matrix"], strict=True)]
    return _table(["true \\ predicted", *columns], rows)


def render_markdown(p: dict[str, Any]) -> str:
    out: list[str] = []
    systems = p["systems"]
    data = p["dataset"]
    banner = "" if p["data_provenance"] == "recorded" else f" **{p['data_provenance']}**"
    out += [f"# Dynamic gesture evaluation: {p['run']}{banner}", ""]
    number = p["evaluation_number_of_this_run"]
    out += [f"Evaluation `{p['eval_id']}`, number **{number}** for this "
            f"run (every scoring of the test split is logged). Split `{p['split_id']}` "
            f"(seed {p['split_seed']}). Data: {p['data_provenance']}.", ""]
    out += ["## Read this first", "",
            "- This is one person's recordings, scored on **one held-out session**. It is a small "
            "pilot, not evidence of reliable real-world operation.",
            "- Nothing was tuned on the test session or on the continuous stream.",
            "- A rejected prediction counts as `no_gesture` in accuracy and F1 (it triggers "
            "nothing); the confusion matrices keep a separate `rejected` column.",
            "- A *false activation* is a `no_gesture` take the system would act on. A *cross "
            "activation* is a true gesture that triggered the wrong gesture.", ""]  # fmt: skip

    a = data["assignments"]
    out += ["## Data and split", ""]
    out += _table(["Role", "Sessions"], [
        ["train", ", ".join(a["train"]) or "-"],
        ["pilot (train only)", ", ".join(a["pilot_train"]) or "-"],
        ["validation", ", ".join(a["validation"]) or "-"],
        ["test (this report)", ", ".join(a["test"]) or "-"],
        ["reserved stream (final false-activation analysis only)",
         ", ".join(a["reserved_stream"]) or "-"],
    ])  # fmt: skip
    out += _table(["Split"] + list(CLASS_NAMES) + ["dates (UTC)"], [
        [name, *[str(data["counts"][name][c]) for c in CLASS_NAMES], ", ".join(data["dates"][name])]
        for name in ("train", "validation", "test")
    ])  # fmt: skip
    out += [f"Takes per class after exclusions. Dataset manifest hash "
            f"`{data['manifest_hash'][:16]}...`.", ""]  # fmt: skip
    excluded = data["excluded_takes"]
    if excluded:
        out += [f"**Excluded from model inputs ({len(excluded)} takes; tracking was not reliable "
                "enough to compute features):**", ""]
        by_reason: dict[tuple[str, str, str], int] = {}
        for e in excluded:
            key = (e["split"], e["label"] + "/" + str(e["subtype"]), e["reason"].split(" (")[0])
            by_reason[key] = by_reason.get(key, 0) + 1
        out += _table(["split", "label/type", "reason", "takes"],
                      [[k[0], k[1], k[2], str(v)] for k, v in sorted(by_reason.items())])
    if data["flagged_takes"]:
        out += ["**Flagged takes** (kept, never relabelled or deleted):", ""]
        out += _table(["take", "split", "label", "note"],
                      [[f["sequence_id"], f["split"], f["label"], f["note"]]
                       for f in data["flagged_takes"]])  # fmt: skip
    if data["warnings"]:
        out += ["Warnings: " + "; ".join(data["warnings"]), ""]
    out += [f"Test takes evaluated: **{p['test_takes_evaluated']}**.", ""]

    out += ["## Systems", ""]
    for name, entry in systems.items():
        out += [f"- **{name}**: {entry['description']['kind']}."]
    out += [""]
    rule_tuned = systems.get("rule_tuned")
    if rule_tuned:
        out += ["Tuned rule parameters: `" + str(rule_tuned["description"]["parameters"]) + "`.",
                ""]  # fmt: skip
    gru = systems.get("gru")
    if gru:
        d = gru["description"]
        out += [f"GRU: hidden size {d['architecture']['hidden_size']}, selected epoch "
                f"{d['training']['selected_epoch']}, seed {d['seed']}, rejection threshold "
                f"**{d['rejection']['threshold']:.2f}** ({d['rejection']['method']}; "
                f"{d['rejection']['validation_negatives']} validation negatives, "
                f"{d['rejection']['validation_false_activations_at_threshold']} still triggering "
                "at that threshold). Weights SHA-256 `" + d["weights_sha256"][:16] + "...`.", ""]

    out += ["## Held-out test results", ""]
    rows = []
    for name, entry in systems.items():
        s = entry["all"]
        fa, cross, rej = s["false_activations"], s["cross_activations"], s["rejection"]
        rows.append([
            f"**{name}**", with_ci(s["macro_f1"]["value"], s["macro_f1"]["ci95"]),
            with_ci(s["accuracy"]["value"], s["accuracy"]["ci95"]),
            rate_text(fa["count"], fa["negatives"], fa["rate_ci95"]),
            f"{cross['count']}/{cross['gestures']}",
            f"{rej['rejected']} of {s['n']}", f"{rej['true_gestures_lost']}/{rej['true_gestures']}",
        ])  # fmt: skip
    out += _table(["system", "macro-F1 (95% CI)", "accuracy (95% CI)",
                   "false activations on negatives", "cross activations", "rejected",
                   "true gestures lost to rejection"], rows)  # fmt: skip
    out += ["Macro-F1 intervals are a bootstrap over takes; proportions use Wilson intervals. See "
            "the limitations: both treat takes as independent.", ""]
    for name, entry in systems.items():
        s = entry["all"]
        out += [f"### {name}", ""]
        out += _table(["class", "precision (95% CI)", "recall (95% CI)", "F1", "support"], [
            [c, with_ci(r["precision"], r["precision_ci"], 2),
             with_ci(r["recall"], r["recall_ci"], 2), f"{r['f1']:.2f}", str(r["support"])]
            for c, r in s["per_class"].items()
        ])  # fmt: skip
        out += ["Confusion matrix (counts; rows are the true class):", ""]
        out += _matrix(s["confusion"])
        sub = s["false_activations"]["by_subtype"]
        if sub:
            out += ["False activations by kind of negative:", ""]
            out += _table(["type", "takes", "false activations"],
                          [[k, str(v["n"]), str(v["false_activations"])]
                           for k, v in sorted(sub.items())])  # fmt: skip

    out += ["## Latency (software timing, this machine)", ""]
    rows = []
    for name, entry in systems.items():
        lat = entry["latency"]
        for part in ("feature_extraction", "inference", "total"):
            value = lat[part]
            if isinstance(value, dict):
                rows.append([name, part, str(value["n"]), f"{value['p50']:.3f}",
                             "n/a" if value["p95"] is None else f"{value['p95']:.3f}"])  # fmt: skip
            else:
                rows.append([name, part, "-", value, value])
    out += _table(["system", "stage", "n", "p50 ms", "p95 ms"], rows)

    out += ["## Continuous stream (reserved for this analysis)", ""]
    stream = p["stream"]
    if not stream:
        out += ["Not run.", ""]
    else:
        first = next(iter(stream.values()))
        out += [f"One recording of {first['stream_seconds']} s of ordinary use. Windows of "
                f"{first['window_seconds']} s every {first['stride_seconds']} s; windows with "
                f"unusable tracking are skipped ({first['windows_evaluated']} of "
                f"{first['windows_total']} evaluated). Flagged windows within "
                f"{first['merge_gap_seconds']} s with the same class are one event.", ""]
        out += _table(["system", "flagged windows", "events", "events per minute (95% CI)",
                       "persistent events*", "per minute (95% CI)"], [
            [f"**{n}**", str(r["flagged_windows"]), str(r["events_any"]["events"]),
             f"{r['events_any']['per_minute']:.2f} ({r['events_any']['ci95_per_minute'][0]:.2f} "
             f"to {r['events_any']['ci95_per_minute'][1]:.2f})",
             str(r["events_persistent"]["events"]),
             f"{r['events_persistent']['per_minute']:.2f} "
             f"({r['events_persistent']['ci95_per_minute'][0]:.2f} to "
             f"{r['events_persistent']['ci95_per_minute'][1]:.2f})"]
            for n, r in stream.items()
        ])  # fmt: skip
        out += [f"\\* {first['persistent_definition']}. Intervals are exact Poisson. "
                f"**{first['caution']}**", ""]

    out += ["## Sensitivity to the flagged takes", ""]
    rows = []
    for name, entry in systems.items():
        with_all, without = entry["all"], entry["without_flagged"]
        rows.append([name, f"{with_all['macro_f1']['value']:.3f}",
                     f"{without['macro_f1']['value']:.3f}",
                     f"{with_all['false_activations']['count']}",
                     f"{without['false_activations']['count']}", str(without["n"])])  # fmt: skip
    out += ["Same trained models, held-out takes scored with and without the flagged takes that "
            "fall in the test session:", ""]
    out += _table(["system", "macro-F1 all", "macro-F1 without flagged",
                   "false activations all", "false activations without", "takes without"], rows)
    flagged_in_test = [t for e in systems.values() for t in e["flagged_takes"]]
    if flagged_in_test:
        out += ["Predictions on the flagged test takes:", ""]
        out += _table(["system", "take", "true", "predicted"],
                      [[n, t["sequence_id"], t["true_label"], t["predicted"]]
                       for n, e in systems.items() for t in e["flagged_takes"]])  # fmt: skip
    else:
        out += ["No flagged take is in the test session.", ""]
    sens = p["sensitivity"]
    if sens:
        out += [f"Retrained without the flagged takes (`{sens['retrained_run']}`), scored on the "
                "same test takes:", ""]
        rows = []
        for name, entry in sens["systems"].items():
            a1, a2 = entry["all"], systems.get(name, {}).get("all")
            rows.append([name, f"{a1['macro_f1']['value']:.3f}",
                         "n/a" if a2 is None else f"{a2['macro_f1']['value']:.3f}",
                         str(a1["false_activations"]["count"]),
                         "n/a" if a2 is None else str(a2["false_activations"]["count"])])
        out += _table(["system", "macro-F1 retrained without flagged", "macro-F1 primary",
                       "false activations retrained", "false activations primary"], rows)
    else:
        out += ["No sensitivity run was supplied (`--sensitivity-run`).", ""]
    out += ["Two flagged takes cannot move a few dozen test takes much; the table is there so "
            "that anyone can check, not to argue a point.", ""]

    if p["seed_stability"]:
        out += ["## Stability across training seeds (same chosen settings)", "",
                "Each seed's model was scored on the test takes for transparency only; none of "
                "this was used to choose the final model.", ""]
        out += _table(["seed", "final model", "threshold", "macro-F1", "accuracy",
                       "false activations", "gestures lost to rejection"],
                      [[str(s["seed"]), "yes" if s["is_final_model"] else "",
                        f"{s['threshold']:.2f}",
                        f"{s['macro_f1']:.3f}", f"{s['accuracy']:.3f}", str(s["false_activations"]),
                        str(s["gestures_lost_to_rejection"])] for s in p["seed_stability"]])

    out += ["## Limitations", ""] + [f"- {text}" for text in p["limitations"]] + [""]
    if p.get("png_note"):
        out += [f"_Confusion-matrix image not written: {p['png_note']}_", ""]
    return "\n".join(out)


def write_confusion_png(path: Path, confusion: dict[str, Any], title: str) -> str | None:
    """Draw the confusion matrix if matplotlib is available; returns a note when it is not."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib is not installed (the text matrices in the report are complete)"
    import numpy as np

    matrix = np.array(confusion["matrix"])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(confusion["columns_predicted"])), confusion["columns_predicted"],
                  rotation=30, ha="right")  # fmt: skip
    ax.set_yticks(range(len(confusion["rows_true"])), confusion["rows_true"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black")  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return None
