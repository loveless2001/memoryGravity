"""
Analyze whether Pythia-1B curvature sign reversal is explained by simple
surface-position features in the saved token-stratification rows.

Inputs:
  results/modal_pythia_token_stratification/*_rows.jsonl

Outputs:
  results/modal_pythia_surface_stratification/surface_stratification_summary.csv
  plans/reports/spike-260509-0110-pythia-surface-stratification.md
"""

from __future__ import annotations

import csv
import json
import math
import string
from collections import defaultdict
from pathlib import Path
from statistics import mean


REPO = Path(__file__).resolve().parent.parent
IN_DIR = REPO / "results" / "modal_pythia_token_stratification"
OUT_DIR = REPO / "results" / "modal_pythia_surface_stratification"
REPORT_PATH = REPO / "plans" / "reports" / "spike-260509-0110-pythia-surface-stratification.md"

SENT_PUNCT = set(".!?")
ANY_PUNCT = set(string.punctuation)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def safe_float(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.6f}"


def pos_bin(token_index: int) -> str:
    if token_index < 20:
        return "tok_006_019"
    if token_index < 50:
        return "tok_020_049"
    if token_index < 100:
        return "tok_050_099"
    return "tok_100_plus"


def punct_kind(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "blank"
    has_sent = any(ch in SENT_PUNCT for ch in stripped)
    has_any = any(ch in ANY_PUNCT for ch in stripped)
    only_punct = all((ch in ANY_PUNCT) for ch in stripped)
    if only_punct and has_sent:
        return "sentence_punct_only"
    if only_punct:
        return "other_punct_only"
    if has_sent:
        return "contains_sentence_punct"
    if has_any:
        return "contains_other_punct"
    return "no_punct"


def sentence_zone(sent_offset: int | None, current_has_sentence_punct: bool) -> str:
    if current_has_sentence_punct:
        return "sentence_punct_token"
    if sent_offset is None:
        return "unknown_before_first_boundary"
    if sent_offset <= 3:
        return "after_sentence_1_3"
    if sent_offset <= 12:
        return "after_sentence_4_12"
    return "after_sentence_13_plus"


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(IN_DIR.glob("*_rows.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["_source"] = path.name
                    rows.append(row)
    return rows


def annotate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row["revision"]), int(row["layer"]), int(row["passage_id"]))
        grouped[key].append(row)

    annotated: list[dict] = []
    for subset in grouped.values():
        subset.sort(key=lambda r: int(r["token_index"]))
        offset: int | None = None
        prev_kind = "none"
        for row in subset:
            text = str(row.get("token_text", ""))
            pk = punct_kind(text)
            has_sent = "sentence" in pk
            zone = sentence_zone(offset, has_sent)
            out = dict(row)
            out["punct_kind"] = pk
            out["position_bin"] = pos_bin(int(row["token_index"]))
            out["prev_punct_kind"] = prev_kind
            out["sentence_zone"] = zone
            out["surface_combo"] = "|".join([
                str(row["token_class"]),
                pk,
                zone,
                out["position_bin"],
            ])
            annotated.append(out)

            if has_sent:
                offset = 0
            elif offset is not None:
                offset += 1
            prev_kind = pk
    return annotated


def summarize_group(rows: list[dict], family: str, group: str) -> dict:
    curv = [float(r["contextual_curvature"]) for r in rows]
    ent = [float(r["entropy"]) for r in rows]
    speed = [float(r["contextual_speed"]) for r in rows]
    return {
        "revision": rows[0]["revision"],
        "layer": int(rows[0]["layer"]),
        "group_family": family,
        "group": group,
        "n": len(rows),
        "mean_curvature": mean(curv),
        "mean_entropy": mean(ent),
        "mean_speed": mean(speed),
        "curvature_entropy_pearson": pearson(curv, ent),
    }


def residualized_pearson(rows: list[dict], key: str) -> tuple[float, int, int]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)

    x_res: list[float] = []
    y_res: list[float] = []
    used_groups = 0
    for subset in groups.values():
        if len(subset) < 3:
            continue
        used_groups += 1
        mx = mean(float(r["contextual_curvature"]) for r in subset)
        my = mean(float(r["entropy"]) for r in subset)
        for row in subset:
            x_res.append(float(row["contextual_curvature"]) - mx)
            y_res.append(float(row["entropy"]) - my)
    return pearson(x_res, y_res), len(x_res), used_groups


def main() -> int:
    rows = annotate(load_rows())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    by_rev_layer: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_rev_layer[(str(row["revision"]), int(row["layer"]))].append(row)

    families = [
        ("all", None),
        ("token_class", "token_class"),
        ("punct_kind", "punct_kind"),
        ("sentence_zone", "sentence_zone"),
        ("position_bin", "position_bin"),
        ("surface_combo", "surface_combo"),
    ]

    summary_rows: list[dict] = []
    residual_rows: list[dict] = []
    for key in sorted(by_rev_layer, key=lambda x: (int(x[0].replace("step", "")), x[1])):
        subset = by_rev_layer[key]
        summary_rows.append(summarize_group(subset, "all", "all"))
        for family, row_key in families[1:]:
            parts: dict[str, list[dict]] = defaultdict(list)
            for row in subset:
                parts[str(row[row_key])].append(row)
            for group, part in sorted(parts.items()):
                summary_rows.append(summarize_group(part, family, group))
        for row_key in ["token_class", "punct_kind", "sentence_zone", "position_bin", "surface_combo"]:
            r, n, k = residualized_pearson(subset, row_key)
            residual_rows.append({
                "revision": key[0],
                "layer": key[1],
                "control": row_key,
                "n": n,
                "n_groups": k,
                "residualized_curvature_entropy_pearson": r,
            })

    csv_path = OUT_DIR / "surface_stratification_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "revision", "layer", "group_family", "group", "n",
            "mean_curvature", "mean_entropy", "mean_speed",
            "curvature_entropy_pearson",
        ])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    resid_path = OUT_DIR / "surface_residualized_correlations.csv"
    with resid_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "revision", "layer", "control", "n", "n_groups",
            "residualized_curvature_entropy_pearson",
        ])
        writer.writeheader()
        for row in residual_rows:
            writer.writerow(row)

    selected = [
        ("step128", 15),
        ("step512", 1),
        ("step512", 5),
        ("step2000", 5),
        ("step8000", 5),
        ("step143000", 4),
    ]

    all_lookup = {
        (r["revision"], int(r["layer"]), r["group_family"], r["group"]): r
        for r in summary_rows
    }
    resid_lookup = {
        (r["revision"], int(r["layer"]), r["control"]): r
        for r in residual_rows
    }

    lines: list[str] = []
    lines.append("# Pythia Surface Stratification Spike")
    lines.append("")
    lines.append("Question: can the Pythia-1B curvature/entropy sign reversal be")
    lines.append("explained by simple surface-position features rather than learned")
    lines.append("context-integration geometry?")
    lines.append("")
    lines.append("Input rows: `results/modal_pythia_token_stratification/*_rows.jsonl`.")
    lines.append("This reuses saved rows only; no new Modal inference was run.")
    lines.append("")
    lines.append("## Selected Correlations")
    lines.append("")
    lines.append("| Revision | Layer | All r | Residual r after token_class | Residual r after surface_combo |")
    lines.append("|---|---:|---:|---:|---:|")
    for revision, layer in selected:
        all_row = all_lookup[(revision, layer, "all", "all")]
        token_resid = resid_lookup[(revision, layer, "token_class")]
        combo_resid = resid_lookup[(revision, layer, "surface_combo")]
        lines.append(
            f"| {revision} | {layer} | "
            f"{safe_float(all_row['curvature_entropy_pearson'])} | "
            f"{safe_float(token_resid['residualized_curvature_entropy_pearson'])} | "
            f"{safe_float(combo_resid['residualized_curvature_entropy_pearson'])} |"
        )
    lines.append("")
    lines.append("`surface_combo` combines token class, punctuation kind, sentence-zone,")
    lines.append("and absolute token-position bin, then correlates residuals after")
    lines.append("subtracting each combo group's mean curvature and entropy.")
    lines.append("")
    lines.append("## Sentence-Zone Checks")
    lines.append("")
    lines.append("| Revision | Layer | Zone | n | r | mean entropy | mean curvature |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    zone_order = [
        "unknown_before_first_boundary",
        "after_sentence_1_3",
        "after_sentence_4_12",
        "after_sentence_13_plus",
        "sentence_punct_token",
    ]
    for revision, layer in selected:
        for zone in zone_order:
            row = all_lookup.get((revision, layer, "sentence_zone", zone))
            if not row:
                continue
            lines.append(
                f"| {revision} | {layer} | {zone} | {row['n']} | "
                f"{safe_float(row['curvature_entropy_pearson'])} | "
                f"{row['mean_entropy']:.3f} | {row['mean_curvature']:.4f} |"
            )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("- Simple surface-position controls do not resolve Q2.")
    lines.append("- The early negative correlations survive token-class demeaning and")
    lines.append("  broad surface-combo demeaning at similar magnitude.")
    lines.append("- Sentence punctuation and immediately-after-sentence tokens are small")
    lines.append("  groups in this LAMBADA subset, so they are not large enough to explain")
    lines.append("  the global sign reversal.")
    lines.append("- This weakens the punctuation/sentence-position version of the surface")
    lines.append("  prior hypothesis, but does not test attention-vs-MLP residual sources.")
    lines.append("")
    lines.append("Next discriminant if reopened: re-extract Pythia-1B rows with separate")
    lines.append("attention-output and MLP-output residual deltas at the same checkpoints.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{csv_path.relative_to(REPO)}`")
    lines.append(f"- `{resid_path.relative_to(REPO)}`")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[surface] wrote {csv_path}")
    print(f"[surface] wrote {resid_path}")
    print(f"[surface] wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
