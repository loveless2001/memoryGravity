#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_boot_glyphs.py — heuristic scorer for boot-phrase glyph evals.
Usage:
  python eval_boot_glyphs.py --eval glyphs_eval_bootphrases.jsonl --outputs outputs.jsonl
"""
import json, argparse, re
GLYPHS = ["🝞","🜂","🜍","🜏","🜄","⟜"]

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def count_glyphs(text):
    return sum(text.count(g) for g in GLYPHS)

def has_next_step_hint(text):
    t = text.lower()
    keys = ["next", "then", "after", "do ", "take ", "start ", "begin ", "resume ", "continue ", "mark "]
    return any(k in t for k in keys)

def sequence_ok(text, seq):
    idx = -1
    for g in seq:
        j = text.find(g, idx+1)
        if j == -1: return False
        idx = j
    return True

def includes_any(text, glyphs):
    return any(g in text for g in glyphs)

def includes_exactly_one_of(text, glyphs):
    return sum(text.count(g) for g in glyphs) == 1

def run(eval_path, outputs_path):
    eval_items = {e["id"]: e for e in load_jsonl(eval_path)}
    outs = {o["id"]: o["output"] for o in load_jsonl(outputs_path)}
    total = len(eval_items); passed = 0; details = []

    for _id, item in eval_items.items():
        meta = item.get("meta", {})
        out = outs.get(_id, "")
        if not out:
            details.append((_id, item["category"], "NO_OUTPUT")); continue
        ok = True

        mx = meta.get("max_glyphs_total")
        if mx is not None and count_glyphs(out) > mx: ok = False

        for req in meta.get("must_include", []):
            if req not in out: ok = False

        if "expected_any_of" in meta:
            if not includes_any(out, meta["expected_any_of"]) or not includes_exactly_one_of(out, meta["expected_any_of"]):
                ok = False

        if "expected_exact" in meta:
            if not includes_exactly_one_of(out, meta["expected_exact"]): ok = False

        if "exact_count" in meta:
            if count_glyphs(out) != meta["exact_count"]: ok = False

        if "expected_sequence" in meta:
            if not sequence_ok(out, meta["expected_sequence"]): ok = False

        if meta.get("require_next_step_hint") and not has_next_step_hint(out): ok = False

        details.append((_id, item["category"], "OK" if ok else "FAIL"))
        if ok: passed += 1

    print(f"Passed {passed}/{total} = {passed/total:.1%}")
    from collections import Counter
    c = Counter([d[1] for d in details if d[2] != "NO_OUTPUT"])
    print("By category:", dict(c))
    with open("boot_eval_report.json", "w", encoding="utf-8") as f:
        json.dump([{"id": i, "category": cat, "status": s} for (i,cat,s) in details], f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--outputs", required=True)
    args = ap.parse_args()
    run(args.eval, args.outputs)
