#!/usr/bin/env python3
"""Write the rewritten AI-Agent answers back into each dialogue folder.

Reads the distilled JSONL produced by ``probe_text_only.py --distilled-out`` and
emits ``align_rewrite.jsonl`` next to each ``aligned_script.jsonl``: the same
utterances in the same order, with the agent's answer text replaced.

The original file is never touched.

Timing caveat: ``start``/``end`` are kept as they are, so the utterance still
occupies its original slot, but a rewritten sentence rarely has exactly the old
duration. ``words`` is therefore re-spaced evenly across that slot and marked
``words_approx: true`` — real timings have to come from re-synthesising the
agent's audio (only the ``*_D.wav`` track) and re-aligning.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def respace_words(text: str, start: float, end: float) -> list[dict]:
    """Even spacing across the original slot — a placeholder until re-synthesis."""
    words = text.split()
    if not words:
        return []
    dur = max(0.0, float(end) - float(start))
    step = dur / len(words) if len(words) else 0.0
    return [{"w": w, "rel_start": round(i * step, 4)} for i, w in enumerate(words)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset root (holds the dialogue folders)")
    ap.add_argument("distilled", help="JSONL from --distilled-out")
    ap.add_argument("--name", default="align_rewrite.jsonl", help="output filename per folder")
    ap.add_argument("--include-fallbacks", action="store_true",
                    help="also write folders where every rewrite was rejected "
                         "(file is then identical to the original)")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    # dialogue -> {utt_id: new_text}
    replacements: dict[str, dict[str, str]] = defaultdict(dict)
    n_rows, n_fallback = 0, 0
    for line in Path(args.distilled).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        n_rows += 1
        if not row.get("rewrite_ok"):
            n_fallback += 1
            continue  # fall back = keep the original text
        dlg, utt = row.get("dialogue"), row.get("a_utt")
        if not dlg or not utt:
            print(f"warn: row without dialogue/a_utt: {row.get('instruction', '')[:50]}")
            continue
        replacements[dlg][utt] = row["response_rewritten"].strip()

    root = Path(args.root)
    n_written, n_replaced, missing = 0, 0, []
    targets = sorted(replacements) if not args.include_fallbacks else \
        sorted({p.parent.name for p in root.glob("*/aligned_script.jsonl")})

    for dlg in targets:
        src = root / dlg / "aligned_script.jsonl"
        if not src.exists():
            missing.append(dlg)
            continue
        subs = replacements.get(dlg, {})
        out_lines, hit = [], 0
        for line in src.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            utt = json.loads(line)
            new_text = subs.get(utt.get("utt_id"))
            if new_text:
                utt["text_original"] = utt.get("text")
                utt["text"] = new_text
                utt["words"] = respace_words(new_text, utt.get("start", 0.0), utt.get("end", 0.0))
                utt["words_approx"] = True
                utt["rewritten"] = True
                hit += 1
            out_lines.append(json.dumps(utt, ensure_ascii=False))

        if hit != len(subs):
            print(f"warn: {dlg}: matched {hit}/{len(subs)} utt_ids")
        n_replaced += hit
        if not args.dry_run:
            (root / dlg / args.name).write_text("\n".join(out_lines) + "\n")
        n_written += 1

    print(f"distilled rows   : {n_rows} ({n_fallback} fell back to the original)")
    print(f"dialogues written: {n_written}" + ("  [dry-run: nothing written]" if args.dry_run else ""))
    print(f"utterances replaced: {n_replaced}")
    if missing:
        print(f"missing aligned_script.jsonl: {len(missing)} e.g. {missing[:3]}")
    if n_written and not args.dry_run:
        sample = root / targets[0] / args.name
        print(f"\nexample: {sample}")
        for line in sample.read_text().splitlines():
            u = json.loads(line)
            if u.get("rewritten"):
                print(f"  {u['utt_id']} ({u['speaker']})")
                print(f"    was: {u['text_original']!r}")
                print(f"    now: {u['text']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
