#!/usr/bin/env python3
"""Write the rewritten AI-Agent answers back into each dialogue folder.

Reads the distilled JSONL produced by ``probe_text_only.py --distilled-out`` and
emits ``align_rewrite.jsonl`` next to each ``aligned_script.jsonl``: the same
utterances in the same order, with the agent's answer text replaced.

The original file is never touched.

Timing: ``start`` never moves, so gaps, overlaps and every other utterance stay
exactly as they were. ``words`` and ``end`` are recomputed with the dataset's own
estimator (syllables / (4 * rate/3) + 0.04s per word; duration = last onset +
average word slot), i.e. the same placeholder timing the originals were built
with — not a measurement of any audio.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


NOMINAL_SPS = 4.0          # rate=3 -> 4 syllables/sec
INTER_WORD_PAUSE = 0.04


def _syllables(word: str) -> int:
    """Rough English syllable count: vowel groups, with a silent-e nudge.

    Copied from slm_dialogue/sample_data.py so rewritten utterances are timed
    exactly the way the dataset's own generator timed the originals.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(1, n)


def retime(text: str, rate: float) -> tuple[list[dict], float]:
    """Word onsets + duration from syllables and rate (an estimate, as before).

    words:    t += syllables/sps + inter_word_pause, per sample_data.build_utt
    duration: last rel_start + average word slot, per schema.__post_init__
    """
    sps = max(NOMINAL_SPS * float(rate) / 3.0, 1e-6)
    words, t = [], 0.0
    for tok in text.split():
        words.append({"w": tok, "rel_start": round(t, 4)})
        t += _syllables(tok) / sps + INTER_WORD_PAUSE
    if not words:
        return [], 0.3
    last = words[-1]["rel_start"]
    avg_slot = (last / max(len(words) - 1, 1)) if len(words) > 1 else 0.3
    return words, last + avg_slot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset root (holds the dialogue folders)")
    ap.add_argument("distilled", help="JSONL from --distilled-out")
    ap.add_argument("--name", default="align_rewrite.jsonl", help="output filename per folder")
    ap.add_argument("--include-fallbacks", action="store_true",
                    help="also write folders where every rewrite was rejected "
                         "(file is then identical to the original)")
    ap.add_argument("--end-mode", default="rule", choices=["rule", "scale", "keep"],
                    help="'rule': end = start + estimator duration; 'scale': keep the file's "
                         "own end but scale the span by the estimator's length ratio; "
                         "'keep': leave end untouched. start never moves in any mode.")
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
                start = float(utt.get("start", 0.0))
                words, duration = retime(new_text, utt.get("rate", 3.0))
                utt["text_original"] = utt.get("text")
                utt["end_original"] = utt.get("end")
                utt["text"] = new_text
                utt["words"] = words
                # start stays put, so gaps and every other utterance are untouched
                if args.end_mode == "rule":
                    utt["end"] = round(start + duration, 4)
                elif args.end_mode == "scale":
                    # The file's own `end` does not match the estimator (see README),
                    # so keep whatever calibration it has and scale it by how much
                    # longer/shorter the rewrite is under the same estimator.
                    _, old_dur = retime(utt["text_original"] or "", utt.get("rate", 3.0))
                    old_end = float(utt.get("end", start))
                    factor = (duration / old_dur) if old_dur > 0 else 1.0
                    utt["end"] = round(start + (old_end - start) * factor, 4)
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
                print(f"    was: {u['text_original']!r}  [{u.get('end_original')}]")
                print(f"    now: {u['text']!r}  [{u.get('end')}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
