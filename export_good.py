#!/usr/bin/env python3
"""Collect the dialogues whose AI-Agent answers were successfully rewritten into
a new folder, audio included.

Run ``apply_rewrites.py`` first: this copies the ``align_rewrite.jsonl`` it
produced, alongside the original script and every audio/metadata file.

Audio is hard-linked by default — same filesystem, no extra disk, but a separate
tree you can move around. Use --mode copy for a real copy, or --mode symlink.

IMPORTANT: the agent track (``*_D.wav``, or ``*_C.wav`` for 3-speaker dialogues)
still holds the ORIGINAL sentence. Rewritten text and agent audio do not match
until that track is re-synthesized; --skip-agent-audio leaves it out entirely so
nothing stale is shipped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

SCRIPT_FILES = ("align_rewrite.jsonl", "aligned_script.jsonl")


def place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # different filesystem — fall back to copying
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset root (the dialogue folders)")
    ap.add_argument("distilled", help="JSONL from --distilled-out")
    ap.add_argument("out", help="destination folder")
    ap.add_argument("--mode", default="hardlink", choices=["hardlink", "copy", "symlink"])
    ap.add_argument("--all-dialogues", action="store_true",
                    help="export every dialogue, not only those with an accepted rewrite")
    ap.add_argument("--skip-agent-audio", action="store_true",
                    help="leave out the agent's track, which still has the original sentence")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    kept: dict[str, list[dict]] = defaultdict(list)
    agents: dict[str, str] = {}
    n_rows = 0
    for line in Path(args.distilled).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n_rows += 1
        if row.get("dialogue"):
            agents[row["dialogue"]] = row.get("agent_id", "D")
            if row.get("rewrite_ok"):
                kept[row["dialogue"]].append(row)

    root, out = Path(args.root), Path(args.out)
    names = sorted({p.parent.name for p in root.glob("*/aligned_script.jsonl")}) \
        if args.all_dialogues else sorted(kept)

    n_dlg, n_files, n_bytes, missing_script, skipped_audio = 0, 0, 0, [], 0
    manifest = []
    total = len(names)
    print(f"exporting {total} dialogues -> {out}", flush=True)
    for i, name in enumerate(names, 1):
        src_dir = root / name
        if not (src_dir / "align_rewrite.jsonl").exists():
            missing_script.append(name)
            continue
        agent_id = agents.get(name, "D")
        dst_dir = out / name

        for f in sorted(src_dir.iterdir()):
            if not f.is_file():
                continue
            if args.skip_agent_audio and f.suffix == ".wav" and f.stem.endswith(f"_{agent_id}"):
                skipped_audio += 1
                continue
            if not args.dry_run:
                place(f, dst_dir / f.name, args.mode)
            n_files += 1
            n_bytes += f.stat().st_size

        manifest.append({"dialogue": name, "agent_id": agent_id,
                         "rewritten": [r["a_utt"] for r in kept.get(name, [])],
                         "n_rewritten": len(kept.get(name, []))})
        n_dlg += 1
        if i % 25 == 0 or i == total:
            print(f"  [{i}/{total}] {n_files} files, {n_bytes / 1e9:.1f} GB  ({name})",
                  flush=True)

    if not args.dry_run and manifest:
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.jsonl").write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest))
        (out / "README.md").write_text(
            "# Rewritten AI-Agent dialogues\n\n"
            f"- dialogues: {n_dlg}\n"
            f"- rewritten utterances: {sum(m['n_rewritten'] for m in manifest)} "
            f"(of {n_rows} agent answers in the source set)\n"
            f"- audio: {args.mode}"
            + (" (agent track excluded)\n" if args.skip_agent_audio else "\n")
            + "\nEach folder keeps `aligned_script.jsonl` (original, untouched) and\n"
              "`align_rewrite.jsonl`, where accepted rewrites replace the agent's `text`\n"
              "and the previous sentence is kept in `text_original`. Timings are unchanged;\n"
              "the reflowed estimate is in `start_reflow` / `end_reflow` / `words_reflow`.\n\n"
            + ("" if args.skip_agent_audio else
               "WARNING: the agent audio track still says the ORIGINAL sentence. Re-synthesize\n"
               "it before using this set for anything that assumes text and audio agree.\n"))

    print(f"agent answers in distilled : {n_rows}")
    print(f"dialogues exported         : {n_dlg}" + ("  [dry-run]" if args.dry_run else ""))
    print(f"rewritten utterances       : {sum(m['n_rewritten'] for m in manifest)}")
    print(f"files                      : {n_files}  ({n_bytes / 1e9:.2f} GB of source data)")
    if args.mode == "hardlink" and not args.dry_run:
        print("                             (hard-linked: no extra disk used)")
    if skipped_audio:
        print(f"agent tracks skipped       : {skipped_audio}")
    if missing_script:
        print(f"no align_rewrite.jsonl     : {len(missing_script)} e.g. {missing_script[:3]}"
              "  — run apply_rewrites.py first")
    if not args.dry_run and manifest:
        print(f"\nwrote {out}/  (+ manifest.jsonl, README.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
