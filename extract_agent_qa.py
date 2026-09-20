#!/usr/bin/env python3
"""Pull (question -> AI Agent, AI Agent's answer) pairs out of the synthesized
instruction-finetuning dialogues, ready for the SDFT rewrite step.

Each dialogue directory holds an ``aligned_script.jsonl`` whose utterances carry
``speaker``, ``addressing`` and ``text``. The AI Agent is the last speaker id
("D" in the 4-speaker set, "C" in the 3-speaker one); humans address it either
by name ("AI Agent, how many ...") or silently via ``addressing: ["D"]`` — so
addressing, not the text, is what we match on.

Output: JSONL of {instruction, response, ...} consumable by
``probe_text_only.py --pairs``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

AGENT_NAME_RE = re.compile(r"\bAI\s*Agent\b", re.I)


def load_script(path: Path) -> list[dict]:
    utts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            utts.append(json.loads(line))
    # files are not always in time order (overlaps are interleaved)
    return sorted(utts, key=lambda u: (float(u.get("start", 0.0)), u.get("utt_id", "")))


def detect_agent_id(utts: list[dict], voices_path: Path | None) -> str | None:
    """Prefer an explicit "AI Agent" name in voices.json; else the last speaker id."""
    if voices_path and voices_path.exists():
        try:
            obj = json.loads(voices_path.read_text())
            entries = obj if isinstance(obj, list) else obj.get("speakers", obj)
            if isinstance(entries, dict):
                for sid, val in entries.items():
                    name = val.get("name", "") if isinstance(val, dict) else str(val)
                    if AGENT_NAME_RE.search(str(name)):
                        return sid
            elif isinstance(entries, list):
                for e in entries:
                    if isinstance(e, dict) and AGENT_NAME_RE.search(str(e.get("name", ""))):
                        return e.get("id")
        except (json.JSONDecodeError, AttributeError):
            pass
    speakers = {u.get("speaker") for u in utts if u.get("speaker")}
    return max(speakers) if speakers else None


def extract_pairs(utts: list[dict], agent_id: str, max_gap_sec: float) -> list[dict]:
    """A question is any non-agent utterance addressing the agent; its answer is
    the agent's next utterance that starts after it."""
    pairs = []
    agent_turns = [u for u in utts if u.get("speaker") == agent_id]

    for u in utts:
        if u.get("speaker") == agent_id:
            continue
        addressing = u.get("addressing") or []
        if agent_id not in addressing:
            continue

        q_start = float(u.get("start", 0.0))
        answer = next(
            (a for a in agent_turns if float(a.get("start", 0.0)) >= q_start
             and float(a.get("start", 0.0)) - q_start <= max_gap_sec),
            None,
        )
        if answer is None:
            continue
        pairs.append({
            "instruction": (u.get("text") or "").strip(),
            "response": (answer.get("text") or "").strip(),
            "q_utt": u.get("utt_id"),
            "a_utt": answer.get("utt_id"),
            "q_speaker": u.get("speaker"),
            "agent_id": agent_id,
            "named": bool(AGENT_NAME_RE.search(u.get("text") or "")),
            "q_start": q_start,
            "a_start": float(answer.get("start", 0.0)),
        })
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory holding the dialogue folders")
    ap.add_argument("--out", default="agent_qa_pairs.jsonl")
    ap.add_argument("--max-gap-sec", type=float, default=30.0,
                    help="ignore an agent turn this long after the question (default 30)")
    ap.add_argument("--limit", type=int, default=0, help="only process N dialogues")
    args = ap.parse_args()

    scripts = sorted(Path(args.root).glob("*/aligned_script.jsonl"))
    if args.limit:
        scripts = scripts[: args.limit]
    if not scripts:
        print(f"no */aligned_script.jsonl under {args.root}")
        return 1

    all_pairs, empty, agent_ids = [], [], {}
    for sp in scripts:
        utts = load_script(sp)
        agent_id = detect_agent_id(utts, sp.parent / "voices.json")
        if agent_id is None:
            empty.append(sp.parent.name)
            continue
        agent_ids[agent_id] = agent_ids.get(agent_id, 0) + 1
        pairs = extract_pairs(utts, agent_id, args.max_gap_sec)
        if not pairs:
            empty.append(sp.parent.name)
        for p in pairs:
            p["dialogue"] = sp.parent.name
            all_pairs.append(p)

    with open(args.out, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    named = sum(1 for p in all_pairs if p["named"])
    lens = sorted(len(p["response"].split()) for p in all_pairs)
    print(f"dialogues        : {len(scripts)}")
    print(f"agent ids        : {agent_ids}")
    print(f"pairs            : {len(all_pairs)}")
    print(f"  addressed by name: {named}  ({len(all_pairs) - named} only via `addressing`)")
    if lens:
        print(f"answer length    : min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]} words")
    print(f"dialogues with no pair: {len(empty)}" + (f" e.g. {empty[:3]}" if empty else ""))
    print(f"wrote {args.out}")

    for p in all_pairs[:3]:
        print(f"\n[{p['dialogue']}] {p['q_utt']} {p['q_speaker']}->{p['agent_id']}")
        print(f"  Q: {p['instruction']}")
        print(f"  A: {p['response']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
