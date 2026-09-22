#!/usr/bin/env python3
"""Re-run the quality checks over an existing distilled JSONL.

The rewrites themselves are unchanged — only the accept/reject decision is
recomputed, so tightening or relaxing a check costs seconds instead of another
GPU pass over 3000 prompts.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from probe_text_only import REFUSAL_RE, check_rewrite  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("distilled")
    ap.add_argument("--out", default="", help="write the updated JSONL here (default: in place)")
    ap.add_argument("--max-len-ratio", type=float, default=1.6)
    ap.add_argument("--min-len-ratio", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(ln) for ln in Path(args.distilled).read_text().splitlines() if ln.strip()]
    cats: collections.Counter = collections.Counter()
    changed = 0
    for r in rows:
        was = r.get("rewrite_ok")
        if r.get("stopped_on") == "skipped_refusal" or REFUSAL_RE.search(r.get("response_original") or ""):
            ok, why = False, "skipped: refusal"
        else:
            ok, why = check_rewrite((r.get("response_rewritten") or "").strip(),
                                    r.get("response_original"), args)
        r["rewrite_ok"] = ok
        r["rewrite_reject_reason"] = why
        r["response"] = r["response_rewritten"] if ok else r["response_original"]
        cats[re.sub(r" .*", "", why) or "kept"] += 1
        changed += (was != ok)

    kept = cats.get("kept", 0)
    print(f"kept {kept}/{len(rows)} ({100 * kept // max(1, len(rows))}%)   [{changed} decisions changed]")
    for k, n in cats.most_common():
        print(f"  {k:12} {n}")

    if not args.dry_run:
        out = Path(args.out or args.distilled)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
