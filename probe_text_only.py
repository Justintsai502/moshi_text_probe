#!/usr/bin/env python3
"""Probe whether PersonaPlex / Moshi still works as a plain text LM.

Idea: the Temporal Transformer was initialised from Helium (a text-only LM).
Feeding ``zero_token_id`` (-1) on all 16 audio streams makes their embeddings
exactly zero (see ScaledEmbedding.forward in moshi/models/lm_utils.py), so the
model reduces to  text_emb -> Temporal Transformer -> text_linear.

This script drives that path directly instead of going through ``LMGen``:
LMGen expects ``num_codebooks - dep_q - 1`` user audio streams per step (0 for
PersonaPlex, since dep_q=16) and gives no way to feed text in, which is exactly
what we need here.

Run --dry-run first (no torch, no weights) to see the frames that would be fed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_QUESTIONS = [
    "What is one plus one?",
    "The capital of France is",
]


# --------------------------------------------------------------------------
# prompt styles
# --------------------------------------------------------------------------
#: Few-shot header. The model is a *base* LM in text-only mode — it was never
#: aligned on this regime, so it has no notion of "the answer is finished": in
#: speech mode a turn ends with PAD/silence, not EOS. The fix is to demonstrate
#: the format, terminator included, and then stop on that terminator.
FEWSHOT_TEMPLATE = """Answer each question briefly, then write {end} on its own line.

Question: What is the capital of Japan?
Answer: Tokyo.
{end}

Question: How many legs does a spider have?
Answer: Eight.
{end}

Question: {q}
Answer:"""


#: The rewrite task itself (arXiv:2402.13669, Figure 3), wrapped in one worked
#: example so an unaligned base LM copies the format — including the terminator.
SDFT_TEMPLATE = """Below are an instruction that describes a task along with a reference answer. Using the reference answer as a guide, write your own response. Finish your response with {end}.

### Instruction:
Name two planets in the Solar System.
### Reference Answer:
Two planets in the Solar System are Mars and Venus.
### Response:
Two of the planets in our Solar System are Mars and Venus.
{end}

### Instruction:
{instruction}
### Reference Answer:
{reference}
### Response:
"""


#: Same idea, but for the AI-Agent dialogues: the persona answers a question put
#: to it with ONE short factual sentence, so the worked example must demonstrate
#: a rewrite that stays that short. The generic template drifts chatty.
SDFT_AGENT_TEMPLATE = """You are an AI assistant in a room with several people. When someone asks you a question, you answer with one short factual sentence. Below is a question and a reference answer. Rewrite the answer in your own words, keeping the same facts and the same length. Finish with {end}.

### Question:
AI Agent, how tall is Mount Everest?
### Reference Answer:
Mount Everest is 8,849 meters tall.
### Your answer:
Mount Everest stands 8,849 meters high.
{end}

### Question:
{instruction}
### Reference Answer:
{reference}
### Your answer:
"""


def build_sdft_prompt(instruction: str, reference: str, end_marker: str = "###",
                      sdft_style: str = "generic") -> str:
    tpl = SDFT_AGENT_TEMPLATE if sdft_style == "agent" else SDFT_TEMPLATE
    return tpl.format(instruction=instruction, reference=reference, end=end_marker)


def build_prompt(question: str, style: str, end_marker: str = "###") -> str:
    if style == "plain":
        return question
    if style == "qa":
        return f"Question: {question}\nAnswer:"
    if style == "chat":
        return (
            "You are a helpful assistant. Answer the question briefly.\n"
            f"User: {question}\nAssistant:"
        )
    if style == "fewshot":
        return FEWSHOT_TEMPLATE.format(q=question, end=end_marker)
    raise ValueError(f"unknown style: {style}")


@dataclass
class Result:
    question: str
    prompt: str
    completion: str
    token_ids: list = field(default_factory=list)
    first_step_top5: list = field(default_factory=list)  # [(tok, id, prob), ...]
    pad_prob_first_step: float | None = None
    n_generated: int = 0
    stopped_on: str = ""
    reference: str | None = None
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# real backend
# --------------------------------------------------------------------------
class RealBackend:
    """Streams one frame at a time through the Temporal Transformer."""

    def __init__(self, args):
        import sentencepiece  # noqa: F401  (import check)
        import torch

        from moshi.models.loaders import get_moshi_lm

        self.torch = torch
        root = Path(args.model_dir).expanduser()
        weights = Path(args.weights) if args.weights else root / "model.safetensors"
        spm_path = Path(args.tokenizer) if args.tokenizer else root / "tokenizer_spm_32k_3.model"
        for p in (weights, spm_path):
            if not p.exists():
                raise FileNotFoundError(p)

        print(f"[load] lm      : {weights}", flush=True)
        lm = get_moshi_lm(str(weights), device=args.device)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
        lm = lm.to(device=args.device, dtype=dtype)
        lm.eval()
        self.lm = lm

        print(f"[load] spm     : {spm_path}", flush=True)
        import sentencepiece as spm_mod

        self.spm = spm_mod.SentencePieceProcessor(str(spm_path))

        # API differs between forks:
        #   upstream moshi   -> lm.forward_text(codes)
        #   PersonaPlex fork -> lm.forward_codes(codes)  (= embed_codes + forward_embeddings)
        # Both take [B, K, S] codes and return (transformer_out, text_logits).
        for name in ("forward_codes", "forward_text"):
            if hasattr(lm, name):
                self._fwd = getattr(lm, name)
                self._fwd_name = name
                break
        else:
            raise AttributeError(
                "LMModel exposes neither forward_codes nor forward_text; "
                f"available: {[a for a in dir(lm) if a.startswith('forward')]}"
            )

        self.K = lm.num_codebooks
        self.device = args.device
        self.pad_id = lm.text_padding_token_id
        self.epad_id = lm.end_of_text_padding_id
        print(f"[info] temporal forward: lm.{self._fwd_name}()", flush=True)
        print(
            f"[info] num_codebooks={self.K} dep_q={lm.dep_q} text_card={lm.text_card} "
            f"pad={self.pad_id} epad={self.epad_id} delays[0]={lm.delays[0]}",
            flush=True,
        )
        if lm.delays[0] != 0:
            print("[warn] text delay is not 0; this loop assumes it is.", flush=True)

    # -- helpers ----------------------------------------------------------
    def _frame(self, text_token: int):
        torch = self.torch
        f = torch.full(
            (1, self.K, 1), self.lm.zero_token_id, dtype=torch.long, device=self.device
        )
        f[0, 0, 0] = text_token
        return f

    def _initial_frame(self, audio_init: str):
        init = self.lm._get_initial_token().to(self.device)  # [1, K, 1]
        if audio_init == "zero":
            init = init.clone()
            init[:, 1:, :] = self.lm.zero_token_id
        return init

    def _step(self, frame):
        # Embeds all 17 streams, runs the Temporal Transformer and the text head.
        # Audio entries are -1 -> ScaledEmbedding returns exactly zero for them.
        _, text_logits = self._fwd(frame)
        return text_logits[0, 0, -1].float()

    # -- generation -------------------------------------------------------
    def generate(self, question: str, prompt: str, args) -> Result:
        torch = self.torch
        prompt_ids = self.spm.encode(prompt)
        res = Result(question=question, prompt=prompt, completion="")

        with torch.no_grad(), self.lm.streaming(1):
            logits = self._step(self._initial_frame(args.audio_init))
            for tok in prompt_ids:
                if args.trace:
                    print(f"  [feed] text={tok:<6} audio=[-1]*{self.K - 1}", flush=True)
                logits = self._step(self._frame(tok))

            out_ids: list[int] = []
            for i in range(args.max_new_tokens):
                if i == 0:
                    probs = torch.softmax(logits, dim=-1)
                    top = torch.topk(probs, 5)
                    res.first_step_top5 = [
                        (self.spm.id_to_piece(int(t)), int(t), float(p))
                        for p, t in zip(top.values, top.indices)
                    ]
                    res.pad_prob_first_step = float(probs[self.pad_id])

                lg = logits.clone()
                if args.mask_pad:
                    lg[self.pad_id] = -float("inf")
                    lg[self.epad_id] = -float("inf")

                nxt = self._pick(lg, args)
                if nxt == self.spm.eos_id():
                    res.stopped_on = "eos"
                    break
                out_ids.append(nxt)

                hit = self._hit_stop(out_ids, args.stop_str)
                if hit is not None:
                    out_ids = hit[0]
                    res.stopped_on = f"stop_str {hit[1]!r}"
                    break

                logits = self._step(self._frame(nxt))
            else:
                res.stopped_on = "max_new_tokens"

        res.token_ids = out_ids
        res.n_generated = len(out_ids)
        res.completion = self.spm.decode(out_ids)
        return res

    def _hit_stop(self, out_ids: list[int], stop_strs: list[str]):
        """Return (truncated_ids, matched) once a stop string appears, else None."""
        if not stop_strs:
            return None
        text = self.spm.decode(out_ids)
        for s in stop_strs:
            s = s.replace("\\n", "\n")
            idx = text.find(s)
            if idx >= 0:
                kept = text[:idx]
                return self.spm.encode(kept), s
        return None

    def _pick(self, logits, args) -> int:
        torch = self.torch
        if args.temp <= 0:
            return int(torch.argmax(logits))
        lg = logits / args.temp
        if args.top_k > 0:
            kth = torch.topk(lg, args.top_k).values[-1]
            lg = torch.where(lg < kth, torch.full_like(lg, -float("inf")), lg)
        probs = torch.softmax(lg, dim=-1)
        return int(torch.multinomial(probs, 1))


# --------------------------------------------------------------------------
# dry-run backend (no torch, no weights) — exercises the same control flow
# --------------------------------------------------------------------------
class DryBackend:
    """Stub with the same interface, so the loop itself can be checked."""

    VOCAB = 32000

    def __init__(self, args):
        self.K = 17
        self.pad_id = 3
        self.epad_id = 0
        self.rng = random.Random(args.seed)
        print(
            "[dry] no model loaded; simulating "
            f"num_codebooks={self.K} dep_q=16 text_card={self.VOCAB} "
            f"pad={self.pad_id} epad={self.epad_id}",
            flush=True,
        )

    def _encode(self, text: str) -> list[int]:
        return [abs(hash(w)) % self.VOCAB for w in text.split()]

    def _decode(self, ids: list[int]) -> str:
        return " ".join(f"<tok{i}>" for i in ids)

    def _fake_logits(self) -> list[float]:
        return [self.rng.random() for _ in range(64)] + [0.0] * (self.VOCAB - 64)

    def generate(self, question: str, prompt: str, args) -> Result:
        ids = self._encode(prompt)
        res = Result(question=question, prompt=prompt, completion="")

        print(f"  [dry] prompt tokens: {ids}", flush=True)
        print(
            f"  [dry] initial frame : shape (1, {self.K}, 1) "
            f"text=<sos> audio={'[-1]*16' if args.audio_init == 'zero' else '[special]*16'}",
            flush=True,
        )
        for tok in ids:
            print(f"  [dry] feed frame    : text={tok:<6} audio=[-1]*16", flush=True)

        out_ids = []
        for i in range(min(args.max_new_tokens, 5)):
            logits = self._fake_logits()
            if i == 0:
                res.first_step_top5 = [("<dry>", j, 0.0) for j in range(5)]
                res.pad_prob_first_step = 0.0
            if args.mask_pad:
                logits[self.pad_id] = -math.inf
                logits[self.epad_id] = -math.inf
            nxt = max(range(len(logits)), key=lambda j: logits[j])
            out_ids.append(nxt)
            print(f"  [dry] sampled       : {nxt} -> feed back as next frame", flush=True)

        res.stopped_on = "dry_limit"
        res.token_ids = out_ids
        res.n_generated = len(out_ids)
        res.completion = self._decode(out_ids)
        return res


def row_key(meta: dict, question: str) -> str:
    """Stable id for resume: the utterance it came from, else the question."""
    if meta.get("dialogue") and meta.get("a_utt"):
        return f"{meta['dialogue']}::{meta['a_utt']}"
    return question


NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")

#: The persona's refusals ("I don't have that information") must never be
#: rewritten: the model happily turns them into invented facts — one sample
#: turned "I don't have current figures for that limit." into a confident
#: "around $20,500". These are skipped before generation, not filtered after.
REFUSAL_RE = re.compile(
    r"\b(i don'?t (have|know)"
    r"|i'?m not (sure|certain)"
    r"|no (current|real[- ]time) (figures?|information|data)"
    r"|don'?t have (that|current|real[- ]time)"
    # "That varies enormously by breed, so I can't give one number."
    r"|(can'?t|cannot) (say|give|provide|tell)"
    r"|no (single|exact|precise) (number|figure|answer))\b", re.I)

#: Vague quantity words carry the fact when no digit does ("thousands of
#: varieties"); losing one silently weakens the answer.
QUANT_WORDS = {"thousands", "hundreds", "millions", "billions", "dozens", "dozen",
               "several", "few", "couple", "many", "most", "all", "none", "no",
               "half", "twice", "double", "triple", "one", "two", "three", "four",
               "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
               "zero", "single", "every", "each"}


def check_rewrite(rewritten: str, reference: str, args) -> tuple[bool, str]:
    """SDFT's verification step: keep the rewrite only if it is a faithful,
    similarly short restatement; otherwise the original answer is kept."""
    if not rewritten:
        return False, "empty"
    if rewritten.strip() == (reference or "").strip():
        return False, "identical"
    if REFUSAL_RE.search(reference or ""):
        return False, "reference is a refusal"
    def norm_nums(text: str) -> list[str]:
        # "43,560" and "43560" are the same number written two ways; so are
        # "20.0" and "20". Only real changes should be rejected.
        out = []
        for n in NUM_RE.findall(text or ""):
            n = n.replace(",", "")
            if "." in n:
                n = n.rstrip("0").rstrip(".")
            out.append(n or "0")
        return sorted(out)

    ref_nums, new_nums = norm_nums(reference), norm_nums(rewritten)
    if ref_nums != new_nums:
        return False, f"numbers changed {ref_nums}->{new_nums}"
    ref_q = {w.strip(".,;:!?'\u2019").lower() for w in (reference or "").split()} & QUANT_WORDS
    new_q = {w.strip(".,;:!?'\u2019").lower() for w in rewritten.split()} & QUANT_WORDS
    if ref_q - new_q:
        return False, f"quantity word dropped {sorted(ref_q - new_q)}"
    ref_w = max(1, len((reference or "").split()))
    ratio = len(rewritten.split()) / ref_w
    if ratio > args.max_len_ratio:
        return False, f"too long ({ratio:.2f}x)"
    if ratio < args.min_len_ratio:
        return False, f"too short ({ratio:.2f}x)"
    return True, ""


# --------------------------------------------------------------------------
def load_questions(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_QUESTIONS
    lines = Path(path).read_text().splitlines()
    # one prompt per line; literal "\n" in the file becomes a real newline, so
    # multi-line prompts (e.g. the SDFT rewrite template) still fit on one line.
    return [ln.strip().replace("\\n", "\n")
            for ln in lines if ln.strip() and not ln.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("PERSONAPLEX_DIR", ""),
                    help="PersonaPlex snapshot dir (default: $PERSONAPLEX_DIR)")
    ap.add_argument("--weights", default="", help="override path to model.safetensors")
    ap.add_argument("--tokenizer", default="", help="override path to the SentencePiece model")
    ap.add_argument("--questions", default="", help="file with one prompt per line")
    ap.add_argument("--pairs", default="",
                    help="JSONL of {\"instruction\":..., \"response\":...} — runs the SDFT "
                         "rewrite task on each pair instead of plain prompting")
    ap.add_argument("--style", default="qa", choices=["plain", "qa", "chat", "fewshot"])
    ap.add_argument("--no-skip-refusals", dest="skip_refusals", action="store_false",
                    default=True,
                    help="also try to rewrite refusals (they get fabricated into facts — "
                         "skipping them is the default)")
    ap.add_argument("--resume", action="store_true",
                    help="append to --distilled-out and skip rows already in it")
    ap.add_argument("--distilled-out", default="",
                    help="with --pairs: write the distilled dataset here (JSONL, original "
                         "fields + response_rewritten, failing rewrites fall back)")
    ap.add_argument("--max-len-ratio", type=float, default=1.6,
                    help="reject a rewrite longer than this multiple of the original")
    ap.add_argument("--min-len-ratio", type=float, default=0.5,
                    help="reject a rewrite shorter than this multiple of the original")
    ap.add_argument("--sdft-style", default="generic", choices=["generic", "agent"],
                    help="'agent': keep the AI-Agent persona (one short factual sentence)")
    ap.add_argument("--end-marker", default="###",
                    help="terminator demonstrated by --style fewshot; also used as a stop string")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--stop-str", action="append", default=[],
                    help="cut generation when this string appears (repeatable; use \\n for newline). "
                         "Needed because the model happily keeps inventing follow-up questions.")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--mask-pad", dest="mask_pad", action="store_true", default=True,
                    help="forbid PAD/EPAD so the model must emit real words (default)")
    ap.add_argument("--no-mask-pad", dest="mask_pad", action="store_false",
                    help="let PAD/EPAD be sampled — shows the raw speech-mode behaviour")
    ap.add_argument("--audio-init", default="zero", choices=["zero", "special"],
                    help="audio streams in the very first frame: -1, or moshi's initial token")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", action="store_true", help="print every frame that is fed")
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--dry-run", action="store_true", help="no torch, no weights: check the flow only")
    args = ap.parse_args()

    if args.style == "fewshot" and not args.stop_str:
        args.stop_str = [args.end_marker]

    if args.pairs:
        items = []
        for line in Path(args.pairs).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            instr, ref = obj["instruction"], obj["response"]
            items.append((instr,
                          build_sdft_prompt(instr, ref, args.end_marker, args.sdft_style),
                          ref, obj))
        if not args.stop_str:
            args.stop_str = [args.end_marker]
    else:
        items = [(q, build_prompt(q, args.style, args.end_marker), None, {})
                 for q in load_questions(args.questions or None)]
    questions = [q for q, _, _, _ in items]
    print(f"[cfg] style={args.style} mask_pad={args.mask_pad} temp={args.temp} "
          f"top_k={args.top_k} max_new_tokens={args.max_new_tokens} n_questions={len(questions)}",
          flush=True)

    if args.dry_run:
        backend = DryBackend(args)
    else:
        if not args.model_dir and not args.weights:
            print("error: --model-dir (or $PERSONAPLEX_DIR) is required without --dry-run",
                  file=sys.stderr)
            return 2
        backend = RealBackend(args)

    done_keys: set[str] = set()
    distilled_f = None
    if args.distilled_out and args.pairs:
        out_path = Path(args.distilled_out)
        if args.resume and out_path.exists():
            for line in out_path.read_text().splitlines():
                line = line.strip()
                if line:
                    row = json.loads(line)
                    done_keys.add(row_key(row, row.get("instruction", "")))
            print(f"[resume] {len(done_keys)} rows already in {out_path}; skipping those",
                  flush=True)
        distilled_f = open(out_path, "a" if args.resume else "w")

    results = []
    n_skipped, n_ok, n_fallback, reasons = 0, 0, 0, {}

    def emit(res: "Result") -> None:
        """Append one distilled row immediately, so a crash costs one item."""
        nonlocal n_ok, n_fallback
        if distilled_f is None:
            return
        row = dict(res.meta)
        rewritten = (res.completion or "").strip()
        if res.stopped_on == "skipped_refusal":
            ok, why = False, "skipped: refusal"
        else:
            ok, why = check_rewrite(rewritten, res.reference, args)
        row["response_original"] = res.reference
        row["response_rewritten"] = rewritten
        row["response"] = rewritten if ok else res.reference
        row["rewrite_ok"] = ok
        row["rewrite_reject_reason"] = why
        row["stopped_on"] = res.stopped_on
        distilled_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        distilled_f.flush()
        if ok:
            n_ok += 1
        else:
            n_fallback += 1
            reasons[why] = reasons.get(why, 0) + 1

    for q, prompt, ref, meta in items:
        if row_key(meta, q) in done_keys:
            continue
        print(f"\n=== {q}", flush=True)
        if ref is not None:
            print(f"  reference: {ref!r}", flush=True)
        if ref and args.skip_refusals and REFUSAL_RE.search(ref):
            print("  [skip] reference is a refusal — keeping it verbatim", flush=True)
            res = Result(question=q, prompt=prompt, completion="",
                         stopped_on="skipped_refusal")
            n_skipped += 1
            res.reference = ref
            res.meta = meta
            results.append(res.__dict__)
            emit(res)
            continue
        res = backend.generate(q, prompt, args)
        res.reference = ref
        res.meta = meta
        if res.first_step_top5:
            top = ", ".join(f"{p!r}:{prob:.3f}" for p, _, prob in res.first_step_top5)
            print(f"  top5@first_step: {top}", flush=True)
            print(f"  P(PAD)@first_step: {res.pad_prob_first_step:.3f}", flush=True)
        print(f"  -> {res.completion!r}  [{res.n_generated} tokens, {res.stopped_on}]", flush=True)
        results.append(res.__dict__)
        emit(res)

    if distilled_f is not None:
        distilled_f.close()
        total = n_ok + n_fallback
        print(f"\n[distill] kept {n_ok} rewrites, kept the original {n_fallback} times "
              f"({100 * n_ok / max(1, total):.0f}% rewritten) -> {args.distilled_out}")
        for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {why}: {n}")

    if n_skipped:
        print(f"\n[skip] {n_skipped} refusal answers kept verbatim", flush=True)

    Path(args.out).write_text(json.dumps(
        {"config": vars(args), "results": results}, indent=2, ensure_ascii=False))
    print(f"\n[done] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
