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
def build_prompt(question: str, style: str) -> str:
    if style == "plain":
        return question
    if style == "qa":
        return f"Question: {question}\nAnswer:"
    if style == "chat":
        return (
            "You are a helpful assistant. Answer the question briefly.\n"
            f"User: {question}\nAssistant:"
        )
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
    def generate(self, question: str, args) -> Result:
        torch = self.torch
        prompt = build_prompt(question, args.style)
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
                logits = self._step(self._frame(nxt))
            else:
                res.stopped_on = "max_new_tokens"

        res.token_ids = out_ids
        res.n_generated = len(out_ids)
        res.completion = self.spm.decode(out_ids)
        return res

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

    def generate(self, question: str, args) -> Result:
        prompt = build_prompt(question, args.style)
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


# --------------------------------------------------------------------------
def load_questions(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_QUESTIONS
    lines = Path(path).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("PERSONAPLEX_DIR", ""),
                    help="PersonaPlex snapshot dir (default: $PERSONAPLEX_DIR)")
    ap.add_argument("--weights", default="", help="override path to model.safetensors")
    ap.add_argument("--tokenizer", default="", help="override path to the SentencePiece model")
    ap.add_argument("--questions", default="", help="file with one prompt per line")
    ap.add_argument("--style", default="qa", choices=["plain", "qa", "chat"])
    ap.add_argument("--max-new-tokens", type=int, default=48)
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

    questions = load_questions(args.questions or None)
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

    results = []
    for q in questions:
        print(f"\n=== {q}", flush=True)
        res = backend.generate(q, args)
        if res.first_step_top5:
            top = ", ".join(f"{p!r}:{prob:.3f}" for p, _, prob in res.first_step_top5)
            print(f"  top5@first_step: {top}", flush=True)
            print(f"  P(PAD)@first_step: {res.pad_prob_first_step:.3f}", flush=True)
        print(f"  -> {res.completion!r}  [{res.n_generated} tokens, {res.stopped_on}]", flush=True)
        results.append(res.__dict__)

    Path(args.out).write_text(json.dumps(
        {"config": vars(args), "results": results}, indent=2, ensure_ascii=False))
    print(f"\n[done] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
