# moshi_text_probe

Does PersonaPlex / Moshi still work as a **plain text LM** when no audio is fed?

## Why this question matters

Moshi's Temporal Transformer was initialised from **Helium**, a text-only 7B LM,
and Moshi was trained with text-only batches (50% during pre-training, 10%
during post-training). Feeding `zero_token_id` (-1) on all 16 audio streams
makes their embeddings exactly zero, so the model collapses to

```
text_emb -> Temporal Transformer -> text_linear
```

i.e. Helium's original path. **But** the later stages dropped text-only batches
(Moshi's Fisher fine-tune: "We no longer sample full text batches"), and
PersonaPlex then fine-tuned further on speech only — so how much of that text
ability survives is untested.

If it survives, we can have the model rewrite answers in its own words
(self-distillation, [arXiv:2402.13669](https://arxiv.org/abs/2402.13669))
**without synthesising any audio**.

## Why not `LMGen`

`LMGen` is the streaming inference driver, not the model. Two blockers:

1. `needed_tokens = num_codebooks - dep_q - 1` is **0** for PersonaPlex
   (`dep_q=16`), so `step()` accepts no input streams — there is no way to feed
   a text prompt in.
2. It runs the Depth Transformer every step to generate audio.

So this script calls `lm.forward_text()` directly — the Temporal half only.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
```

Use the PersonaPlex checkout for `moshi` (do **not** pip install moshi as well):

```bash
export PYTHONPATH=/path/to/personaplex/moshi:$PWD
export PERSONAPLEX_DIR=/path/to/personaplex/snapshot
```

## Run

Check the control flow with no model at all:

```bash
python probe_text_only.py --dry-run --questions questions.txt
```

Then on a GPU node:

```bash
./run_hpc.sh
# or directly:
python probe_text_only.py --questions questions.txt --style qa --temp 0
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--style plain\|qa\|chat` | prompt format |
| `--no-mask-pad` | let PAD/EPAD be sampled (shows raw speech-mode behaviour) |
| `--audio-init zero\|special` | audio in the first frame: `-1`, or moshi's initial token |
| `--temp 0` | greedy (default); `--temp 0.7 --top-k 25` to sample |
| `--trace` | print every frame fed into the model |

## Reading the results

Each question reports the top-5 next tokens at the first generated step,
`P(PAD)` at that step, and the completion.

| Observation | Reading |
|---|---|
| Sensible answers ("two", "Paris") | Text-only path is alive → self-distillation without TTS is on |
| Grammatical but wrong answers | Language survives, knowledge/instruction-following weakened |
| With `--no-mask-pad`, output is all PAD | It still expects the speech regime; masking PAD is mandatory |
| Garbage even with PAD masked | Text-only path has degraded → fall back to speech-mode generation + ASR |

`--audio-init zero` vs `special` is worth comparing: the training-time frames
almost certainly used moshi's initial token in the first position, so if `zero`
looks broken, try `special` before concluding anything.

## Status

Only the dry run has been executed (no torch on the dev machine). Nothing here
has been validated against real weights yet.

## Next step if the probe passes

Swap `questions.txt` for the SDFT rewrite template and batch-generate the
distilled dataset:

```
Below are an instruction that describes a task along with a reference answer.
Using the reference answer as a guide, write your own response.

### Instruction:
{instruction}
### Reference Answer:
{original response}
### Response:
```
