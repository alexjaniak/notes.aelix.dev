# Day 1 Writeup — The spec sheet called it

*Polished narrative for [[Day 1 — Local Inference]] (Unit 1.1, [[Curriculum]]). Lab-notebook warts stay in the day note; this is the version the post gets built from.*

**The claim:** LLM decode is memory-bandwidth-bound. If true, a laptop's *spec sheet* — no benchmark, no code — should predict its token rate. It did, to 13% on the absolute number and **0.5% on the context-scaling law**.

## Setup

- Apple M4, 24 GB unified memory, **120 GB/s** bandwidth (Apple spec). Unified memory means CPU and GPU (Metal) share one bus — one number predicts both.
- llama.cpp (brew), Metal backend. Model: **Qwen3-8B, Q4_K_M GGUF** — 8.19B params, file = 4.68 GiB = **5.03 GB**.
- Instrument: `llama-bench` — `pp512` = prefill tok/s (512-token prompt, batched), `tg128` = decode tok/s (128 tokens, one at a time). 5 reps each + warmup, mean ± σ.

## The math

**Decode (bandwidth roofline).** Batch-1 generation touches every weight once per token with zero reuse — the cost is streaming the model across the bus, not the FLOPs:

> decode ceiling = bandwidth ÷ bytes per token = 120 GB/s ÷ 5.03 GB = **23.9 tok/s**

**KV cache (the context tax).** Each token of context adds cached K/V that decode must also read every step:

> bytes/token of context = 2 (K+V) × 36 layers × 8 KV heads × 128 head_dim × 2 bytes (fp16) = **144 KB**

At depth d, bytes per decode step = 5.03 GB + d × 144 KB → at 8k: 6.24 GB → predicted slowdown to **81%** of baseline. (The 8 is *KV* heads — GQA: 32 query heads share 8 K/V heads, a 4× cache compression.)

**Prefill (compute roofline).** A forward pass costs ≈ 2P FLOPs/token = 16.4 GFLOPs. Batched over 512 tokens, weights are reused — the bus stops mattering and the M4's ~4.4 TFLOPS takes over:

> prefill ceiling = 4.4e12 ÷ 16.4e9 ≈ **270 tok/s**

Same game, other spec: the machine does ~37 FLOPs per byte of bandwidth (4.4 TFLOPS ÷ 120 GB/s); decode offers it ~2 FLOPs per byte, prefill offers hundreds. That ratio *is* the prefill/decode gap.

## Predictions vs measurements

Predictions written before running (see [[Day 1 — Local Inference]] for the untouched originals):

| Quantity | Predicted | Measured | Verdict |
|---|---|---|---|
| Model file size | ~5 GB | 5.03 GB | ✅ |
| Decode (short ctx) | ≤ 23.9 tok/s ceiling | 20.80 ± 0.05 | **87% of the bus limit** |
| Prefill | guessed 360 tok/s | 228.94 ± 0.10 | ❌ miss — see below |
| Decode @ 8k vs 0 | 81% of baseline | 80.6% | **0.5% error** |

Context sweep (`-d 0,2048,4096,8192`), predictions re-anchored on that run's baseline since we're testing the *scaling law*:

| depth | bytes/token | predicted tok/s | measured tok/s | error |
|---|---|---|---|---|
| 0 | 5.03 GB | 19.82 (anchor) | 19.82 ± 0.38 | — |
| 2,048 | 5.33 GB | 18.7 | 19.20 ± 0.28 | 2.7% |
| 4,096 | 5.63 GB | 17.7 | 17.99 ± 0.26 | 1.6% |
| 8,192 | 6.24 GB | 16.05 | 15.97 ± 0.06 | **0.5%** |

![[xdoi-day1-decode-vs-context.png]]

**The prefill miss is the honest content.** "10–20× decode" was folklore, and grabbing the midpoint (→360) ignored that 360 would *exceed the compute roofline* (270). Derived properly, measured prefill = 228/270 = **85% of the compute ceiling** — the same ~85–87% engine efficiency in both regimes, each against its own roof. Lesson: don't guess multipliers, derive the second roofline too.

**Bonus finding:** prefill *also* degrades with depth (228 → 154 by 8k, −32%) — but for a different reason than decode. Decode pays for context in **bytes** (KV cache reads); prefill pays in **FLOPs** (each prompt token attends to all d cached tokens — the parameter-free QK^T/AV math is ~590k × d FLOPs/token, ~30% on top of 2P at 8k). Same cause, two currencies.

## Caveats collected along the way

- **GiB ≠ GB.** llama.cpp reports 4.68 *GiB*; bandwidth specs are decimal *GB/s*. 4.68 GiB = 5.03 GB — a silent 7% error if skipped.
- **Q4_K_M ≈ 4.9 bits/weight effective, not 4.** Block scales + the "_M" mixed recipe (sensitive tensors kept at Q6_K). File-size ratio, not nominal bit ratio, is what predicts speed ratios — Q8/Q4 file ratio is ~1.73×, not 2×.
- **The KV cache stays fp16** even in a Q4 model — only weights are quantized (llama.cpp default).
- **2P FLOPs/token omits the parameter-free attention math.** Fine at short context (~2% at d=0), ~30% extra at 8k, dominant at ~28k+. "Attention is the bottleneck" is a claim about scaling, not the constant.
- **tg128 is synthetic:** starts from an empty cache, feeds arbitrary tokens, no sampling — the forward pass costs the same regardless of token identity, so it cleanly isolates the engine.
- **Run-to-run variance is real:** baseline was 20.80 ± 0.05 on a quiet machine, 19.82 ± 0.38 with background load. Watch σ; it's the honesty meter.
- Expect **70–90% of any roofline**, not 100% — memory-controller overhead, non-weight work. Apple's unified memory is at the high end.

## Q8 experiment

Same logic, reversed: Q8_0 ≈ 8.5 bits/weight → ~8.7 GB file → decode should scale down by the **file-size ratio** (1.73×), not the nominal bit ratio (2×); prefill ~unchanged (FLOPs don't care how weights are stored).

| Quantity | Predicted | Measured | Verdict |
|---|---|---|---|
| File size | ~8.7 GB (8.1 GiB) | 8.71 GB (8.11 GiB) | ✅ |
| Decode | ~11.5–12.0 tok/s | 12.52 ± 0.08 | **91% of the 13.8 ceiling** |
| Prefill | ~228 tok/s | 226.70 ± 7.47 | ✅ unchanged — compute-bound confirmed |

**"Does decode halve?" No — 1.66×, and the miss decomposes cleanly.** Nominal bits promised 2×; real file bytes promised 1.73× (Q4_K_M is ~4.9 bits effective); measured was 1.66×. The residual is *format efficiency*: Q8_0 hit 91% of its roofline vs Q4_K_M's 87% — plain 8-bit blocks dequantize more cheaply than Q4_K's nested super-block scales. Bytes predict the ratio to first order; format overhead is the second-order term. Practical reading: quantization buys slightly less speedup than the bytes alone promise, and the cheaper the format's decode path, the closer you run to the bus.

## CPU control experiment (`-ngl 0`)

The falsifiable test: switching off Metal removes most of the compute but keeps the exact same memory bus (unified memory). If decode is bandwidth-bound, it should barely care; prefill should crater.

| Phase | GPU (Metal) | CPU only | Drop |
|---|---|---|---|
| Prefill | 228.94 | 22.15 ± 1.74 | **÷10** — compute-bound confirmed |
| Decode | 19.82–20.80 | 12.69 ± 1.36 | ÷1.6 — see below |

Prefill collapsed 10× exactly as predicted. Decode dropped more than "barely" — and inverting the formula explains it. Compute the bandwidth each backend actually pulled:

> GPU: 19.82 tok/s × 5.03 GB ≈ **100 GB/s** (83% of spec)
> CPU: 12.69 tok/s × 5.03 GB ≈ **64 GB/s** (53% of spec)

Decode is still bandwidth-bound on CPU — cores that do 22 tok/s of prefill *compute* aren't compute-limited at 12.7 tok/s of decode. The refinement: **"the bus is 120 GB/s" ≠ "any client can pull 120 GB/s."** A 4-core CPU cluster has its own link into the memory fabric (~60–65 GB/s achievable); the GPU's dozens of parallel memory-hungry cores extract far more. Spec-sheet bandwidth is the chip's ceiling; each compute unit has a lower, achievable one. Final formula: `tok/s = achievable bandwidth ÷ bytes per token` — and *achievable* depends on who's asking. (Untested: `-t 8` — do the efficiency cores pull more bandwidth or straggle? Also note the ±1.36 stddev, ~11% — noisier than every other run.)

## X post — final draft (single long post, chart attached)

> Day 1/45 of Inference Engineering: Basic Local Inference
>
> Thanks to @maxxfuu & @mohitwt_ , I've been inspired to take on this challenge and hopefully you guys can learn alongside me.
>
> I've decided to start easy and just get Qwen3 8B running locally on my M4 24GB MacBook Pro (llama.cpp) — with one rule: predict every number from the spec sheet before running anything.
>
> The theory: generating a token means reading every weight once. No reuse at batch 1. So decode isn't a compute problem — it's a memory-bandwidth problem, and the ceiling is just division:
>
> 120 GB/s (M4 bus) ÷ 5.03 GB (Q4 file) ≈ 24 tok/s
>
> Measured with llama-bench: 20.8 — 87% of the theoretical bus limit, predicted before a single token ran.
>
> Three more predictions, all from paper math:
>
> • KV cache adds 144 KB per token of context (2 × 36 layers × 8 KV heads × 128 dims × fp16, straight from config.json) → predicted decode at 8k context falls to 81% of baseline. Measured: 80.6%.
>
> • Q8 quant: the file is 1.73× bigger than Q4, so decode should be 1.73× slower — not the 2× the nominal bits promise. Measured: 1.66×.
>
> • The control: CPU-only (-ngl 0), same unified-memory bus, most of the compute gone. Prefill cratered 10×. Decode barely moved (÷1.6). Decode never needed the FLOPs.
>
> What surprised me: my prefill guess missed by 60% — until I derived the compute roofline too (2P FLOPs/token ÷ 4.4 TFLOPS ≈ 270 tok/s), and the same ~85% efficiency fell out of both regimes. Two phases, two spec-sheet numbers, one engine.
>
> The formula for your own machine: decode tok/s ≈ memory bandwidth ÷ model file size. That's it.
>
> Day 2: renting a real GPU and standing up my own endpoint.
>
> [attach: xdoi-day1-decode-vs-context.png]

## X post — earlier alternates

Thread — one experiment per tweet, chart on the hook:

> **1/** Before running a single token, my laptop's spec sheet predicted its LLM speed.
>
> M4: 120 GB/s memory bandwidth. Qwen3-8B Q4: a 5.03 GB file. Decode reads every weight for every token, so ceiling = 120 ÷ 5.03 ≈ 24 tok/s.
>
> Measured: 20.8 — 87% of the bus limit. And that was the *least* accurate prediction of the day. 🧵
>
> [attach: xdoi-day1-decode-vs-context.png]

> **2/** Long context: every token in the KV cache adds 144 KB that decode must re-read each step (2 × 36 layers × 8 KV heads × 128 dims × fp16 — straight out of config.json).
>
> At 8k context that's +1.2 GB per token → predicted 81% of baseline speed. Measured: 80.6%. The scaling law landed within 0.5%.

> **3/** Does going Q4 → Q8 halve decode speed? No — 1.66×.
>
> Nominal bits promise 2×. Real file bytes promise 1.73× (Q4_K_M is ~4.9 bits/weight effective, not 4). Format overhead eats the rest — Q8_0's simpler blocks ran at 91% of the roofline vs Q4's 87%.
>
> Bytes are the first-order truth; dequant cost is the second-order term.

> **4/** The control experiment: kill the GPU, keep the bus (`-ngl 0`).
>
> Prefill: 229 → 22 tok/s. ÷10. Compute-bound, compute gone.
> Decode: 19.8 → 12.7. ÷1.6. It never needed the FLOPs.
>
> And the decode gap is *itself* bandwidth: the GPU pulls ~100 GB/s of the 120; 4 CPU cores can only pull ~64. Achievable bandwidth depends on who's asking.

> **5/** The reusable formula for any machine:
>
> decode tok/s ≈ achievable memory bandwidth ÷ (model file size + 144-ish KB × context)
>
> What surprised me: my prefill guess missed by 60% — until I derived the *compute* roofline (2P FLOPs/token ÷ 4.4 TFLOPS ≈ 270), and the same ~85% efficiency fell out of both regimes.
>
> Tomorrow: renting a real GPU and standing up my own endpoint.

*Single-post version (if not threading):*

> My MacBook's spec sheet predicted its LLM decode speed before I ran anything: 120 GB/s ÷ 5.03 GB model = 24 tok/s ceiling; measured 20.8 (87%). KV-cache bytes predicted the 8k-context slowdown to 0.5%. Q8 vs Q4 scaled by file-size ratio, not nominal bits. And CPU-only barely touched decode while prefill fell 10× — same bus, no compute. Decode is a memory-bandwidth problem; the chart is the proof.

## What surprised me — candidates

- The 0.5% error at 8k: a number derived from `config.json` fields showed up on a stopwatch.
- The CPU run breaking the naive theory in an instructive way: decode dropped 1.6× not because bandwidth stopped mattering, but because 4 cores can only *pull* half the bus. "Achievable bandwidth" became the day's most refined concept.
- Prefill folklore ("10–20×") vs deriving the compute roofline — the miss that taught the most.
- Prefill degrading *more* than decode with depth (FLOPs tax vs bytes tax).
- `dflash-`/`dspark-` draft models in the wild: speculative decoding exists *because* of everything above — decode wastes 95% of the compute, so fake a batch.
