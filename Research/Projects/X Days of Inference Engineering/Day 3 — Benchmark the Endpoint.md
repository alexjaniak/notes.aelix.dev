# Day 3 — Benchmark the Endpoint

*Unit from Arc 1 of [[Curriculum]]. Follows [[Day 2 — Rent a GPU, Serve an Endpoint]]. Pair with a skim of Orca (OSDI '22) — iteration-level scheduling is the mechanism being measured today.*

**The question:** batch-1 decode is a waste of an A100 — every token reads all 16.38 GB of weights to do ~16.4 GFLOPs, an arithmetic intensity of ~1 FLOP/byte on a card whose ridge point is ~161. Continuous batching amortizes the weight read across concurrent requests. Can the roofline predict the knee — the concurrency where the GPU stops being memory-bound — before the sweep runs?

Day 2 was plumbing with one number at the end. Today is five predictions, each derived from chip specs and the model config before touching the box. The sweep only gets run after every blank below is filled in ink.

## The mechanism (~30 min, before the math)

[[Serving Models]] has one line on this ("requests join/leave every token step, up to ~23x vs static") — today that claim gets a mechanism and a measurement. Skim Orca (OSDI '22) §1–3 plus the [Anyscale continuous-batching post](https://www.anyscale.com/blog/continuous-batching-llm-inference), then answer these in my own words at the top of the Measurements section:

- **Static batching's two wastes.** A static batch is admitted together and released together, padded to the longest sequence. Name both costs: the padding FLOPs, and the head-of-line blocking where a 20-token completion holds its slot while a 2,000-token neighbor finishes. Which one does Arc 2.5 measure directly?
- **Orca's two ideas, separately.** (1) *Iteration-level scheduling*: the schedulable unit is one decode step, not one request — finished sequences exit and queued ones enter between any two steps. (2) *Selective batching*: the matmuls (QKV/MLP projections) batch fine across ragged sequences because they're per-token, but attention can't — different KV lengths per sequence — so it runs per-sequence while everything else batches. Which of the two does vLLM keep, and what did PagedAttention replace?
- **Why decode makes this possible at all.** Every request needs the identical weight pass every step, so a decode step is a natural synchronization point and batching = amortizing the one weight read — this is *the same fact* as the arithmetic-intensity ≈ B claim in the math below. If that isn't obviously the same fact, stop and make it obvious.
- **What admission actually costs.** When the scheduler admits a new request mid-flight, its prefill has to run somewhere — either stalling decode iterations (TTFT for it, TPOT spike for everyone else) or chunked in alongside them. That tension is prediction 5, and the `/metrics` scrape should show it.

Arc 2.5/2.6 (padding-waste measurement, Rust batching simulator) build this from scratch later; today is concept + live measurement.

## Debts from Day 2 (do first)

- [ ] Check what Prime Intellect actually offers for volume/image persistence; write the resurrection steps at the top of this note once known
- [ ] Get a box/image with a working CUDA toolkit so FlashInfer's fused sampler JITs (`nvcc --version` before anything else; grep the vLLM startup log to confirm the fused path). This removes Day 2's 63.5%-of-ceiling asterisk
- [ ] Re-measure clean batch-1 decode: predict ___ tok/s (Day 1 hit 87% of ceiling on the Mac; ceiling here is 118.12)
- [ ] `nvidia-smi` before and after `vllm serve` — capture the "80 GB looks full at idle" moment. That's KV pre-allocation, not weights; the block math below says exactly how much

## The math (fill in predictions BEFORE running)

Inputs, all public or from Day 2: A100 80GB PCIe **1,935 GB/s** HBM, **312 TFLOPS** dense bf16. Qwen3-8B: P ≈ 8.2B params, **16.38 GB** bf16 weights, 36 layers × 8 KV heads × 128 head dim → **147 KB/token** of KV in bf16 (the Day 1 formula, same chips different bus).

**1. The knee, from the roofline.** One decode step at batch B: compute ≈ B × 2P FLOPs; weight traffic = 2P bytes, read once and shared across the batch. So arithmetic intensity ≈ B FLOPs/byte (KV ignored for now). Ridge point = 312e12 ÷ 1,935e9 ≈ **161 FLOPs/byte**.
- Naive predicted knee B* (intensity = ridge): ___
- Now add KV: each request also streams its own cache every step — kv_bytes(B, ctx) = B × ctx × 147 KB. At 1k context, per-step traffic = 16.38 GB + B × 0.147 GB. Intensity(B) = 2PB ÷ (2P·2bytes + B·ctx·147KB)... solve for where it crosses 161, or note that it asymptotes below the ridge and the "knee" is really a flattening: revised B* ≈ ___
- Sanity: what does vLLM's default `max_num_seqs` cap the running batch at, and does it bind before B*? ___

**2. Three points on the predicted throughput curve.** Memory-bound step time ≈ (16.38 GB + B × ctx × 147 KB) ÷ 1,935 GB/s; aggregate tok/s ≈ B ÷ step_time. At 1k context:
- B=1: ___ tok/s (should reproduce the 118 ceiling)
- B=16: ___ tok/s
- B=64: ___ tok/s, and per-request tok/s = aggregate ÷ 64 = ___ (the graceful-degradation number: how much does each user feel 63 neighbors?)

**3. KV block accounting, predicted from memory.** vLLM takes `gpu_memory_utilization` (default 0.9) of 80 GB → ~72 GB budget; subtract 16.38 GB weights and ~a few GB activation/workspace → predicted KV pool ≈ ___ GB → ÷ 147 KB/token → ___ max cached tokens → ÷ block_size 16 → predicted `num_gpu_blocks` ≈ ___. The startup log prints the real number; target within ±10%.
- Corollary: max concurrent 1k-token-context requests before preemption ≈ cached tokens ÷ ~2k (1k prompt + 1k gen) = ___

**4. Little's law cross-check.** L = λW: concurrency ≈ throughput (req/s) × mean request latency (s). This must hold at every sweep point in a closed-loop load generator — if it doesn't, I'm misreading what `vllm bench serve` holds constant (closed-loop fixed concurrency vs open-loop fixed arrival rate are different queueing systems; know which one is running). Networking day job, finally admissible.

**5. TTFT under load.** Prediction: TTFT stays near-flat while the scheduler admits prefills immediately, then p99 blows up once requests queue (running batch or KV blocks exhausted, or prefills serialize behind decode iterations). Predicted concurrency where p99 TTFT > 2× p50: ___

## Steps

- [ ] Resurrect endpoint; confirm fused sampler; capture startup log — pull `num_gpu_blocks`, `max_num_seqs`, KV pool size → score prediction 3
- [ ] Clean batch-1 baseline → score the Day 2 debt
- [ ] Sweep concurrency 1, 2, 4, 8, 16, 32, 64, 128 with `vllm bench serve` (genai-perf as fallback), fixed ~1k in / ~1k out, ≥3 min per point. Record: aggregate tok/s, per-request tok/s, TTFT p50/p99, TPOT (time per output token) p50/p99
- [ ] Scrape `/metrics` during the sweep — `vllm:num_requests_running`, `num_requests_waiting`, `gpu_cache_usage_perc`, preemption counters. The knee should be visible server-side (waiting > 0, cache % pinned) before it's visible client-side; catching that ordering is the learning
- [ ] Chart 1: aggregate throughput vs concurrency, predicted curve from prediction 2 overlaid, both knees marked
- [ ] Chart 2: TTFT p50/p99 vs concurrency — the latency price of the throughput
- [ ] Little's law check at every point (a one-line calc per row)
- [ ] Goodput row: max throughput subject to TTFT < 500 ms and TPOT < 50 ms — the SLO-constrained number a provider could actually sell (DistServe's framing). Compare it to raw max throughput; the gap is the marketing

## Measurements

| Concurrency | Agg tok/s | Per-req tok/s | TTFT p50/p99 (ms) | TPOT p50/p99 (ms) | Running/waiting | KV cache % | Little's law ✓? |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 4 | | | | | | | |
| 8 | | | | | | | |
| 16 | | | | | | | |
| 32 | | | | | | | |
| 64 | | | | | | | |
| 128 | | | | | | | |

- Predicted knee: ___ · Measured knee: ___ · % error: ___
- Predicted `num_gpu_blocks`: ___ · Log says: ___
- Clean batch-1: ___ tok/s = ___% of 118.12 ceiling (Day 2 got 63.5% with the crippled sampler)
- Goodput (TTFT<500ms, TPOT<50ms): ___ tok/s vs raw max ___ tok/s

## If time remains

- **8k-context arm:** rerun a mini-sweep at 8k context. KV traffic per step is 8×, so the model in prediction 2 says the curve flattens earlier — one chart that proves the KV term is real, not decorative
- **Preemption hunt:** push past the block budget from prediction 3's corollary, watch the preemption counter tick, and note whether vLLM recomputes or swaps — feel what "KV cache pressure" actually does to a request
- **Chunked prefill dial:** toggle/tune `max_num_batched_tokens`, re-check p99 TTFT at high concurrency — the Sarathi-Serve idea, felt before reading the paper (Arc 4.4)
- **Open vs closed loop:** rerun one mid-level point in request-rate mode; same nominal load, different queue behavior — which one resembles production traffic?

## Post skeleton

- Hook: "One GPU serves 1 user at 118 tok/s — or 64 users at ___ tok/s *each*. The chip's spec sheet predicted the crossover before I ran anything."
- The chart: throughput vs concurrency, predicted curve overlaid, knee marked.
- The number: predicted vs measured knee, % error.
- What surprised me: ____
- Tomorrow: quantize the checkpoint myself — fp8 halves the bytes every decode step reads, and today's roofline says exactly what that should buy at every batch size. Prediction before download, as always.
