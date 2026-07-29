# Day 2 Writeup — The plumbing has its own roofline

*Polished narrative for [[Day 2 — Rent a GPU, Serve an Endpoint]] (Unit 1.2, [[Curriculum]]). Follows [[Day 1 — Writeup]].*

**The claim:** Day 1's formula (bandwidth ÷ model bytes) should still predict decode speed on a rented datacenter GPU — 8x the Mac's bandwidth, same physics. It did, roughly. What it didn't predict was everything *around* the number: a cold start is its own set of rooflines, and most of them aren't physics, they're software.

## Setup

- **A100 80GB PCIe**, Prime Intellect, $1.20/hr on-demand. (4090 was the original plan — unavailable that day; grabbed whatever was on-demand with good availability instead.) 1,935 GB/s memory bandwidth, 312 TFLOPS bf16 dense tensor-core throughput.
- **Qwen3-8B, bf16 safetensors** (not the Q4 GGUF from Day 1) — vLLM's default, HF-native sharded weights. Exact size read straight from the safetensors header metadata (`total_size`): **16,381,470,720 bytes = 16.38 GB** — no ambiguity about what counts as a "parameter" the way GGUF's block-quantization overhead created on Day 1.
- **vLLM**, OpenAI-compatible server. Same reason as always for choosing it over llama.cpp here: serving-first design (continuous batching, PagedAttention, native OpenAI routes), CUDA-first.

## The math

**Decode (bandwidth roofline) — same formula, new silicon:**

> decode ceiling = bandwidth ÷ weight bytes = 1,935 GB/s ÷ 16.38 GB = **118.12 tok/s**

**Prefill (compute roofline):**

> prefill ceiling = compute ÷ FLOPs-per-token = 312×10¹² ÷ (2 × 8.19×10⁹) = **19,047.6 tok/s**

The ratio matters more than either number alone: A100 prefill/decode ≈ **161×**, versus the M4's ≈ **11×** (Day 1: 228.9/20.8). The A100 has vastly more compute relative to its own memory bandwidth than the Mac does — its "balance point" (FLOPs it can do per byte moved) is much further out. Batch-1 decode can't touch that compute at all; it's purely bandwidth-bound. So the more lopsided a chip's compute-to-bandwidth ratio, the *more* of its capability sits idle at batch 1. Datacenter GPUs are built assuming you batch; running them one request at a time is close to adversarial to the hardware.

## Predictions vs measurements

Predictions written before running (see [[Day 2 — Rent a GPU, Serve an Endpoint]] for the untouched originals):

| Quantity | Predicted | Measured | Verdict |
|---|---|---|---|
| Decode ceiling | — | 118.12 tok/s | (ceiling itself, not a prediction to score) |
| Decode efficiency | 80% (reasoning: bigger/non-unified memory subsystem costs some, CUDA-graph capture claws some back) | **63.5%** (75.0 / 118.12) | ❌ miss, and lower than Day 1's 87% Mac number — confounded, see below |
| pip install vllm | 50s | 186.566s | ❌ 73% under, missed 3.7× |
| HF weight download | 640s (from a single-stream speedtest) | 46.966s | ❌ 13.6× overestimate |
| Engine init (load + CUDA graphs) | 20s | 51.299s | ❌ 61% under, missed 2.6× |
| TTFT (laptop, tunneled) | 140ms | 95ms | ❌ 47% overestimate |

![[xdoi-day2-coldstart-decode.png]]

**Four misses, four different mechanisms — that's the actual finding.** Not one wrong intuition repeated, but four distinct failure modes:

1. **pip install** — heavy CUDA/PyTorch dependency trees are bigger than "a pip install" sounds like. `user`+`sys` time (161.7s) was still less than `real` (186.6s) — the ~25s gap was I/O wait, not CPU, a small preview of "not all wall-clock time is compute."
2. **HF download** — the single biggest miss, in the *safe* direction (overestimated 13.6×), and the most instructive. The speedtest used one TCP stream to an unrelated public test server (thinkbroadband, UK). The real download used Hugging Face's Xet backend — content-addressed chunks fetched in parallel from a well-peered CDN (365 MB/s → 1.18 GB/s reported mid-download). A single-stream benchmark to the wrong server told us almost nothing about the path that mattered. This is Day 1's "achievable bandwidth depends on who's asking" lesson wearing a networking costume.
3. **Engine init** — underestimated. Unlike the roofline numbers, there's no closed-form formula for CUDA graph capture + kernel warmup time; it's a software/engineering cost (how many batch-size buckets get captured, how the warmup loop is written), not a physical one. Worth remembering: some cold-start stages are physics, some are pure engineering overhead, and only the first kind is derivable from a datasheet.
4. **TTFT** — right about the mechanism (dominated by tunnel/network round-trip, not compute), wrong about the magnitude. Both the prefill (~20 tokens ÷ ~19,000 ceiling ≈ 1ms) and one decode step (~8.5ms) are single-digit milliseconds — the other ~85ms was pure network.

**The efficiency number has an asterisk, and the asterisk is real content.** This box shipped with an NVIDIA driver but no CUDA toolkit — no `nvcc`. vLLM's fastest sampling path (a fused FlashInfer kernel) tries to JIT-compile on first use; without a compiler, the engine crashed outright the first time. The fix (`VLLM_USE_FLASHINFER_SAMPLER=0`) unblocked serving but forces a slower fallback sampler for every single decode step — so part of the 63.5% (vs. the Mac's 87%) is a genuine environment gap, not a clean statement about A100-vs-M4 hardware efficiency. Chased the "proper" fix afterward: installed the matching CUDA 13.0 toolkit (confirmed against `torch.version.cuda`), fought a `PATH` conflict with Ubuntu's own pre-baked (and mismatched) `nvidia-cuda-toolkit`, got `nvcc` itself working — and then hit a third wall, `nvcc fatal: Failed to preprocess host compiler properties`, likely a `/tmp noexec`-class container restriction. Called it there. **A confounded number with a clearly-explained cause is better content than a clean one** — "here's what an infra gap actually costs in tok/s" is a real, tellable story.

## Caveats collected along the way

- **Driver ≠ toolkit.** `nvidia-smi` reporting "CUDA Version: 13.0" only tells you the *driver's* max-supported version — it says nothing about whether `nvcc` (the compiler) is even installed. Cloud GPU images often ship only the former.
- **TFLOPS is 10¹², not 10⁹** — dividing raw FLOPS by GFLOPs/token without matching powers of ten silently produces an answer ~1000× too small. Always convert to the same order of magnitude before dividing.
- **"Sparsity" numbers on a datasheet are conditional.** NVIDIA lists dense vs. 2:4-structured-sparsity throughput; the sparse number only applies to weights actually pruned into that pattern via NVIDIA's toolkit. A stock checkpoint is fully dense — use the dense number.
- **`real` / `user` / `sys` from `time` mean different things.** `real` = wall clock (what a user waits); `user` = CPU time in your own code; `sys` = CPU time in the kernel on your behalf. `user+sys < real` means time was spent waiting (I/O), not computing.
- **A generic speedtest measures the wrong path.** Testing bandwidth to an unrelated server (even a reputable one) doesn't predict throughput to the server you actually care about — different peering, different protocol (HF's chunked/parallel Xet fetch vs. a plain single-stream GET). Test the actual mechanism when you can.
- **`nvidia-smi` showing ~90% VRAM used isn't a leak.** vLLM's PagedAttention pre-allocates most of VRAM as its KV-cache block pool at startup — a deliberate reservation, not a runaway process. (Confirmed idle baseline at a clean 0 MiB before vLLM ever launched; didn't get to capture the "looks full" moment live this session — queued for Day 3.)
- **SSH host-key warnings on a fresh cloud instance are usually IP reuse, not a real MITM.** Every new VM gets a fresh host key; providers commonly recycle IPs across ephemeral instances. `ssh-keygen -R <ip>` and reconnect — but actually look at *why* the key changed before assuming it's benign.
- **Non-default SSH keys need `-i` (or a config entry) — they aren't tried automatically.** OpenSSH only auto-offers a fixed set of default-named identity files in `~/.ssh/`; a differently-named or differently-located key (e.g. a marketplace-provided keypair) is silently never offered unless pointed to explicitly.

## Cold-start stage breakdown

The waterfall (chart above, left panel): pip install (186.6s) → weight download (47.0s) → engine init (51.3s) → +95ms TTFT. **Total, nothing → serving: 284.9s ≈ 4.75 minutes** — excluding provisioning, whose clean timestamp got lost in an SSH-key detour (own the miss: this is exactly the "measurement hygiene" lesson from Day 1, just applied to logistics instead of a benchmark).

Every one of these stages is something production serving infra spends real engineering effort eliminating: baked container images skip the pip install, pre-staged weights on fast storage (or literal memory snapshotting) skip the download, warm pools skip engine init entirely. Today's cold start is the bill that always-on or aggressively-optimized serverless infra pays to make invisible.

## X post — final draft

> Day 2/45 of Inference Engineering: Rent a GPU, Serve an Endpoint
>
> Day 1 was the physics (spec sheet → predicted decode speed, 87% of ceiling). Day 2 was the plumbing — and the plumbing is where the time actually goes.
>
> Rented an A100 80GB (Prime Intellect, $1.20/hr), served Qwen3-8B bf16 with vLLM, and predicted every stage of the cold start before running it — same rule as Day 1.
>
> The physics still held: 1,935 GB/s ÷ 16.38 GB of weights = 118.12 tok/s predicted ceiling. Measured 75.0 — 63.5% of it (with an asterisk below).
>
> The plumbing did not go how I predicted, in four completely different ways:
>
> • pip install vllm: guessed 50s, took 186.6s — a CUDA/PyTorch dependency tree is heavier than "a pip install" sounds.
>
> • HF weight download: guessed 640s from a speedtest, took 47.0s — 13.6x faster than predicted, because I tested bandwidth to a random server instead of Hugging Face's actual (parallel, chunked) CDN path. Wrong proxy entirely.
>
> • Engine init (CUDA graph capture): guessed 20s, took 51.3s — this one isn't physics, it's software overhead, and there's no datasheet for that.
>
> • TTFT: guessed 140ms, measured 95ms — right about the cause (almost all network, prefill+decode are single-digit ms), wrong about the magnitude.
>
> The asterisk on 63.5%: this box had no CUDA compiler, so vLLM's fastest sampling kernel couldn't compile and fell back to a slower path. Spent a while trying to fix it properly (installed the toolkit, fought a PATH conflict, hit a third wall) before calling it — an honest infra gap, not a hidden mistake.
>
> What surprised me: the A100 is *less* efficient than my Mac at batch 1 (63.5% vs 87%), which is the opposite of "bigger GPU wins." Datacenter chips have a way more lopsided compute-to-bandwidth ratio than the M4 (161x prefill/decode gap vs 11x) — which means batch-1 decode wastes even more of a datacenter GPU than it wastes a laptop chip. They're built assuming you'll batch requests together.
>
> Tomorrow: concurrency — watching continuous batching multiply throughput without multiplying hardware.
>
> [attach: xdoi-day2-coldstart-decode.png]

## What surprised me — candidates

- The A100 being *less* batch-1-efficient than the M4 — the "bigger hardware should just win" intuition failing in a derivable, not folklore, way (the 161x vs 11x compute/bandwidth ratio).
- The download prediction missing by 13.6x in the *safe* direction — proof that a speedtest to the wrong server is worse than no speedtest, because it's confidently wrong.
- Losing a clean provisioning timestamp to an SSH-key mistake — the plumbing's own version of Day 1's "measurement hygiene" lesson.
- Chasing the fused-sampler fix three layers deep (missing nvcc → toolkit PATH conflict → host-compiler/tmp issue) and calling it — deciding when a confound is good enough content vs. worth fully resolving.
- Four cold-start misses, four unrelated causes (dependency weight, wrong benchmark proxy, no-formula software cost, network-dominated latency) — nothing here was "the same mistake twice."
