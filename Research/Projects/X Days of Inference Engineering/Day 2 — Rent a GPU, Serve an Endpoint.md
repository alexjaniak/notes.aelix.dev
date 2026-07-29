# Day 2 — Rent a GPU, Serve an Endpoint

*Unit from Arc 1 of [[Curriculum]]. Follows [[Day 1 — Local Inference]].*

**The question:** what does it actually take — in minutes and dollars — to go from nothing to my own OpenAI-compatible endpoint on a rented GPU? And does Day 1's formula (bandwidth ÷ model bytes) still predict decode speed when the hardware is 8x faster?

Day 1 was the physics; Day 2 is the plumbing. The plumbing is where the time goes, so today the *cold start itself* is a measurement, not an obstacle.

## The math (fill in predictions BEFORE running)

Same formula, new silicon. Memory bandwidth of common rentals: RTX 4090 **1,008 GB/s**, L40S **864**, A100 80GB **2,039**, H100 SXM **3,350**.

Serving Qwen3-8B in bf16 this time (vLLM default) — ~16 GB of weights, so it needs a 24 GB card minimum.

- GPU chosen + $/hr: **A100 80GB PCIe, Prime Intellect, $1.20/hr on-demand** (4090 unavailable that day; scarcity meant "grab whatever's available," not the 4090 in the original plan)
- Predicted batch-1 decode ceiling (1,935 GB/s PCIe ÷ 16.38 GB, exact bf16 weight bytes from the safetensors header): **118.12 tok/s**
- Predicted total wall-clock, nothing → first token: guessed per-stage, not as one number (see table) — sum of clean stages came to 284.9s ≈ 4.75 min
- Predicted slowest stage of the cold start: guessed **weight download** (640s, from a single-stream speedtest) — actually the *fastest* clean stage at 47.0s; **pip install** (guessed 50s) was the real slowest at 186.6s
- Predicted total spend today: not fixed in advance; actual GPU-hour cost was small (~$1.20/hr × well under 2 hrs of actual billed time)

## Steps

Marketplace: Vast.ai or RunPod ([[GPU Rental Markets]]). RunPod is the least-friction first time: PyTorch template, expose HTTP port 8000, add a persistent volume so tomorrow starts warm.

Time every stage — this table is today's chart:

- [x] **T0** Rent the instance (note provisioning wait) — rented, but the clean timestamp got lost in an SSH-key detour
- [x] **T1** SSH in; `pip install vllm` — 186.566s
- [x] **T2** `vllm serve Qwen/Qwen3-8B --max-model-len 16384` — download 46.966s, engine init 51.299s (after routing around a missing-`nvcc` crash with `VLLM_USE_FLASHINFER_SAMPLER=0`)
- [x] **T3** First token from the laptop — 95ms TTFT, via curl through an SSH tunnel (`ssh -L 8000:localhost:8000`, since this instance was SSH-only, no exposed HTTP proxy)
- [x] Batch-1 decode rate: 75.0 tok/s from vLLM's own periodic log line, one long completion (`max_tokens: 1000`) — 63.5% of ceiling, confounded by the disabled fused sampler
- [ ] Restart `vllm serve` with weights already on the volume → **warm** start time vs cold — no separate persistent volume was configured on Prime Intellect this time; the timed rerun (51.3s) is a reasonable stand-in but isn't a true separate-volume warm start. **Deferred to Day 3**: check what Prime Intellect actually offers for persistence before relying on it.
- [ ] Save the template/image + volume; write down the exact resurrection steps for Day 3 — **not done**, same reason. Chased the fused-sampler/CUDA-toolkit fix instead of this; worth doing first thing next time so Day 3 doesn't restart from zero.

## Measurements

| Stage | Time | Notes |
|---|---|---|
| Provisioning | not cleanly captured | SSH-key mismatch (wrong key pasted, then key file location confusion) ate the clock before a clean T0→SSH timestamp was taken |
| Env setup (`pip install vllm`) | **186.566s** (real) | predicted 50s — 73% under, missed 3.7×. Heavy CUDA/PyTorch dependency tree; user+sys (161.7s) < real (186.6s), ~25s was I/O wait, not CPU |
| Weight download (`hf download`) | **46.966s** (real) | predicted 640s from a single-stream speedtest (25.6 MB/s) — actual was 13.6× faster. HF's Xet backend does parallel chunked fetch (365 MB/s → 1.18 GB/s reported) from a well-peered CDN; a single curl stream to an unrelated server (thinkbroadband, UK) was the wrong proxy entirely |
| Engine init (load → CUDA graphs → ready) | **51.299s** (real) | predicted 20s — 61% under. First attempt crashed (`nvcc` missing — FlashInfer's fused sampler tried to JIT-compile and failed); fixed by `VLLM_USE_FLASHINFER_SAMPLER=0`, timed on the clean rerun |
| First token from laptop (TTFT, tunneled) | **95ms** | predicted 140ms — 47% over. Reasoning ("almost entirely network") was right in kind; prefill (~20 tok) + one decode step are both sub-10ms next to the SSH-tunnel round trip |
| **Total cold, nothing → token** (excl. provisioning) | **284.9s ≈ 4.75 min** | |
| Warm restart (weights + deps already present) | **51.3s** | this *is* the engine-init number above — no separate volume was configured, but the second timed `vllm serve` run had weights on local disk and vllm already installed, which is functionally the warm case |

- Batch-1 decode: measured **75.0 tok/s** vs predicted 118.12 ceiling → **63.5% of ceiling** (vs Day 1's 87% on the Mac — the datacenter card was *less* efficient at batch 1, not more, and it's largely explained below, not just hand-waved)
  - **Confound, not a clean read:** this box had no CUDA toolkit (`nvcc`), so vLLM's fused FlashInfer sampling kernel couldn't JIT-compile and the run used the slower fallback sampler. Chased a fix (installed `cuda-toolkit-13-0` from NVIDIA's apt repo, fixed a PATH conflict with Ubuntu's own older `nvidia-cuda-toolkit`) but hit a third wall — `nvcc fatal: Failed to preprocess host compiler properties`, likely a `/tmp noexec`-class issue — and called it there rather than burn the rest of the session on infra. The 63.5% stands with that asterisk attached.
- Total spent: modest — single A100 80GB on-demand at $1.20/hr, well under 2 hours of actual pod time including the troubleshooting detours

## If time remains

- [ ] **Not done this session** — One `watch nvidia-smi` while generating: note how much of the 80GB is KV cache pre-allocation, not weights — vLLM grabs ~90% by default. Session did confirm the mechanism verbally (idle baseline was a clean 0 MiB before vLLM ever ran) but never captured the "looks full" moment live. Easy quick win for Day 3's opener.
- [ ] Hit the endpoint with 4 concurrent requests — does per-request tok/s drop 4x? (It shouldn't. That gap is Day 3's whole story — deliberately deferred, not skipped.)
- [ ] Skim the vLLM startup log line by line — it narrates the whole [[Serving Models]] stack: config → weights → graphs → scheduler → server.

## Post skeleton

- Hook: "Yesterday my MacBook's spec sheet predicted its LLM speed to 87%. Today I rented a GPU 8x its bandwidth for $__/hr — same formula, and a cold-start stopwatch."
- The chart: cold-start waterfall (stacked bar of the stages) — nobody posts this; everyone who's tried it will feel it.
- The number: predicted vs measured decode tok/s on datacenter silicon, next to the Mac's.
- What surprised me: ____
- Tomorrow: concurrency sweep — watching continuous batching multiply throughput without multiplying hardware.
