# Day 2 — Rent a GPU, Serve an Endpoint

*Unit from Arc 1 of [[Curriculum]]. Follows [[Day 1 — Local Inference]].*

**The question:** what does it actually take — in minutes and dollars — to go from nothing to my own OpenAI-compatible endpoint on a rented GPU? And does Day 1's formula (bandwidth ÷ model bytes) still predict decode speed when the hardware is 8x faster?

Day 1 was the physics; Day 2 is the plumbing. The plumbing is where the time goes, so today the *cold start itself* is a measurement, not an obstacle.

## The math (fill in predictions BEFORE running)

Same formula, new silicon. Memory bandwidth of common rentals: RTX 4090 **1,008 GB/s**, L40S **864**, A100 80GB **2,039**, H100 SXM **3,350**.

Serving Qwen3-8B in bf16 this time (vLLM default) — ~16 GB of weights, so it needs a 24 GB card minimum.

- GPU chosen + $/hr: ____
- Predicted batch-1 decode ceiling (bandwidth ÷ ~16.4 GB): ____ tok/s
- Predicted total wall-clock, nothing → first token (be honest, then compare): ____ min
- Predicted slowest stage of the cold start (guess before timing: weight download? pip install? engine init?): ____
- Predicted total spend today: $____

## Steps

Marketplace: Vast.ai or RunPod ([[GPU Rental Markets]]). RunPod is the least-friction first time: PyTorch template, expose HTTP port 8000, add a persistent volume so tomorrow starts warm.

Time every stage — this table is today's chart:

- [ ] **T0** Rent the instance (note provisioning wait)
- [ ] **T1** SSH in; `pip install vllm` (or use a vLLM docker image and note the pull time instead)
- [ ] **T2** `vllm serve Qwen/Qwen3-8B --max-model-len 16384` — the HF weight download (~16 GB) and engine init (CUDA graph capture, KV allocation) log themselves; keep the timestamps
- [ ] **T3** First token from the laptop:
  ```python
  from openai import OpenAI
  client = OpenAI(base_url="http://<POD_IP>:8000/v1", api_key="x")
  # stream a completion; note TTFT by eye, then check vllm's logged metrics
  ```
- [ ] Batch-1 decode rate: one long streamed generation, tok/s from the vLLM log (or `vllm bench serve --num-prompts 1`) → compare against the predicted ceiling
- [ ] Restart `vllm serve` with weights already on the volume → **warm** start time vs cold
- [ ] Save the template/image + volume; write down the exact resurrection steps for Day 3

## Measurements

| Stage | Time | Notes |
|---|---|---|
| Provisioning | | |
| Env setup / image pull | | |
| Weight download | | GB/s achieved vs advertised network |
| Engine init (load → CUDA graphs → ready) | | |
| First token from laptop | | |
| **Total cold, nothing → token** | | |
| Warm restart (weights on volume) | | |

- Batch-1 decode: measured ____ tok/s vs predicted ____ → ____ % of ceiling (Day 1 hit 87% — does a datacenter card do better or worse at batch 1?)
- Total spent: $____

## If time remains

- One `watch nvidia-smi` while generating: note how much of the 24/80 GB is KV cache pre-allocation, not weights — vLLM grabs ~90% by default. First taste of PagedAttention's job (unit 4.2).
- Hit the endpoint with 4 concurrent requests — does per-request tok/s drop 4x? (It shouldn't. That gap is Day 3's whole story.)
- Skim the vLLM startup log line by line — it narrates the whole [[Serving Models]] stack: config → weights → graphs → scheduler → server.

## Post skeleton

- Hook: "Yesterday my MacBook's spec sheet predicted its LLM speed to 87%. Today I rented a GPU 8x its bandwidth for $__/hr — same formula, and a cold-start stopwatch."
- The chart: cold-start waterfall (stacked bar of the stages) — nobody posts this; everyone who's tried it will feel it.
- The number: predicted vs measured decode tok/s on datacenter silicon, next to the Mac's.
- What surprised me: ____
- Tomorrow: concurrency sweep — watching continuous batching multiply throughput without multiplying hardware.
