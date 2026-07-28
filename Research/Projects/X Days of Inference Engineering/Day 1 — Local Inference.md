# Day 1 — Local Inference

*Planned for Friday 2026-07-24. Unit 1.1 from [[Curriculum]].*

**The question:** decode is supposedly memory-bandwidth-bound ([[Inference]], [[Transformer Inference Basics]]). Can my laptop's memory bus predict its token rate before I run anything?

## The math (fill in predictions BEFORE running)

Every decode step reads the entire model once (plus the KV cache), so the ceiling is:

> predicted decode tok/s ≈ memory bandwidth ÷ bytes read per token ≈ bandwidth ÷ model file size (at short context)

Hardware: Apple M4, 24 GB unified memory, **120 GB/s** bandwidth (Apple spec). CPU and GPU (Metal) share the same bus, so the prediction holds either way.

- Model file size (GB): 5GB. 
	- Qwen 8B has around ~8.2B Parameters. 
	- Q4_K_M is ~4.2 bits/weight. 
	- So 8.2B × 0.6 Bytes = ~5GB. 
- Predicted decode ceiling (tok/s): 120 GB/s ÷ 5GB = 24 Tok/s
- Predicted prefill (should be much faster — compute-bound, parallel over the prompt): guess: 10-20x faster than decode, so lets just grab an average
	- 360 tok/s
- Prediction: decode tok/s at 8k context vs 0 context — how much slower and why (KV cache adds bytes per token): ==---

## Steps

- [ ] `brew install llama.cpp`
- [ ] Grab a Q4 8B model — llama.cpp pulls straight from HF: `llama-bench -hf ggml-org/Qwen3-8B-GGUF:Q4_K_M` (or `hf download` first and pass `-m`)
- [ ] Baseline bench: `llama-bench -m <model> -p 512 -n 128` → note `pp` (prefill tok/s) and `tg` (decode tok/s)
- [ ] Context depth sweep: `llama-bench -m <model> -n 128 -d 0,2048,4096,8192` — watch decode degrade as the KV cache adds bytes per token
- [ ] Chat with it once (`llama-cli` or `llama-server`) — feel TTFT vs stream rate; this is the thing the numbers describe
- [ ] Compare predicted vs measured; compute the % error

## Measurements

| Run                  | Prefill tok/s | Decode tok/s | Notes |
| -------------------- | ------------- | ------------ | ----- |
| baseline (p512/n128) | 228.94 ± 0.10 | 20.80 ± 0.05 |       |
| d=2048               |               |              |       |
| d=4096               |               |              |       |
| d=8192               |               |              |       |

Efficiency: measured decode ÷ predicted ceiling = ____ % of the theoretical bus limit.

## If time remains

- Same bench on a Q8 quant — halves-the-bytes logic in reverse: does decode tok/s halve?
- CPU-only run (`-ngl 0`): same memory bus, no Metal — does decode barely change while prefill craters?

## Post skeleton

- Hook: "Before running a single token, my laptop's spec sheet predicted its LLM speed. Here's the delta."
- The one chart: decode tok/s across context depths, with the predicted ceiling as a horizontal line.
- The formula, in one sentence anyone can reuse for their own machine.
- What surprised me: ____
- Tomorrow: renting a real GPU and standing up my own endpoint.
