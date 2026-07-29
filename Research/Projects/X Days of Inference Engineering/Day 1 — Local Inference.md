Unit 1.1 from [[Curriculum]].*

**The question:** decode is supposedly memory-bandwidth-bound ([[Inference]], [[Transformer Inference Basics]]). Can my laptop's memory bus predict its token rate before I run anything?

## The math (fill in predictions BEFORE running)

Every decode step reads the entire model once (plus the KV cache), so the ceiling is:

> predicted decode tok/s ≈ memory bandwidth ÷ bytes read per token ≈ bandwidth ÷ model file size (at short context)

Hardware: Apple M4, 24 GB unified memory, **120 GB/** bandwidth (Apple spec). CPU and GPU (Metal) share the same bus, so the prediction holds either way.

- Model file size (GB): 5GB. 
	- Qwen 8B has around ~8.2B Parameters. 
	- Q4_K_M is ~4.2 bits/weight. 
	- So 8.2B × 0.6 Bytes = ~5GB. 
- Predicted decode ceiling (tok/s): 120 GB/s ÷ 5GB = 24 Tok/s
- Predicted prefill (should be much faster — compute-bound, parallel over the prompt): guess: 10-20x faster than decode, so lets just grab an average
	- 360 tok/s
- Prediction: decode tok/s at 8k context vs 0 context — how much slower and why (KV cache adds bytes per token): KV = 2 × 36 layers × 8 KV heads × 128 dim × 2 bytes (fp16) = **144 KB/token** → +1.18 GB at 8k → bytes go 5.03 → 6.24 GB → predict **~81% of baseline** (~16 tok/s)

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
| baseline (p512/n128) | 228.94 ± 0.10 | 20.80 ± 0.05 | quiet machine; ceiling 23.9 → **87%** |
| d=2048               | 195.62 ± 0.65 | 19.20 ± 0.28 | sweep-run baseline was 19.82 ± 0.38 (background load) |
| d=4096               | 179.63 ± 0.72 | 17.99 ± 0.26 | predicted (re-anchored) 17.7 → 1.6% err |
| d=8192               | 154.57 ± 0.84 | 15.97 ± 0.06 | predicted 16.05 → **0.5% err**; 80.6% of baseline vs 81% predicted |
| Q8_0 (p512/n128)     | 226.70 ± 7.47 | 12.52 ± 0.08 | 8.71 GB file; ÷1.66 vs file-ratio 1.73; **91%** of its 13.8 ceiling |
| CPU only (-ngl 0)    | 22.15 ± 1.74  | 12.69 ± 1.36 | prefill ÷10 (compute gone); decode ÷1.6 → CPU pulls only ~64 GB/s of the bus |

Efficiency: measured decode ÷ predicted ceiling = 20.80 ÷ 23.9 = **87%** of the theoretical bus limit. (Prefill: 228.94 ÷ ~270 compute ceiling = 85% — same story, other roofline.) Full narrative + caveats: [[Day 1 — Writeup]].

## If time remains

- Same bench on a Q8 quant — halves-the-bytes logic in reverse: does decode tok/s halve?
- CPU-only run (`-ngl 0`): same memory bus, no Metal — does decode barely change while prefill craters?

## Post skeleton

- Hook: "Before running a single token, my laptop's spec sheet predicted its LLM speed. Here's the delta."
- The one chart: decode tok/s across context depths, with the predicted ceiling as a horizontal line.
- The formula, in one sentence anyone can reuse for their own machine.
- What surprised me: ____
- Tomorrow: renting a real GPU and standing up my own endpoint.
