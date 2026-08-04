# Day 6 — Quantize It Yourself

*Unit 1.4 from Arc 1 of [[Curriculum]] (pushed Day 4 → 5 → today). Follows [[Day 4 — Writeup]] ("serving is not decode") and [[Day 5 — Writeup]] (counting dots). Readings paired per section below; the anchors are the JAX book's [inference chapter](https://jax-ml.github.io/scaling-book/inference/) (its bytes-per-param arithmetic is Day 1/5's roofline) and Maarten Grootendorst's [Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization). This day previews Arc 5 — units 5.1–5.3 do the deep version later.*

**The question:** Day 4's throughput model — f(B) = 2,039·B ÷ (16.38 + 0.147·B) — says batch-1 decode is 99% weight streaming. Quantization attacks exactly that term: INT4 shrinks the 16.38 GB weight read ~4×. So the roofline predicts batch-1 decode should roughly triple… and *also* predicts the gain evaporates at high concurrency, because the 0.147·B KV term and Day 4's unbudgeted prefill tax don't shrink by a single byte. Can I quantize the checkpoint myself — twice, two different formats — and watch both predictions land on the same chart?

## The mechanism (~40 min reading, before touching a GPU)

Every block below names its reading and ends in a question to answer in my own words at the top of Measurements. The point is to know what the tool is doing before running it — llm-compressor is three lines of Python; the three lines are not the skill.

**1. What a quantized tensor literally is.** *Read: [Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization), first half.* Storage is `q = clamp(round(w / s))` with a shared scale `s` (absmax of the group ÷ max representable), reconstruction is `w ≈ s·q`. The entire design space is *who shares a scale*: per-tensor → per-output-channel → per-group-of-128-weights-along-the-input-dim. This answers the open [[GPUs]] question about what "group scaling factors" are — a W4A16-g128 checkpoint is int4 codes plus one bf16 scale per 128 weights (~3% size overhead).
→ *Q1: one outlier weight lands in a group — what happens to the effective bits of the other 127? Why does shrinking the group from "whole tensor" to 128 contain the damage?*

**2. INT4 vs FP8 are different kinds of grid.** *Read: FP8 formats paper, [arXiv 2209.05433](https://arxiv.org/abs/2209.05433) §2–3 (skim).* INT4 = 16 *uniform* levels stretched by a scale. FP8 E4M3 = sign + 4 exponent + 3 mantissa bits — a *logarithmic-ish* grid, dense near zero, max value 448 (its sibling E5M2 trades mantissa for range; training gradients want range, inference weights want mantissa).
→ *Q2: why does FP8 weight quantization typically need no calibration data and no groups (per-tensor/per-channel scale suffices), while INT4 needs both? Hint: which grid absorbs outliers in its exponent?*

**3. Why outliers exist at all.** *Read: LLM.int8, [arXiv 2208.07339](https://arxiv.org/abs/2208.07339) — §3 + Fig. 3; skim the rest.* Past ~6B params, transformers develop *emergent outlier feature channels* — activation magnitudes ~100× the median, concentrated in a few hidden dims, load-bearing for quality. This is the empirical fact the whole field is downstream of: it's why naive per-tensor W8A8 broke, why groups exist (block 1), and why AWQ exists (block 4).
→ *Q3: outliers live in the __activations__ — so why do they force decisions about how the __weights__ are quantized?*

**4. RTN vs GPTQ vs AWQ — three answers to "round to what."** *Read: [AWQ, arXiv 2306.00978](https://arxiv.org/abs/2306.00978) §3 (the readable one); [GPTQ, arXiv 2210.17323](https://arxiv.org/abs/2210.17323) abstract + §1 only (the math is unit 5.x material).*
- **RTN** (round-to-nearest): each weight independently to its nearest grid point. The baseline.
- **GPTQ**: minimizes *layer output* error, not weight error — quantize column-by-column, and after each column push its rounding error onto the not-yet-quantized columns (Hessian-weighted, from calibration activations). Compensation.
- **AWQ**: no compensation — *protection*. Observation: ~1% of weight channels are salient (judged by **activation** magnitude, not weight magnitude — that's the "activation-aware"). Scale those channels up before rounding so they get finer effective resolution; fold the inverse scale into the preceding op. Mathematically free, and it needs no backprop-era machinery.
→ *Q4: both GPTQ and AWQ consume calibration data — what different thing does each use it FOR?*

**5. W4A16 vs W8A8 — where the speedup actually comes from.** *Read: [vLLM quantization docs](https://docs.vllm.ai/en/latest/features/quantization/) (the supported-hardware matrix) + JAX book [inference chapter](https://jax-ml.github.io/scaling-book/inference/) quantization discussion.* The letters: W = weight bits, A = activation bits.
- **W4A16 (today's INT4 arm):** weights *stored* int4, *dequantized to bf16 inside the kernel*, GEMM runs on the same bf16 tensor cores as ever. Pure memory-bandwidth win; zero compute win. Marlin's entire reason to exist is doing that dequant at full memory bandwidth, fused into the GEMM.
- **W8A8 (fp8 or int8):** activations quantized too, on the fly (per-token dynamic scales), GEMM runs on 8-bit tensor cores → the *compute* also gets ~2× faster.
In [[Day 5 — Writeup]] language: weight-only quant helps the intensity-≈1 row of the table (decode, matvec); activation quant is the only kind that helps the intensity-≈1000 work (prefill). "Serving is not decode" therefore has a quantization corollary: **weight-only quantization is a decode drug, and Day 4 proved my endpoint is only half decode.**
→ *Q5: state, before measuring, which arm should help TTFT (a prefill metric) and which shouldn't, and why.*

**6. Hardware reality check — the roulette matters today.** *Read: same vLLM docs page, "FP8" section.* FP8 tensor cores require compute capability ≥ 8.9 (Hopper/Ada). **A100 is 8.0.** vLLM will happily serve an FP8 checkpoint on an A100 — via Marlin, as *weight-only* (dequant to bf16 for compute). So if the rental roulette hands me the usual A100: both of today's arms are weight-only decode drugs, and "FP8 = 2× compute" is a claim I *cannot* test today (H100 day, later — noted in "If time remains").
→ *Q6: given that, predict whether the FP8 arm speeds up the C=64 point on an A100 at all, and via which term of f(B).*

## Debts from Day 4 (bench hygiene — these bit me once already)

- [ ] **Prefix cache contamination:** every benchmark server today runs `--no-enable-prefix-caching` (or per-point `--seed`). Day 4's audit showed 50–66% free prefills mid-sweep; today's cross-*variant* comparison dies instantly if one arm gets cache help.
- [ ] **Fused sampler check per server:** bench requests set `top_p`/`top_k`, never `seed`; grep each startup log for the FlashInfer sampler line. Three servers today = three chances to silently measure the fallback.
- [ ] **Toolchain checklist** (Day 4's six seams): `nvcc --version`, `gcc -print-prog-name=cc1plus` returns a full path, torch CUDA major matches. Before anything else.
- [ ] Open-loop control rerun: still owed, still deferred — today is already three servers deep.

## The math (fill in predictions BEFORE running)

Anchors from Day 4, same box class assumed (A100-SXM4, **2,039 GB/s**; if roulette hands PCIe, re-anchor ×1935/2039): Qwen3-8B bf16 **16.38 GB**, KV **147 KB/token**, measured clean batch-1 **88.4 tok/s = 11.31 ms/step** vs 8.10 ideal → **~3.2 ms/step engine overhead**, C=64 measured **2,505 agg**.

**1. Checkpoint size ≠ params × bits ÷ 8.** The recipe `ignore=["lm_head"]` keeps the LM head bf16, and embeddings aren't Linear modules — Qwen3-8B's vocab is 151,936 × 4,096 hidden ≈ 0.62B params *each*, untied. So: quantizable Linear params ≈ 8.19B − ___ B; predicted disk size FP8 ≈ ___ GB, INT4-g128 (+~3% scales) ≈ ___ GB. `du -sh` scores this.
- Corollary — **per-decode-step weight *traffic*** W (what f(B) actually wants) differs from disk: the embedding table contributes one 8 KB row per token (~free), but the bf16 lm_head is read *in full* every step (a [vocab × hidden] matvec = ~1.24 GB/step that no weight-only recipe touched). W_bf16 = 16.38 · W_fp8 ≈ ___ · W_int4 ≈ ___ GB.

**2. Batch-1: two models disagree — the day's real experiment.** How does the 88.4 baseline scale when the weight read shrinks?
- **Model A, multiplicative:** measured efficiency 71.6% of ideal is a constant factor → tok/s = 0.716 × 2,039 ÷ (W_q + 0.147). Predicts FP8 ≈ ___, INT4 ≈ ___ tok/s.
- **Model B, additive overhead:** Day 4 measured the overhead as a *fixed* ~3.2 ms/step that doesn't scale with bytes → step = (W_q + 0.147)/2,039 + 3.2 ms. Predicts FP8 ≈ ___, INT4 ≈ ___ tok/s.
The models diverge hard at INT4 (roughly 320-ish vs 190-ish — fill in exact). **Commit in ink to which one I believe and why** (Day 4's evidence points at B: overhead was already visible as a constant at batch 1). If B wins, the punchline writes itself: *shrink the weights 4× and Amdahl's law presents the bill — the 3 ms nobody billed becomes the bottleneck.* Predicted winner: ___. Any Marlin dequant cost on top would push below even Model B; the gap, if any, is measurable as ms/step minus both models.
- Also predict in ink: measured INT4 speedup over bf16 = ___× (the tweet number; the naive reader expects ~4×).

**3. The whole curve: what quantization can't do.** f_q(B) = 2,039·B ÷ (W_q + 0.147·B). Quantizing weights moves the **half-saturation point** left — B½ = W_q/0.147 goes 111 (bf16) → ~___ (fp8) → ~___ (int4) — but the **asymptote 2,039/0.147 ≈ 13,870 tok/s doesn't move**, because it's set by the KV term, which weight quantization never touches. The three predicted curves *converge* as B grows: quantization is a low-batch drug. Predicted modeled C=64 ratio fp8/bf16 = ___, int4/bf16 = ___; measured ratios will land *below* modeled (the prefill tax and per-step overhead are unchanged and were already 2× of the story on Day 4). Predicted measured C=64 gain: fp8 ___×, int4 ___×.
- The chart this produces — three measured points-per-variant over three predicted curves, converging — is the day's artifact.

**4. Quality, predicted.** Same eval, all three arms, greedy where possible. Primary: wikitext perplexity via lm-eval (loglikelihood-based — fast, no generation, no thinking-mode confounds with Qwen3). Predicted: FP8 within noise of bf16 (ΔPPL < ___), INT4-g128-GPTQ visibly nonzero (ΔPPL ≈ ___). This is the shallow check; per-token KL (unit 5.3) is the honest instrument and is *not* today.

## Steps

- [ ] Rent box (target A100-SXM4 class again for anchor continuity, ~$1.20/hr; session budget ≈ 3 hr). Toolchain checklist. `python -m venv .venv && source .venv/bin/activate`; `pip install vllm llmcompressor lm-eval`
- [ ] **FP8 arm** (no calibration — minutes):
  ```python
  # quantize_fp8.py
  from llmcompressor import oneshot
  from llmcompressor.modifiers.quantization import QuantizationModifier

  oneshot(
      model="Qwen/Qwen3-8B",
      recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]),
      output_dir="Qwen3-8B-FP8-Dynamic",
  )
  ```
- [ ] **INT4 arm** (GPTQ, 512 calibration samples, ~20–40 min on A100 — watch `nvidia-smi` during the Hessian pass; the quantizer is itself a workload):
  ```python
  # quantize_w4a16.py
  from llmcompressor import oneshot
  from llmcompressor.modifiers.quantization import GPTQModifier

  oneshot(
      model="Qwen/Qwen3-8B",
      dataset="open_platypus",
      recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
      max_seq_length=2048,
      num_calibration_samples=512,
      output_dir="Qwen3-8B-W4A16-G128",
  )
  ```
- [ ] `du -sh` both output dirs → score prediction 1. Then actually **read the artifact**: `config.json`'s `quantization_config` (scheme, group_size, ignored modules) and the safetensors index — find the scale tensors, check their shapes against the group-of-128 story from mechanism block 1. This is the "one-diagram explainer of a quantized tensor" (unit 5.1) drawn from a real file.
- [ ] Serve each variant in turn, same flags as Day 4 **plus `--no-enable-prefix-caching`**: `vllm serve <path> --no-enable-prefix-caching`. Per server: grep startup log for (a) FlashInfer sampler line, (b) **which quant kernel it picked** (expect `marlin` on A100 for both arms — the mechanism-block-6 story, confirmed or refuted in one grep), (c) `num_gpu_blocks` — freed weight memory should reappear as a bigger KV pool; predicted new block counts from Day 3's accounting: fp8 ___, int4 ___
- [ ] Batch-1 bench each arm (`vllm bench serve`, C=1, 1k in/1k out, `ignore_eos`, top_p/top_k set, no seed, ≥3 min) → score prediction 2, crown Model A or B
- [ ] One C=64 point each arm (mini-sweep 1/8/64 if pace allows) → score prediction 3
- [ ] Quality: `lm_eval --model vllm --model_args pretrained=<path>,gpu_memory_utilization=0.9 --tasks wikitext --batch_size auto` per arm → score prediction 4
- [ ] Chart: measured points over the three predicted f_q curves, converging at high B; second panel or annotation for batch-1 Model A vs Model B vs measured
- [ ] If short on time: **FP8 arm alone is a complete day** (no calibration, one comparison); INT4 rolls to a spillover morning. Don't let arm 2 kill the post.

## Measurements

*(Answers to Q1–Q6 in own words go here first.)*

| Variant | Disk (GB) | W traffic (GB/step) | Batch-1 tok/s | ms/step | C=64 agg tok/s | Wikitext PPL | Kernel (from log) | num_gpu_blocks |
|---|---|---|---|---|---|---|---|---|
| bf16 (Day 4 anchor) | 16.38 | 16.38 | 88.4 | 11.31 | 2,505 | | — | 25,640 |
| FP8-Dynamic | | | | | | | | |
| W4A16-g128 | | | | | | | | |

- Prediction 1 (sizes): predicted ___ / ___ · measured ___ / ___
- Prediction 2 (batch-1, Model A vs B): A said ___ / ___, B said ___ / ___ · measured ___ / ___ · winner: ___
- Prediction 3 (C=64 convergence): predicted ratios ___ · measured ___
- Prediction 4 (quality): ΔPPL predicted ___ / ___ · measured ___ / ___

## If time remains

- **AWQ third arm:** llm-compressor's `AWQModifier`, same eval — does protection beat compensation on this checkpoint? (Mechanism block 4, measured.)
- **fp8 KV cache teaser (unit 5.4):** `--kv-cache-dtype fp8` on the bf16 server. This is the knob that moves what weight quantization can't: halving KV bytes doubles the 13,870 asymptote *and* doubles max concurrent contexts. One C=64 point would preview it.
- **The H100 question:** rent one (~$2–3/hr) and rerun the FP8 arm with real fp8 tensor cores — the only way to see the W8A8 *compute* win, which should show up in TTFT/prefill, exactly where today's arms show nothing. Probably its own day.
- **Load-time comparison:** cold `vllm serve` wall-clock per variant — 4× fewer bytes off disk should show up here too (Day 2's cold-start anatomy, revisited).

## Reading-share post (posted before the hands-on work)

*Drafted 2026-08-03 after reading the Visual Guide; the measurement post still comes after the runs. No em dashes. Format: context recap → what I learned.*

*Attach (up to 4, all Grootendorst's, credited in-post): (a) the GGUF super-block/sub-block structure diagram — strongest pick, it IS the Day 1 tie-in; (b) the dynamic quantization flowchart (per-layer scale computed at runtime) — item 2; (c) the weight-vs-activation distributions diagram — item 3; (d) the GPTQ error-redistribution visual — item 3's GPTQ line. The absmax scale-factor mapping is the substitute if any of these crop badly.*

> Day 6/45 of Inference Engineering: a Crash Course in Quantization
>
> If you're looking for a quick intro to help understand quantization, I highly recommend "A Visual Guide to Quantization" by Maarten Grootendorst (https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
>
> In the next day or so I'll be quantizing my own checkpoint and benchmarking it, but not before I do a bunch of background reading!
>
> Some interesting facts I recently learned:
>
> 1. On most GPUs, 4-bit weights must get dequantized inside the kernel. That means most of the quantization benefits are found in memory and bandwidth, rather than in compute. Native hardware is required to support lower-bit math. Notably, NVIDIA's Blackwell generation (B200, RTX 50 series) and AMD's MI355X both have native FP4 Tensor Cores. Decode is memory-bound, so it still gets faster. But prefill is compute-bound, so the extra dequantization work can actually make the 4-bit model SLOWER than the 16-bit one.
>
> 2. Some quantization happens at inference time, per token. Activations depend on the prompt, so they can't be pre-quantized. Schemes like dynamic W8A8 scale and quantize each token's activations on the fly, which does imply additional work during the forward pass, but without assuming the distribution of the activations beforehand.
>
> 3. Weights and activations also get treated very differently. Weights never change, so they can go through expensive offline algorithms. GPTQ, for example, doesn't just round each weight to its nearest value. It quantizes a layer weight by weight and pushes each rounding error onto the weights that haven't been quantized yet. The goal is to preserve the layer's OUTPUT, not the weights themselves. Activations have extreme outliers and need to be quantized quickly at runtime, so they usually stay at 8 bits while weights go down to 4.
>
> 4. GGUF is a quantization format, not just a file format. Funny thing: I had already measured this without knowing it. On Day 1 my "4-bit" Q4_K_M model came out to ~4.9 bits per weight, because of exactly the block scale factors this article explains, and the file byte ratio (not the nominal bits) is what predicted my measured Q8 vs Q4 speed difference.
>
> (All visuals are Maarten's, from the linked article.)

## Post skeleton

- Hook: "I shrank my model 4× and it decoded only ___× faster. I predicted the shortfall before running — and the missing speed has two names."
- The chart: three predicted curves converging, measured points on top.
- The numbers: batch-1 Model A vs Model B vs measured (Amdahl's bill); C=64 ratios (quantization is a low-batch drug — weight-only quant can't touch KV or prefill).
- The artifact moment: opening `config.json` and finding the group-128 scales — what a quantized tensor actually is, from a real file.
- What surprised me: ___
- Tomorrow: ___ (candidates: the week-1 "life of a token" walkthrough (unit 1.5) with all the measured numbers now in hand, or Arc 2.1 naive generation)
