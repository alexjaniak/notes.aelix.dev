# Inference Economics

Research dump (July 2026) answering: *Why is cost to serve models decreasing so much if GPU prices aren't dumping? How do people know they're getting served the right model?* From [[Questions]]. Related: [[Serving Models]], [[Inference]], [[Frontier Chips]].

## Why is cost-to-serve dropping ~10x/year?

**The trend:** constant-quality inference cost falls **~10x/year** (a16z "LLMflation"): GPT-3-level output cost $60/M tokens in 2021 → $0.06/M in late 2024 = 1,000x in 3 years. Epoch AI measures 9–900x/yr depending on task (median ~50x/yr, accelerating after 2024). Stanford HAI: GPT-3.5-level fell 280x in ~2 years. A more careful MIT FutureTech estimate (arXiv:2511.23455): 5–10x/yr at the benchmark level, of which **~3x/yr is pure algorithmic efficiency**. Crucial counterpoint: the cost of running the *frontier* model is simultaneously **rising** 3–18x/yr (bigger models + longer reasoning traces) — cheapness is at fixed capability, not at the frontier.

**The factor stack — and why "GPU prices" is the wrong lens.** One 2026 econometric decomposition attributes essentially all of the decline to software/algorithms and ~0% to GPU hardware prices:

1. **Algorithmic efficiency (~3x/yr alone)** — distillation of frontier models into mini/flash tiers (probably the single biggest driver); MoE sparsity (DeepSeek-V3: 671B total / 37B active — you pay for active params only); **MLA** compressing KV cache ~7x vs Llama-405B (70 KB/token vs 516 KB — the cheap-long-context unlock, see [[Inference]]); quantization FP16→FP8→FP4 (Llama-70B at NVFP4 ≈ 43 GB at ~98–99% of BF16 quality)
2. **Serving software multipliers** (mostly quality-neutral): continuous batching (up to ~23x), PagedAttention (2–4x), prefix caching (up to 5x), speculative decoding (2–3x, output-identical), prefill/decode disaggregation (+59–498% goodput) — see [[Serving Models]]. SemiAnalysis: B200 cost/M tokens on gpt-oss-120B fell $0.11 → $0.02 in two months *purely from software*
3. **Hardware generations** (real but smaller than headlines): H100→B200 like-for-like ≈ 4x decode throughput ≈ ~3x cheaper/M tokens on-demand; NVIDIA's "25–30x" claims are rack-scale GB200-vs-H100-BF16 comparisons
4. **GPU *rental* prices did crash even though chip ASPs didn't**: H100 $8/hr (2023) → ~$2.85 (early 2024) → $1.20–3.50/hr (Q2 2026). The right unit is tokens/$/GPU-hour, not chip sticker price
5. **Utilization**: batching amortizes one weight-load across many users; provider cluster utilization rose (Lambda 65%→85%)
6. **Prompt caching as pricing**: cache reads at 10% of input price (Anthropic, OpenAI), DeepSeek ~2% — agentic workloads have high hit ratios, so *effective* $/token fell faster than list prices
7. **Competition**: open weights commoditize the model layer — any capability appearing in open weights gets instant multi-provider price war (5 providers converged at $2.17/M for DeepSeek V4 Pro within weeks). DeepSeek made a 75% cut permanent (May 2026)

**Is it below cost?** Mostly not anymore. DeepSeek disclosed a theoretical 84.5% gross margin at R1 list prices; SemiAnalysis estimates Anthropic inference gross margins went −94% (2024) → ~65–70% (2026). Loss-leader pricing was real in 2023–24; the subsidy today lives in subscriptions (Max/Pro plans deliver 20–70x list-price token value) and free tiers, not the API.

**One-line answer:** GPU chips aren't cheaper, but everything around them is — ~3x/yr algorithms, big software-serving multipliers, ~30%/yr hardware $/perf plus collapsed rental prices, and open-weights competition compressing margins — compounding to ~10x/yr at fixed quality.

## How do you know you're getting the right model?

**The trust problem is real and measured:**

- **Model Equality Testing** (Stanford, arXiv:2410.20247): statistical two-sample test on outputs, <$1 per audit. Applied to 31 commercial Llama endpoints across 9 providers: **11 of 31 statistically differed from Meta's reference weights**. Perplexity failed all tests; some deviations were larger than substituting a *different model entirely*. Only 2 of 9 providers disclosed their optimizations.
- **Kimi K2 Vendor Verifier** (Moonshot): official-vs-vendor comparison on 4,000 tool-call prompts — official/Fireworks/DeepInfra at 100% schema accuracy, Together 72%, Nebius 50.6%. "Making a model open is only half the battle."
- **Artificial Analysis** provider benchmarks: same open model shows meaningful quality spread across providers, tightening over weeks ("burn-in") but never fully.
- **Surprising nuance** (OpenRouter's data): quantization alone often *isn't* the culprit — some FP4 providers beat FP8 ones. The bigger causes are **tool-call parsers, chat templates, sampler defaults, and engine bugs**. But silent quantization does exist (DeepInfra served DeepSeek at FP4 with a 66K-vs-1M context window at the same price).
- Degradation concentrates where agents live: free-form text barely moves under quantization, but JSON-mode drops 3–8 pts and tool-argument correctness 4–10 pts at INT4.

**Verification approaches, weakest to strongest:**

1. **Benchmark/statistical auditing** (black-box, cheap, probabilistic): Artificial Analysis, K2 Vendor Verifier, Model Equality Testing; logprob drift checks vs a self-hosted reference
2. **Market/reputation mechanisms** — what actually protects most users today: **OpenRouter Exacto/Auto Exacto** (re-scores providers every ~5 min on tool-call telemetry + benchmarks, deranks statistical outliers; cut gpt-oss error rates 5.6%→3.5%); a `quantizations` request filter with provider-disclosed precision labels
3. **Activation fingerprinting** (research → production): **TOPLOC** (Prime Intellect, ICML 2025 — 258-byte locality-sensitive hash of top-k activations per 32 tokens; 100% detection of model/precision substitution, robust to GPU nondeterminism), SVIP, Ambient proof-of-logits, Hyperbolic proof-of-sampling
4. **TEE attestation — the practical winner**: NVIDIA Confidential Computing (H100+) + Intel TDX/AMD SEV-SNP, remote attestation; **<7–9% throughput overhead, ~0% for large models**. Used by Azure confidential GPU, Near AI, Phala, Atoma. Trust shifts to the chip vendor's keys
5. **zkML — not viable for LLMs yet**: zkLLM needs ~15 min proving per forward pass on a 13B (a 100-token answer ≈ 25 hours); fine for small-model/DeFi-grade integrity only

**Synthesis:** the accuracy cost of efficiency tricks is borne mostly invisibly by users of third-party-hosted open models. Detection today = independent benchmarks + statistical tests; proof today = TEEs; the pragmatic middle layer = router reputation systems. Closed labs (OpenAI/Anthropic/Google) rest purely on reputation — the same tricks are available to them, and only aggregate evals or TEEs would catch it. And provider variance exists *because* of the price competition above: converged pricing pushes differentiation into precision/engine choices users can't see.
