When a market has structural information asymmetry—the suppliers observe product quality that the buyers aren't privy to—buyers ask at the expected quality of the pool, the sellers of above-average goods exist, the pool degrades, and the equilibrium unravels toward only lemons trading. This was Akerlof's insight 

Third-party LLM inference satisfies both conditions almost exactly. Open weights collapsed the model layer into a commodity: within weeks of DeepSeek V4 Pro's release, five of six providers converged on an identical $2.17/M blended price. At a pooled price, the remaining competitive margin is serving cost, and the cheapest levers are the ones that alter the output distribution. Each cuts cost immediately and measurably; each degrades fidelity in ways that are per-request invisible and statistically diffuse.

 

The empirical evidence matches the prediction. Gao et al. (ICLR 2025) formalized the audit problem as Model Equality Testing, a two-sample test using Maximum Mean Discrepancy over a string kernel, achieving 77% median power against realistic distortions at roughly ten samples per prompt, under a dollar per audit. Applied to 31 commercial endpoints serving four Llama models across nine providers in Summer 2024: 11 of 31 deviated statistically from Meta's reference weights. Perplexity's Llama-3 8B endpoint failed against both the fp32 and int8 null distributions, with an estimated MMD exceeding the distance to a different model, and only two of nine providers disclosed any distribution-altering optimization. Moonshot's K2 Vendor Verifier found the same structure from the model owner's side: on 4,000 tool-call prompts scored against the official API, third-party endpoints ranged from 100% schema conformance down to 50.6%.

What makes this worth writing about is that both halves of Akerlof's paper apply. The first half predicts the degradation; the second half enumerates the remedies: disclosure, reputation, certification, and technologies that make quality observable. All four now exist in embryonic form: mandatory quantization labels on OpenRouter, telemetry-based provider deranking re-scored every five minutes, model owners certifying their own resellers, and TEE attestation that replaces statistical detection with cryptographic proof at sub-9% overhead. This post covers the whole arc: why serving costs fell ~10x/year, which optimizations in that factor stack are output-preserving and which are not, and whether the emerging verification layer can clear the market before it pools.

## "LLMflation"

![[llm-inference-price-trends.png]]

Constant-quality inference cost falls **~10x/year** ([a16z "LLMflation"](https://a16z.com/llmflation-llm-inference-cost/), Guido Appenzeller, Nov 2024): GPT-3-level output (MMLU 42) cost $60/M tokens in Nov 2021 → $0.06/M (Llama 3.2 3B on Together) in late 2024 = 1,000x in 3 years.

[Epoch AI](https://epoch.ai/data-insights/llm-inference-price-trends) measures 9–900x/yr depending on the benchmark milestone (GPT-4-level GPQA fell 40x/yr), with the fastest drops in the most recent year. 

The [Stanford HAI AI Index 2025](https://hai.stanford.edu/ai-index/2025-ai-index-report): GPT-3.5-level (MMLU 64.8) fell $20.00/M (Nov 2022) → $0.07/M (Oct 2024), more than 280-fold. 

A more careful MIT FutureTech estimate ([*The Price of Progress*](https://arxiv.org/abs/2511.23455), Gundlach et al.): 5–10x/yr at the benchmark level, of which **~3x/yr is pure algorithmic efficiency**.

Crucial counterpoint from the same paper: the cost of running the *frontier* model is simultaneously **rising** 3–18x/yr (bigger models + longer reasoning traces) — cheapness is at fixed capability, not at the frontier.

## Why is cost-to-serve plummeting?

A 2026 econometric decomposition ([*Tiered Super-Moore's Law*](https://arxiv.org/abs/2603.28576), using OpenRouter + Epoch data) attributes essentially all of the ~600x 2020–2026 decline to software/algorithms (TFP residual ≈ 103.7% of cost reduction) and ~0% to GPU hardware (−0.9%): 

**Algorithmic efficiency (~3x/yr alone)**
* distillation of frontier models into mini/flash tiers (probably the single biggest driver). 
	* Get example
* MoE sparsity ([DeepSeek-V3](https://arxiv.org/abs/2412.19437): 671B total / 37B active — you pay for active params only)
* **MLA** compressing the KV cache ~7.3x vs Llama-3.1-405B (70 KB/token vs 516 KB, per DeepSeek's [hardware-reflections paper](https://arxiv.org/abs/2505.09343) — the cheap-long-context unlock); 
	* What is MLA? 
* quantization FP16→FP8→FP4 (Llama-70B at NVFP4 ≈ 43 GB at ~98–99% of BF16 quality). 
	* Doesn't this need to be cited? 

Serving Software Multipliers: 
* continuous batching (up to ~23x), PagedAttention (2–4x), prefix caching (up to 5x), speculative decoding (2–3x, output-identical), prefill/decode disaggregation (+59–498% goodput)
	* what is this stuff and where are these numbrrs from
* On SemiAnalysis's [InferenceMAX](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference) benchmark, [NVIDIA reported](https://blogs.nvidia.com/blog/blackwell-inferencemax-benchmark-results/) B200 cost/M tokens on gpt-oss-120b falling $0.11 → $0.02 in two months *purely from software* (TensorRT-LLM updates)

Hardware Generations
* H100→B200 like-for-like ≈ 4x decode throughput ≈ ~3x cheaper/M tokens on-demand; NVIDIA's "25–30x" claims are rack-scale GB200-vs-H100-BF16 comparisons.
	* What evidence is there for this?

**GPU *rental* prices did crash even though chip ASPs didn't**: H100 $8/hr (2023) → ~$2.85 (early 2024) → $1.20–3.50/hr (Q2 2026) — see [*$2 H100s: How the GPU Bubble Burst*](https://www.latent.space/p/gpu-bubble) (Eugene Cheah, Oct 2024). The right unit is tokens/$/GPU-hour, not chip sticker price
* how big of a factor is this? 

**Utilization**: batching amortizes one weight-load across many users; interactivity (tok/s/user) is the dial providers turn between margin and UX

**Prompt caching as pricing**: cache reads at 10% of input price ([Anthropic](https://platform.claude.com/docs/en/build-with-claude/prompt-caching); [DeepSeek](https://api-docs.deepseek.com/news/news0802/) started at 10% in Aug 2024, now ~2% on V4 Flash) — agentic workloads have high hit ratios, so *effective* $/token fell faster than list prices
* so the price for caches fell or?

**Competition**: open weights commoditize the model layer — any capability appearing in open weights gets instant multi-provider price war ([five of six providers converged at $2.17/M blended](https://deepinfra.com/blog/deepseek-v4-pro-max-api-benchmarks-latency-throughput-cost) for DeepSeek V4 Pro Max). DeepSeek then made its 75% cut [permanent](https://api-docs.deepseek.com/quick_start/pricing/) (May 2026: $0.435/$0.87 per M, 1M context)

**Is it below cost?** Mostly not anymore. DeepSeek [disclosed](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) (Feb 2025 data; [Reuters](https://www.reuters.com/technology/chinas-deepseek-claims-theoretical-cost-profit-ratio-545-per-day-2025-03-01/)) a theoretical 545% cost-profit ratio = **84.5% gross margin** at R1 list prices ($87k/day cost vs $562k theoretical revenue).

 [SemiAnalysis](https://newsletter.semianalysis.com/p/anthropic-growth-and-bedrock-mix) estimates Anthropic inference gross margins went **−94% (2024) → 38% (2025) → mid-60s (2026)**. Loss-leader pricing was real in 2023–24; the subsidy today lives in subscriptions (Max/Pro plans deliver far above list-price token value) and free tiers, not the API.

**One-line answer:** GPU chips aren't cheaper, but everything around them is — ~3x/yr algorithms, big software-serving multipliers, hardware $/perf gains plus collapsed rental prices, and open-weights competition compressing margins — compounding to ~10x/yr at fixed quality.

## 

If there is a market for it, aka if there is the opportunity for scamming, it will happen
	find a quote for this

The unfortunate truth is that no one knows what model you are serving, and you can silently degrade it to stay at the perado frontier

Third-party inference is now a serious business growing at venture-fantasy rates. Enterprise LLM API spend hit $8.4B by mid-2025 — up 2.4x in six months — and is projected to reach ~$15B by the end of 2026 ([Menlo Ventures](https://menlovc.com/perspective/2025-mid-year-llm-market-update/)). The specialist layer serving open weights crossed into unicorn-factory territory in a single year: Together AI passed $1.15B in annual bookings and raised at $8.3B (July 2026), Fireworks processes 10T+ tokens a day and is in talks at a $15B valuation — nearly 4x its price seven months earlier — Baseten hit ~$600M ARR at up to $13B, and NVIDIA paid ~$20B just to license Groq's inference chips. But here's the catch that matters: this gold rush has no pricing power at the model layer. The weights are open, switching costs are an API string, and five of six providers serving DeepSeek V4 Pro converged on an identical $2.17/M within weeks of release. Fireworks — the category leader — runs ~50% gross margins ([Sacra](https://sacra.com/c/fireworks-ai/)), well below software norms, because GPU costs sit in COGS and prices are set by the most aggressive competitor. So every provider faces the same equation: billions in revenue at stake, prices pinned to the floor by perfect competition, and exactly one cost lever left that customers can't observe — the fidelity of the model itself. Quantize to FP4, cap the context window, relax the speculative decoder, and your margin improves immediately and measurably; the quality loss is diffuse, per-request invisible, and lands on someone else's eval. A market this large, this commoditized, and this unobserved is a textbook setup for quality shading — and the audits ([11 of 31 endpoints deviating from reference weights](https://arxiv.org/abs/2410.20247), [providers at 50.6% tool-call accuracy](https://github.com/MoonshotAI/K2-Vendor-Verifier)) confirm it's not hypothetical.



### Silent Quantization
- Mechanism: serve INT4/FP4 weights while advertising the full model. Halves+ memory cost per replica.
- Harm: degradation is uneven — INT4 causes a significant accuracy drop specifically for decoder-only models (Wu et al., ICML 2023, [arXiv:2301.12017](https://proceedings.mlr.press/v202/wu23k.html)), and practitioner consensus is to avoid INT4 for math, code, and reasoning-heavy tasks where loss is most noticeable. There's also a safety angle: Egashira et al., "Exploiting LLM Quantization" (NeurIPS 2024, arXiv:2405.18137) shows quantization can activate adversarial behavior absent in the full-precision model.
- Evidence of practice: Gao et al. found 11 of 31 Llama endpoints statistically differed from Meta's reference weights, with only 2 of 9 providers disclosing any distribution-altering optimization ([arXiv:2410.20247](https://arxiv.org/abs/2410.20247), ICLR 2025).

### Context Window Trunctation

- Mechanism: cap or silently truncate context far below the advertised window. KV cache is the memory bottleneck, so this is a huge cost lever.
- Harm: the model literally never sees part of your prompt; failures look like model stupidity, not infrastructure.
- Evidence: DeepInfra serving DeepSeek V4 Pro at 66K vs 1M elsewhere at identical $2.17/M pricing ([their own benchmark page](https://deepinfra.com/blog/deepseek-v4-pro-max-api-benchmarks-latency-throughput-cost)), plus a [live HF forum dispute](https://discuss.huggingface.co/t/deepinfra-deepseek-v4-pro-context-size-wrong-on-huggingface/176578) where DeepInfra's site claimed 1024K while the endpoint capped at 64K.

**KV cache quantization & eviction**

- Mechanism: quantize the cache (W4A4KV4-style) or evict "unimportant" tokens mid-generation (H2O, arXiv:2306.14048; StreamingLLM, arXiv:2309.17453) to fit larger batches.
- Harm: long-context recall degrades; the model "forgets" earlier conversation. Nearly impossible to attribute from outside.
- ⚠️ These are published as legitimate techniques; evidence of _undisclosed provider use_ is thin — frame as "possible and invisible" rather than "documented in the wild."

**5. Lenient speculative decoding**

- Mechanism: standard spec decoding is output-identical (target model verifies every draft token). Relax the acceptance rule ("typical acceptance," Medusa, arXiv:2401.10774) and you accept tokens the big model would reject — you're partially reading the draft model's output.
- Harm: a tunable dial between "free speedup" and "quietly serving a smaller model," invisible per-response.
- ⚠️ Same caveat as #3: published technique, no documented in-the-wild abuse case. Its undetectability is the point worth making.

**. Model substitution / model mixing**

- Mechanism: serve a distilled, older, or smaller model under the flagship's name; or mix models across requests to blur the statistical signature.
- Harm: the maximal version of everything above.
- Evidence: some provider deviations in the Stanford audit were larger than substituting a different model entirely; the rank-based uniformity test paper explicitly treats providers mixing multiple models as an evasion strategy ([arXiv:2506.06975](https://arxiv.org/pdf/2506.06975)).

**7. Undisclosed system prompts, fine-tuning, watermarking**

- Mechanism: prepend a hidden system prompt, fine-tune for bias, or watermark outputs — all change the output distribution without changing the advertised model name.
- Evidence: this is the explicit threat model of TOPLOC — a provider could secretly reduce precision, fine-tune to introduce biases, or prepend an undisclosed system prompt ([arXiv:2501.16007](https://arxiv.org/pdf/2501.16007), ICML 2025) — and of Model Equality Testing (quantize/watermark/finetune, possibly without notifying users).

**8. Incompetence (the biggest one in practice)**

- Mechanism: not an attack — buggy tool-call parsers, wrong chat templates, bad sampler defaults, engine version drift.
- Harm: empirically larger than quantization. Moonshot's K2VV measured Together at 72% and Nebius at 50.6% tool-call schema accuracy vs 100% official ([repo](https://github.com/MoonshotAI/K2-Vendor-Verifier), [announcement](https://medium.com/@kimi_moonshot/announcing-the-k2-vendor-verifier-ensuring-consistent-toolcall-performance-for-kimi-k2-04c568f4a1dd)); Chutes' postmortem attributed most failures to vLLM/sglang parsing bugs. OpenRouter's telemetry across billions of requests confirmed the variance ([provider variance blog](https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/)).
- Structural point for the essay: converged pricing ($2.17/M across five providers) is _why_ this happens — when price differentiation is gone, corners get cut in the only dimension users can't observe.

**Framing device for the section**: every entry shares one signature — savings are immediate and measurable to the provider; damage is diffuse, per-query invisible, and lands hardest on agentic workloads (tool calls, JSON, long context) rather than chat. That's the bridge to your fixes section: detection methods are ranked precisely by which of these they can catch (statistical tests catch #1 and #6; telemetry catches #8; only TEE attestation catches #4 and #5).

## Attestation

### Router Governance
OpenRouter now _requires_ providers to declare quantization in their models endpoint (valid values: int4, int8, fp4, fp6, fp8, fp16, bf16, fp32) along with disclosure of whether prompts are logged and used for training, and it publicly tracks TTFT, throughput, and uptime per provider, with Auto Exacto routing tool-calling traffic based on tool-call success rates. With 70+ providers and 10M+ developers, that's real market power — but it only governs traffic through OpenRouter. Nothing stops a provider from serving degraded models direct-to-customer. It's private regulation, not a standard.

**The trust problem is real and measured:**

- **[Model Equality Testing](https://arxiv.org/abs/2410.20247)** (Gao, Liang, Guestrin — Stanford, ICLR 2025): statistical two-sample test on outputs, <$1 per audit. Applied to 31 commercial Llama endpoints across 9 providers (Summer 2024): **11 of 31 statistically differed from Meta's reference weights**. Perplexity failed all tests; some deviations were larger than substituting a *different model entirely*. Only 2 of 9 providers disclosed their optimizations.
- **[Kimi K2 Vendor Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier)** (Moonshot): official-vs-vendor comparison on ~4,000 tool-call prompts — official/Fireworks/DeepInfra at 100% schema accuracy, Together ~72%, Nebius at 50.6% tool-call *trigger similarity* (Nov 2025 snapshot; the table updates). "Making a model open is only half the battle."
- **[Artificial Analysis](https://artificialanalysis.ai/models/gpt-oss-120b/providers)** runs per-provider quality benchmarks (GPQA ×16 runs) on the same open model: meaningful spread across providers, tightening over weeks ("burn-in") but never fully. The famous Aug 2025 gpt-oss-120b episode ([Simon Willison's writeup](https://simonwillison.net/2025/Aug/15/inconsistent-performance/)) saw AIME scores range 36.7%–93.3% across providers in week one.
- **Surprising nuance** (OpenRouter's data): quantization alone often *isn't* the culprit — some FP4 providers beat FP8 ones. The bigger causes are **tool-call parsers, chat templates, sampler defaults, and engine bugs**. But undisclosed serving differences do exist: [DeepInfra served DeepSeek V4 Pro Max at FP4 with a 66K context window](https://deepinfra.com/blog/deepseek-v4-pro-max-api-benchmarks-latency-throughput-cost) vs 1M elsewhere in the same blended-price tier (though V4 was QAT-trained largely at FP4, so the quant is less of a downgrade than it sounds).
- Degradation concentrates where agents live: free-form text barely moves under quantization, but JSON-mode drops 3–8 pts and tool-argument correctness 4–10 pts at INT4 ([FutureAGI's vLLM eval guide](https://futureagi.com/blog/evaluating-vllm-self-hosted-llm-2026/) — vendor-reported customer-workload numbers, not peer-reviewed).

**Verification approaches, weakest to strongest:**

1. **Benchmark/statistical auditing** (black-box, cheap, probabilistic): Artificial Analysis, K2 Vendor Verifier, Model Equality Testing; logprob drift checks vs a self-hosted reference
2. **Market/reputation mechanisms** — what actually protects most users today: OpenRouter **[Exacto](https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/)** (Oct 2025) and **[Auto Exacto](https://openrouter.ai/blog/announcements/auto-exacto/)** (Mar 2026, on by default: re-scores providers roughly every 5 minutes on tool-call telemetry + benchmarks, deranks statistical outliers; cut gpt-oss-120b error rates 5.6%→3.5%); plus a `quantizations` request filter with provider-disclosed precision labels
3. **Activation fingerprinting** (research → production): **[TOPLOC](https://arxiv.org/abs/2501.16007)** (Prime Intellect — 258-byte locality-sensitive hash of top-k activations per 32 tokens, ~1000x smaller than raw activations; 100% detection of model/prompt/precision substitution in their evals, robust to GPU nondeterminism), **[SVIP](https://arxiv.org/abs/2410.22307)** (hidden-state proxy-task verification, <5% FNR / <3% FPR), Ambient proof-of-logits, Hyperbolic proof-of-sampling
4. **TEE attestation — the practical winner**: NVIDIA Confidential Computing (H100+) with Intel TDX/AMD SEV-SNP and remote attestation; a [third-party benchmark study](https://arxiv.org/abs/2409.03992) (io.net-funded, not NVIDIA's) found **<7% throughput overhead for typical queries, near zero for large models/long sequences** (the bottleneck is CPU–GPU PCIe encryption, not compute). Used by Azure confidential GPU, Near AI, Phala, Atoma. Trust shifts to the chip vendor's keys
5. **zkML — not viable for LLMs yet**: [zkLLM](https://arxiv.org/abs/2404.16109) (CCS 2024) proves an entire 2048-token LLaMA-2-13B inference in under 15 minutes on an A100 (~803s proving, 1–3s verification, <200 kB proof) — impressive research, still orders of magnitude too slow for production serving. Survey: [Equilibrium Labs, *State of Verifiable Inference*](https://equilibrium.co/writing/state-of-verifiable-inference)

**Synthesis:** the accuracy cost of efficiency tricks is borne mostly invisibly by users of third-party-hosted open models. Detection today = independent benchmarks + statistical tests; proof today = TEEs; the pragmatic middle layer = router reputation systems. Closed labs (OpenAI/Anthropic/Google) rest purely on reputation — the same tricks are available to them, and only aggregate evals or TEEs would catch it. And provider variance exists *because* of the price competition above: converged pricing pushes differentiation into precision/engine choices users can't see.


**1. Router governance (closest thing to a de facto standard, but it's one company's walled garden)**

OpenRouter now _requires_ providers to declare quantization in their models endpoint (valid values: int4, int8, fp4, fp6, fp8, fp16, bf16, fp32) along with disclosure of whether prompts are logged and used for training, and it publicly tracks TTFT, throughput, and uptime per provider, with Auto Exacto routing tool-calling traffic based on tool-call success rates. With 70+ providers and 10M+ developers, that's real market power — but it only governs traffic through OpenRouter. Nothing stops a provider from serving degraded models direct-to-customer. It's private regulation, not a standard.

**2. Model-owner self-policing (fragmenting per vendor, no shared framework)**

Moonshot expanded K2VV into the full Kimi Vendor Verifier in April 2026 — six benchmarks including long-output stress tests designed to catch KV cache bugs and quantization degradation, pre-release early access so vendors can validate before launch, and published performance rankings. That's the most complete template anyone has. But it's per-model-family: DeepSeek, Meta, Qwen, Z.AI would each need their own. No shared harness or certification body exists.

**3. Cryptographic/statistical verification (research is ahead of adoption — and the researchers say so)**

The telling evidence that no standard exists: the DiFR paper (Nov 2025) explicitly closes by _advocating_ that "inference providers adopt standardized reference implementations and consistent sampling algorithms, enabling rigorous verification by any third-party user" — that's a plea, not a description of reality. TOPLOC frames the same gap: without robust verification methods, users can only rely on provider claims. There's a steady stream of papers (rank-based uniformity tests, NANOZK layerwise ZK proofs, AttestLLM, "Attesting LLM Pipelines" proposing an AI Bill of Materials) — plenty of tooling, zero adoption mandate.

**4. TEE attestation (the technology exists industry-wide; deployment doesn't)**

This is furthest along on the trust axis. Azure now offers confidential inferencing for Azure OpenAI — encrypted prompts that can only be decrypted inside CPU+GPU TEEs, with remote verifiability rooted in hardware and a tamper-proof transparency ledger auditable by external parties. That's the first hyperscaler shipping verifiable claims for a _closed_ model. Phala processed over 1.34B tokens in a day at ~0.5–5% overhead and became a verified OpenRouter provider, and co-founded a "Verifiable Compute Consortium" with Sentient. But: the consortium is a decentralized-AI/crypto grouping, not an industry body; Azure's system primarily attests _privacy handling_, and TEE-served tokens remain a tiny fraction of total inference. Notably there IS an underlying standard here — IETF's RATS architecture (RFC 9334) for remote attestation — but nothing binds inference providers to use it.


**1. The Akerlof spine is a placeholder but it's your thesis.** Develop the lemons logic explicitly: quality is unobservable → honest providers can't charge a premium for full-precision serving → adverse selection pushes them out or down. Then your verification section maps cleanly onto Akerlof's classic remedies — signaling (quantization disclosure), certification (K2VV, Artificial Analysis), warranties (none exist!), brands (closed labs). That last one is a nice line: OpenAI/Anthropic are the "brand" solution to the lemons problem, and it's the only reason the closed API market functions at all.

**2. No quality SLAs exist anywhere.** Providers contract on uptime and latency, never accuracy — because quality is unverifiable, contracts are incomplete. One paragraph, strengthens the market-failure argument.

**3. Temporal shading.** Everything you list is static substitution. Degrading under peak load (quantize harder at 6pm, restore at midnight) is strictly harder to catch with point-in-time audits. The Anthropic Sept 2025 degradation episode is a perfect case study: users accused them of load-based quality shading, the postmortem showed serving bugs, and Anthropic had to publicly state "we never degrade quality due to demand" — demonstrating that bugs and shading are _indistinguishable from outside_, which is exactly your point about incompetence (#8) vs. malice.

**4. The counterargument.** Why hasn't the market fully lemon-ed? Repeated games with sophisticated buyers — big customers (Cursor, Perplexity-as-buyer) run continuous evals and switch instantly, OpenRouter deranking is expensive, and the audit papers themselves are the market producing discipline. Addressing this makes the piece credible rather than doomer.

**5. A "what should you actually do" close.** Right now it ends on taxonomy. Practical takeaway: canary evals on your own workload, pin providers via quantization filters, weight tool-call benchmarks over MMLU, prefer TEE endpoints where latency allows.

Small things: "perado frontier" → Pareto; the Perplexity/Stanford story you want to insert is presumably Gao et al. (Perplexity failed all model-equality tests — good cold open); and you have ~6 inline "where is this from / needs citation" notes to resolve — the FP4 quality claim and the serving-multiplier numbers (23x batching, 2–4x PagedAttention) especially, since those get scrutinized.