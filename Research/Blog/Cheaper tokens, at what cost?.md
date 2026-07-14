> "Show me the incentive and I will show you the outcome." — Charlie Munger, 1995

When a market has an information asymmetry, where suppliers can observe product quality and buyers can't, buyers purchase at the expected quality of the pool. They don't know better. Sellers of above-average goods can't get paid for the difference, so they exit; the pool degrades, buyers rationally mark down further, and the equilibrium unravels toward only 'lemons' trading. The mechanism requires no bad actors, only an information asymmetry and a pooled price. This was George Akerlof's insight in his seminal 1970 paper "The Market for 'Lemons'" and the catalyst for the field of information economics.

Third-party LLM inference often satisfies both conditions. Open weights collapse the model layer into a commodity: within weeks of DeepSeek V4 Pro's release, five of six providers converged on an identical $2.17/M blended price. And the asymmetry is structural: token buyers only receive tokens and aren't privy to the computations behind them.

Competition among providers serving supposedly "identical" weights runs on two observable metrics: cost and speed. [Artificial Analysis](https://artificialanalysis.ai/models/deepseek-v4-pro/providers) benchmarks every major endpoint on $/M tokens, throughput, and time-to-first-token, and providers [advertise their positions](https://fireworks.ai/blog/blazing-fast-inference-on-top-oss-models) on that frontier. Unfortunately, many of the methods that allow providers to reach the fast, cheap end of that frontier involve degrading fidelity. Quality verification is left to the buyer, must be repeated as serving stacks change, and so almost no one does it. Token buyers are left hoping they got the model on the label.

Historically, the concern has not been hypothetical either. [Gao et al. (ICLR 2025)](https://arxiv.org/abs/2410.20247) formalized the audit problem as *Model Equality Testing*: a two-sample test that compares an API's output distribution against the reference weights, cheap enough to run for under a dollar per endpoint. Applied to 31 commercial endpoints serving four Llama models across nine providers in Summer 2024, the test found that 11 of 31 deviated statistically from Meta's reference weights. The worst case was [Perplexity](https://x.com/perplexity_ai?lang=en), whose Llama-3 8B endpoint's outputs were further from the reference distribution than an entirely different model's. Only two of nine providers disclosed any distribution-altering optimizations.

Similarly, in October 2025, [Moonshot's K2 Vendor Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier) found that on 4,000 tool-call prompts scored against the official API, third-party endpoints ranged from 100% schema conformance down to [50.6%](https://medium.com/@kimi_moonshot/announcing-the-k2-vendor-verifier-ensuring-consistent-toolcall-performance-for-kimi-k2-04c568f4a1dd).

So are we going to be left with lemons?

Akerlof's paper predicts the degradation of such markets, but it also enumerates remedies: disclosure, reputation, certification, and technologies that make quality observable. All four now exist for model inference in embryonic form.

This post covers why serving costs fell ~10x/year, which of the optimizations behind that decline are output-preserving and which are not, and whether the emerging verification layer is even needed.
## "LLMflation"
![[Screenshot 2026-07-14 at 12.11.41 PM.png|623]]

A [MIT FutureTech estimate](https://arxiv.org/abs/2511.23455) found that, between April 2024 and November 2025, constant-quality inference plummeted at **~5-10x** per year, of which **~3x** is attributable to algorithmic efficiency. The findings also show that higher-performance models dropped almost 32x per year, with less performant models only around 1.7x per year. An a16z blog by Guide Appenzeller aptly dubbed this ["LLMflation"](https://a16z.com/llmflation-llm-inference-cost/).

Interestingly, the study also found that the cost of running a model with frontier capabilities is simultaneously rising 3-18x per year. So the industry is seeing increasing costs to push frontier capabilities, but decreasing costs to serve those behind that frontier. 

Note that these estimates are respective to fixed capabilities, i.e., progress in a benchmark like GPQA-D, not any individual model. 

![[Screenshot 2026-07-14 at 1.03.46 PM.png]]

On a per-model basis, we see predictable behavior expected from a contestable hosting market, but nor a purely smooth decline. We reconstructed the cheapest listed OpenRouter output price for 94 open-weight and 78 closed-weight model varieties and benchmarked on a sample of 172 of 873 price changes from Sept. 2024 to July 2026. To make models with uneven pricing histories comparable, we converted the data into weekly snapshots using each model's latest observed price each Sunday.

Our analysis found that open-weight models ended the week at a different output price in 10.9% of observed model-weeks, compared with 0.3% for closed models. The lowest currently available price for an open-weight model *increased* at a pooled annualized rate of 49%, while its lowest paid price observed to date *fell* by 42% per year (95% confidence interval: a 29%–53% decline). This apparent contradiction reflects a sawtooth hypothesis: cheap providers enter and establish new lows, then exit or raise prices, causing the current minimum to rebound. 

![[Pasted image 20260714155449.png]]

The data unfortunately doesn't reveal the reason behind these price drops, nor for the rebound. Finer data and more analysis, perhaps a collection of all providers of a model and their pricing at a given time, is likely needed to get a better estimate an answer in future work. What we can look at is what structural reasons enable a ~10x/yr fixed-capability cost decrease and ~42%/yr decrease in floor price within a model.
## So why is cost-to-serve plummeting?

The decline partly comes from optimizations across the entire AI stack. A non-exhaustive tour of the big categories:

Models are increasingly designed to be cheap to run, so improvements are being baked into the **architecture**, and [DeepSeek-V3](https://arxiv.org/abs/2412.19437) illustrates that perfectly. Its [MoE sparsity](https://arxiv.org/abs/1701.06538) means you only pay for active parameters, 37B of its 671B total. And its [MLA](https://arxiv.org/abs/2405.04434) compresses the KV cache, ~70 KB per token versus ~516 KB for Llama-3.1-405B ([hardware reflections paper](https://arxiv.org/abs/2505.09343)), which is what made long context economical.

**Distillation** allows frontier capabilities to get [compressed](https://arxiv.org/abs/1503.02531) into cheaper mini/flash tiers, likely the single biggest driver: [Gemini 3 Flash](https://artificialanalysis.ai/articles/gemini-3-flash-everything-you-need-to-know) launched at half the price of 3 Pro while giving up just 2 points on Artificial Analysis's Intelligence Index.

There is also optimizations tuned to **serving software**, most of it being lossless: think kernel optimizations like [FlashAttention](https://arxiv.org/abs/2205.14135), [continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference), [PagedAttention](https://arxiv.org/abs/2309.06180), prefix caching, and speculative decoding. Each change how the weights are served without adjusting the output distribution. The effects can also be drastic: on [InferenceMAX](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference), [NVIDIA reported](https://blogs.nvidia.com/blog/blackwell-inferencemax-benchmark-results/) B200 cost on gpt-oss-120b falling from $0.11 to $0.02 per million tokens in two months, purely from innovations in software. There are also lossy techniques, quantization and KV cache eviction among them, that push the price-speed frontier further at the cost of accuracy. We'll cover these in the following sections.

**Hardware** improvements and pricing can also have a big impact. H100 to B200 is roughly 4x decode throughput like-for-like, netting ~3x cheaper per million tokens. H100s fell from ~$8/hr in 2023 to $1.20-3.50 today ([_How the GPU Bubble Burst_](https://www.latent.space/p/gpu-bubble)).

Most of these improvements are what push the industry forward, but they can also be used by providers to deviate from the labeled model via silent degredation.
### Silent Quantization

The most direct lever a provider has is precision. If you serve weights at INT4 or FP4 instead of BF16, each replica needs a quarter of the memory, which mean more replicas per node, larger batches, and a materially cheaper token.

[Wu et al. (ICML 2023)](https://proceedings.mlr.press/v202/wu23k.html) found that INT4 causes a significant accuracy drop specifically for decoder-only models, and practitioner consensus is to keep INT4 away from math, code, and reasoning-heavy workloads, where the loss concentrates.

[Egashira et al. (NeurIPS 2024)](https://arxiv.org/abs/2405.18137) showed that quantization can activate adversarial behavior that is absent in the full-precision model, meaning a quantized endpoint can be behaviorally different. 

Has this been found in practice? In [Gao et al. (ICLR 2025)](https://arxiv.org/abs/2410.20247), the paper that found 11 of 31 commercial Llama endpoints statistically deviated from reference weights, and while the cause is unknown, quantization seems like the most obvious candidate (find evidence).

Importantly, quantization is not inherently a scam. It is a perfectly disclosed legitimate practice. Open Router requires labels for (find evidence). Some models are being trained for it: DeepSeek V4 was QAT-trained largely at FP4, so an FP4 endpoint is far less of a downgrade than the label suggests. Additionally, OpenRouter's telemetry shows that some FP4 providrs outscore FP8 ones. 
### Context Window Truncation

Another technique is to cap or silently truncate context far below the advertised window. A single 1M-token request on a model without aggressive compression can eat hundreds of gigabytes and is a huge bottleneck. A trim means that the model next sees part of the prompt and, unbeknownst to the buyer, it appears as context rot or hallucinations. 

In practice, the documented case is less silent truncation than pooled-price divergence. DeepInfra served DeepSeek V4 Pro at a 66K context window while Fireworks, Novita, SiliconFlow, and the official API offered the full 1M, all tied at the same $2.17/M blended price ([DeepInfra's own benchmark page](https://deepinfra.com/blog/deepseek-v4-pro-max-api-benchmarks-latency-throughput-cost)). To their credit, the cap was disclosed, if only in the fine print. Less flattering was a [Hugging Face forum dispute](https://discuss.huggingface.co/t/deepinfra-deepseek-v4-pro-context-size-wrong-on-huggingface/176578) where their listing claimed 1024K while the live endpoint rejected anything past 64K. Probably a metadata bug rather than deception, but that's the point: from the buyer's side, the two are indistinguishable. Same model name, same price, a 15x difference in what you actually get.
### KV cache quantization & eviction

The cache doesn't have to be stored faithfully either. Providers can quantize it to 4 bits or evict "unimportant" tokens mid-generation ([H2O](https://arxiv.org/abs/2306.14048) keeps only heavy-hitter tokens, [StreamingLLM](https://arxiv.org/abs/2309.17453) a sliding window) to fit larger batches. The model saw your full prompt at prefill, then had its working memory pruned along the way, so recall degrades with depth and it looks like ordinary long-context mediocrity. One caveat: these are legitimate published techniques and there's no documented case of undisclosed provider use. But short-prompt statistical tests can't catch a cache policy that only bites at depth, so the honest framing is possible and invisible, not caught in the wild.
### Lenient Speculative Decoding 

Standard speculative decoding is output-identical: a small draft model proposes tokens and the target model verifies every one, so you get speed for free. But the acceptance rule is relaxable ("typical acceptance," [Medusa](https://arxiv.org/abs/2401.10774)), and once you accept tokens the big model would have rejected, you're partially reading the draft model's output. It's a continuous dial between free speedup and quietly serving a smaller model, invisible per-response. Same caveat as before: published technique, no documented abuse case. Its undetectability is the point.
### Model Substitution & Mixing

The maximal version of everything above: serve a distilled, older, or smaller model under the flagship's name, or mix models across requests to blur the statistical signature. 

This isn't hypothetical distance — some provider deviations in the Stanford audit were larger than substituting a different model entirely, and the [rank-based uniformity test paper](https://arxiv.org/pdf/2506.06975) explicitly treats model mixing as an evasion strategy against exactly these audits. ## Undisclosed System Prompts, Fine-Tuning, Watermarking A provider can also change what the model does without touching how it's served: prepend a hidden system prompt, fine-tune for bias, or watermark outputs. All shift the output distribution while the model name stays the same. This is the explicit threat model of [TOPLOC](https://arxiv.org/pdf/2501.16007) and of Model Equality Testing.
### Incompetence

The biggest one in practice isn't an attack at all: buggy tool-call parsers, wrong chat templates, bad sampler defaults, engine version drift. Empirically it outweighs quantization. Moonshot's [K2 Vendor Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier) measured Together at 72% and Nebius at 50.6% tool-call schema accuracy against the official API's 100%, Chutes' postmortem traced most failures to vLLM/sglang parsing bugs, and [OpenRouter's telemetry](https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/) across billions of requests confirmed the variance. The structural cause is the converged price: when five providers charge an identical $2.17/M, differentiation gets pushed into the one dimension buyers can't observe, and corners get cut there, deliberately or not.
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