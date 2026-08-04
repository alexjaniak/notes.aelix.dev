# Inference Economics

Research dump (July 2026) answering: *Why is cost to serve models decreasing so much if GPU prices aren't dumping? How do people know they're getting served the right model?* From [[Questions]]. Related: [[Serving Models]], [[Inference]], [[Frontier Chips]]. Published as [[Cheaper tokens, at what cost?]]. All citations verified July 2026.

## The framing: a market for lemons

- Akerlof's 1970 "The Market for 'Lemons'": when suppliers can observe quality and buyers can't, buyers pay the pool's expected quality → above-average sellers exit → pool degrades → equilibrium unravels toward only lemons trading. **Requires no bad actors** — just information asymmetry plus a pooled price.
- Third-party LLM inference often satisfies both conditions:
	- **Pooled price:** open weights commoditize the model layer — five of six providers converged on an identical $2.17/M blended price within weeks of DeepSeek V4 Pro's release.
	- **Asymmetry:** buyers receive tokens but aren't privy to the computations beyond what providers claim. Competition runs on the two observable metrics — cost and speed ([Artificial Analysis](https://artificialanalysis.ai/models/deepseek-v4-pro/providers) benchmarks $/M, throughput, TTFT; providers [advertise positions](https://fireworks.ai/blog/blazing-fast-inference-on-top-oss-models) on that frontier) — while many methods that reach the fast/cheap end degrade fidelity invisibly.
- Quality verification is left to the buyer, must be repeated as serving stacks change, so almost no one does it.
- Akerlof also enumerates the remedies: **disclosure, reputation, certification, and technology that makes quality observable** — all four now exist for inference in embryonic form (see [[#Verification & trust]]).

## Why is cost-to-serve dropping ~10x/year?

### Headline numbers

- Constant-quality inference cost falls **~10x/year** ([a16z "LLMflation"](https://a16z.com/llmflation-llm-inference-cost/), Guido Appenzeller, Nov 2024): GPT-3-level output (MMLU 42) cost $60/M tokens in Nov 2021 → $0.06/M (Llama 3.2 3B on Together) in late 2024 = 1,000x in 3 years.
- [Epoch AI](https://epoch.ai/data-insights/llm-inference-price-trends) measures 9–900x/yr depending on the benchmark milestone (GPT-4-level GPQA fell 40x/yr), with the fastest drops in the most recent year.
- [Stanford HAI AI Index 2025](https://hai.stanford.edu/ai-index/2025-ai-index-report): GPT-3.5-level (MMLU 64.8) fell $20.00/M (Nov 2022) → $0.07/M (Oct 2024), more than 280-fold.
- Most careful: MIT FutureTech ([*The Price of Progress*](https://arxiv.org/abs/2511.23455), Gundlach et al.): **5–10x/yr** at the benchmark level (Apr 2024–Nov 2025), of which **~3x/yr is pure algorithmic efficiency**. Higher-performance models dropped almost 32x/yr; less performant ones only ~1.7x/yr.
- **Crucial counterpoint** from the same paper: the cost of running the *frontier* model is simultaneously **rising 3–18x/yr** (bigger models + longer reasoning traces) — cheapness is at fixed capability, not at the frontier. These estimates hold capability fixed (progress against a benchmark like GPQA-D), not any individual model.

### The factor stack — and why "GPU prices" is the wrong lens

A 2026 econometric decomposition ([*Tiered Super-Moore's Law*](https://arxiv.org/abs/2603.28576), using OpenRouter + Epoch data) attributes essentially all of the ~600x 2020–2026 decline to software/algorithms (TFP residual ≈ 103.7% of cost reduction) and ~0% to GPU hardware (−0.9%):

1. **Algorithmic efficiency (~3x/yr alone)** — distillation of frontier models into mini/flash tiers (probably the single biggest driver; [Gemini 3 Flash](https://artificialanalysis.ai/articles/gemini-3-flash-everything-you-need-to-know) launched at half the price of 3 Pro while giving up just 2 points on AA's Intelligence Index); MoE sparsity ([DeepSeek-V3](https://arxiv.org/abs/2412.19437): 671B total / 37B active — you pay for active params only); **MLA** compressing the KV cache ~7.3x vs Llama-3.1-405B (70 KB/token vs 516 KB, per DeepSeek's [hardware-reflections paper](https://arxiv.org/abs/2505.09343) — the cheap-long-context unlock, see [[Inference]]); quantization FP16→FP8→FP4 (Llama-70B at NVFP4 ≈ 43 GB at ~98–99% of BF16 quality)
2. **Serving software multipliers** (mostly quality-neutral): continuous batching (up to ~23x), PagedAttention (2–4x), prefix caching (up to 5x), speculative decoding (2–3x, output-identical), prefill/decode disaggregation (+59–498% goodput) — see [[Serving Models]]. On SemiAnalysis's [InferenceMAX](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference) benchmark, [NVIDIA reported](https://blogs.nvidia.com/blog/blackwell-inferencemax-benchmark-results/) B200 cost/M tokens on gpt-oss-120b falling $0.11 → $0.02 in two months *purely from software* (TensorRT-LLM updates)
3. **Hardware generations** (real but smaller than headlines): H100→B200 like-for-like ≈ 4x decode throughput ≈ ~3x cheaper/M tokens on-demand; NVIDIA's "25–30x" claims are rack-scale GB200-vs-H100-BF16 comparisons
4. **GPU *rental* prices did crash even though chip ASPs didn't**: H100 $8/hr (2023) → ~$2.85 (early 2024) → $1.20–3.50/hr (Q2 2026) — see [*$2 H100s: How the GPU Bubble Burst*](https://www.latent.space/p/gpu-bubble) (Eugene Cheah, Oct 2024). The right unit is tokens/$/GPU-hour, not chip sticker price
5. **Utilization**: batching amortizes one weight-load across many users; interactivity (tok/s/user) is the dial providers turn between margin and UX
6. **Prompt caching as pricing**: cache reads at 10% of input price ([Anthropic](https://platform.claude.com/docs/en/build-with-claude/prompt-caching); [DeepSeek](https://api-docs.deepseek.com/news/news0802/) started at 10% in Aug 2024, now ~2% on V4 Flash) — agentic workloads have high hit ratios, so *effective* $/token fell faster than list prices
7. **Competition**: open weights commoditize the model layer — any capability appearing in open weights gets instant multi-provider price war ([five of six providers converged at $2.17/M blended](https://deepinfra.com/blog/deepseek-v4-pro-max-api-benchmarks-latency-throughput-cost) for DeepSeek V4 Pro Max). DeepSeek then made its 75% cut [permanent](https://api-docs.deepseek.com/quick_start/pricing/) (May 2026: $0.435/$0.87 per M, 1M context)

### Per-model price dynamics (own analysis, from the blog)

Reconstructed the cheapest listed OpenRouter output price across 873 model listings, Sept 2024 – July 2026; 172 matched by exact slug to Artificial Analysis metadata (94 open-weight, 78 closed-weight). Weekly snapshots (latest observed price each Sunday). Caveat: the matched sample skews toward prominent, benchmarked models.

- **Repricing frequency:** open-weight models ended the week at a different output price in **10.9%** of observed model-weeks vs **0.3%** for closed models.
- **Two floors per model:** the all-time paid floor **fell 42%/yr** (95% CI: 29–53% decline) while the *current* floor **rose at a pooled annualized 49%**.
- **Sawtooth hypothesis:** providers enter and establish new floors, then exit or raise prices, so the current minimum rebounds. Among open models observed ≥90 days: 81% reached a lower paid floor, median total decline 45%, median model's own annualized floor slope −25%.
- **Plausible mechanism:** providers undercut (perhaps below cost) to claw market share; unsustainable lows → exit → pressure alleviates. Serving different models on the same hardware introduces cold-starts, so providers are selective about which markets they contest, and abandoning one is cheap.
- **Open question:** the data doesn't reveal the actual reason for drops or rebounds. Needs a full record of each model's providers and their prices over time.

### Is it below cost?

- Mostly not anymore. DeepSeek [disclosed](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) (Feb 2025 data; [Reuters](https://www.reuters.com/technology/chinas-deepseek-claims-theoretical-cost-profit-ratio-545-per-day-2025-03-01/)) a theoretical 545% cost-profit ratio = **84.5% gross margin** at R1 list prices ($87k/day cost vs $562k theoretical revenue).
- [SemiAnalysis](https://newsletter.semianalysis.com/p/anthropic-growth-and-bedrock-mix) estimates Anthropic inference gross margins went **−94% (2024) → 38% (2025) → mid-60s (2026)**.
- Loss-leader pricing was real in 2023–24; the subsidy today lives in subscriptions (Max/Pro plans deliver far above list-price token value) and free tiers, not the API.

**One-line answer:** GPU chips aren't cheaper, but everything around them is — ~3x/yr algorithms, big software-serving multipliers, hardware $/perf gains plus collapsed rental prices, and open-weights competition compressing margins — compounding to ~10x/yr at fixed quality.

## The degradation levers

The same techniques driving the cost decline can be applied discreetly. All are legitimate and widely used; the problem is disclosure, not the techniques.

### Silent quantization

- Serving INT4/FP4 instead of BF16 quarters replica memory → more replicas per node, larger batches, materially cheaper tokens.
- [Wu et al. (ICML 2023)](https://proceedings.mlr.press/v202/wu23k.html): INT4 causes a significant accuracy drop *specifically for decoder-only models*; practitioner consensus keeps INT4 away from math/code/reasoning workloads where the loss concentrates.
- [Egashira et al. (NeurIPS 2024)](https://arxiv.org/abs/2405.18137): quantization can *activate adversarial behavior absent in the full-precision model* — a quantized endpoint can be behaviorally different, not just noisier.
- In practice: [Gao et al. (ICLR 2025)](https://arxiv.org/abs/2410.20247) found failures concentrated on knowledge-intensive tasks where quantization degrades most, though cause is unknown. Perplexity produced suspiciously *low-entropy* outputs — suggesting caching or a broken temperature parameter rather than low precision.
- **Not inherently a scam:** OpenRouter requires precision declaration (int4/int8/fp4/fp6/fp8/fp16/bf16/fp32) as a [condition of listing](https://openrouter.ai/docs/guides/get-started/for-providers) with a [buyer-side filter](https://openrouter.ai/docs/features/provider-routing#quantization); DeepSeek V4 was [trained largely at FP4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) so an FP4 endpoint is far less of a downgrade than the label suggests; in [OpenRouter × AA joint benchmarks](https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/) on DeepSeek 3.1, DeepInfra (only FP4 host) stayed competitive with full-precision hosts. Quantization is only a villain if undisclosed.

### Context window truncation

- Cap or truncate context far below the advertised window: a 1M-token request without aggressive compression can eat hundreds of GB. A trim means the model never processes part of the prompt — to the buyer it just looks like the model performing badly in longer sessions.
- Documented case is pooled-price *divergence*, not silent truncation: DeepInfra served DeepSeek V4 Pro at a 66K window while Fireworks, Novita, SiliconFlow, and the official API offered the full 1M — all at the same $2.17/M blended price ([DeepInfra's own benchmarks](https://deepinfra.com/blog/deepseek-v4-pro-max-api-benchmarks-latency-throughput-cost)). Disclosed, but shows how much specs can vary along the Pareto frontier at one price.
- Less flattering: a [Hugging Face forum dispute](https://discuss.huggingface.co/t/deepinfra-deepseek-v4-pro-context-size-wrong-on-huggingface/176578) where DeepInfra's listing claimed 1024K while the live endpoint rejected anything past 64K. Likely a metadata bug — but from the buyer's side, bug and deception are indistinguishable.

### KV cache quantization & eviction

- Quantize the cache or evict "unimportant" tokens mid-generation to fit larger batches: [H2O](https://arxiv.org/abs/2306.14048) keeps only heavy-hitter tokens, [StreamingLLM](https://arxiv.org/abs/2309.17453) a sliding window.
- The endpoint *accepts* the advertised context length but quality erodes — the buyer just experiences worse context rot.
- **Audit gap:** cheap short-prompt statistical tests can't catch a cache policy that only activates at depth. Published techniques, no documented abuse case — "possible and invisible," not "caught in the wild."

### Lenient speculative decoding

- Standard speculative decoding is distribution-identical (draft proposes, target verifies) — speed for free.
- But the acceptance rule is relaxable ("typical acceptance," [Medusa](https://arxiv.org/abs/2401.10774)): once you accept tokens the target would have rejected, you're partially reading the draft model's output. A **continuous dial** between free speedup and quietly serving a smaller model, invisible per-response.

### Undisclosed system prompts, fine-tuning, watermarking

- Change what the model *does* without touching how it's served: hidden system prompt, fine-tune for bias, watermark outputs.
- Includes an incentive attack: alter the model so requests generate *more tokens*, milking unaware buyers.

### Incompetence (the most likely in practice)

- Buggy tool-call parsers, wrong chat templates, bad sampler defaults, engine version drift — empirically outweighs deliberate degradation.
- [Moonshot's K2 Vendor Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier): Together 72%, Nebius 50.6% tool-call schema accuracy vs the official API's 100%; a [vLLM postmortem](https://blog.vllm.ai/2025/10/28/Kimi-K2-Accuracy.html) traced most failures to brittle tool-call parsing and chat-template bugs; [OpenRouter telemetry](https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/) across billions of requests confirmed the variance.
- gpt-oss-120b launch (Aug 2025): AA benchmarked eleven providers in week one; AIME ranged 80%→93.3% among providers claiming faithful serving (36.7% including the worst; [Simon Willison's writeup](https://simonwillison.net/2025/Aug/15/inconsistent-performance/)). **Azure's 80% traced to an old vLLM commit that ignored reasoning effort, silently defaulting to medium.** Within days of publication, Azure and Groq both fixed their stacks and converged to 93.3% — evidence that *published measurement disciplines providers*.
- Structural cause: converged pricing pushes differentiation into the one dimension buyers can't observe, and corners get cut there — deliberately or not.

## How do you know you're getting the right model?

### The trust problem is real and measured

- **[Model Equality Testing](https://arxiv.org/abs/2410.20247)** (Gao, Liang, Guestrin — Stanford, ICLR 2025): statistical two-sample test on outputs, <$1 per audit. Applied to 31 commercial Llama endpoints across 9 providers (Summer 2024): **11 of 31 statistically differed from Meta's reference weights**. Perplexity failed all tests; some deviations were larger than substituting a *different model entirely*. Only 2 of 9 providers disclosed their optimizations.
- **[Kimi K2 Vendor Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier)** (Moonshot): official-vs-vendor comparison on ~4,000 tool-call prompts — official/Fireworks/DeepInfra at 100% schema accuracy, Together ~72%, Nebius at 50.6% tool-call *trigger similarity* (Nov 2025 snapshot; the table updates). "Making a model open is only half the battle." Expanded into the full [Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier) after K2 Thinking benchmark anomalies; their framing: *"If users cannot distinguish between 'model capability defects' and 'engineering implementation deviations,' trust in the open-source ecosystem will inevitably collapse."*
- **[Artificial Analysis](https://artificialanalysis.ai/models/gpt-oss-120b/providers)** runs per-provider quality benchmarks (GPQA ×16 runs) on the same open model: meaningful spread across providers, tightening over weeks ("burn-in") but never fully.
- **Surprising nuance** (OpenRouter's data): quantization alone often *isn't* the culprit — some FP4 providers beat FP8 ones. The bigger causes are **tool-call parsers, chat templates, sampler defaults, and engine bugs**.
- Degradation concentrates where agents live: free-form text barely moves under quantization, but JSON-mode drops 3–8 pts and tool-argument correctness 4–10 pts at INT4 ([FutureAGI's vLLM eval guide](https://futureagi.com/blog/evaluating-vllm-self-hosted-llm-2026/) — vendor-reported customer-workload numbers, not peer-reviewed).
- **Baseline disclosure is thin:** Hugging Face's [Inference Providers router](https://huggingface.co/docs/inference-providers/en/hub-api) compares hosts on eight fields (status, context length, price, tool support, structured output, latency, throughput, is-model-author); context length and price are [self-reported by the provider's own API](https://huggingface.co/docs/inference-providers/en/register-as-a-provider), and HF's automated checks only verify the endpoint responds, meets latency thresholds, and handles tool calls. Everything else lives in each provider's docs, if anywhere.

### Verification approaches, weakest to strongest

1. **Benchmark/statistical auditing** (black-box, cheap, probabilistic): Artificial Analysis, K2 Vendor Verifier, Model Equality Testing; logprob drift checks vs a self-hosted reference
2. **Market/reputation mechanisms** — what actually protects most users today: OpenRouter **[Exacto](https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/)** (Oct 2025) and **[Auto Exacto](https://openrouter.ai/blog/announcements/auto-exacto/)** (Mar 2026, on by default: re-scores providers roughly every 5 minutes on tool-call telemetry + benchmarks, deranks statistical outliers; cut gpt-oss-120b error rates 5.6%→3.5%); plus a `quantizations` request filter with provider-disclosed precision labels. Both provider and router reputations are on the line — but it only governs traffic through the router; nothing stops degraded direct-to-customer serving. Private regulation, not a standard.
3. **Activation fingerprinting** (research → production): **[TOPLOC](https://arxiv.org/abs/2501.16007)** (Prime Intellect — 258-byte locality-sensitive hash of top-k activations per 32 tokens, ~1000x smaller than raw activations; 100% detection of model/prompt/precision substitution in their evals, robust to GPU nondeterminism), **[SVIP](https://arxiv.org/abs/2410.22307)** (hidden-state proxy-task verification, <5% FNR / <3% FPR), [Ambient proof-of-logits](https://ambient.xyz/), [Hyperbolic proof-of-sampling](https://arxiv.org/abs/2405.00295)
4. **TEE attestation — the practical winner**: [NVIDIA Confidential Computing](https://www.nvidia.com/en-us/data-center/solutions/confidential-computing/) (H100+) with [Intel TDX](https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/overview.html)/[AMD SEV-SNP](https://www.amd.com/en/developer/sev.html) and remote attestation — buyer can cryptographically verify the serving stack down to a hash of the loaded weights; a [third-party benchmark study](https://arxiv.org/abs/2409.03992) (io.net-funded, not NVIDIA's) found **<7% throughput overhead for typical queries, near zero for large models/long sequences** (the bottleneck is CPU–GPU PCIe encryption, not compute). Used by Azure confidential GPU, Near AI, Phala, Atoma. Trust shifts to the chip vendor's keys
5. **zkML — not viable for LLMs yet**: [zkLLM](https://arxiv.org/abs/2404.16109) (CCS 2024) proves an entire 2048-token LLaMA-2-13B inference in under 15 minutes on an A100 (~803s proving, 1–3s verification, <200 kB proof) — impressive research, still orders of magnitude too slow for production serving. Survey: [Equilibrium Labs, *State of Verifiable Inference*](https://equilibrium.co/writing/state-of-verifiable-inference). (Context on how far the field came: proving a 269K-param MNIST net in Noir circa 2023 meant proving only the final layer with hand-rolled fixed-point arithmetic — [zkMNIST-Noir](https://github.com/alexjaniak/zkMNIST-Noir).)

### Does anyone actually want verification? (demand side)

- **The credence-good problem:** if buyers can't detect degradation, they don't know they're buying lemons, so there's no demand for verification. Degradation in a credence good only gets demanded away if buyers can perceive it.
- Evidence is ambiguous: even after Gao et al. found a third of audited endpoints deviating, most traffic still routes on price and speed. But tool-call failures *fail loudly* — they caused complaints, investigations, and rolling (if minimal) verifications — whereas misinformed writing or marginally worse code fails silently.
- **Clearest demand signal comes from where lemons are most likely: permissionless/decentralized inference.** Prime Intellect built TOPLOC for exactly this; Ambient bakes proof-of-logits into consensus; Hyperbolic runs proof-of-sampling; Phala, Atoma, Near AI serve from attested TEEs. Nearly all have crypto ties — "the ultimate lemon factory with origins aimed at preventing lemon factories." Decentralized inference remains relatively illiquid for now.
- Closed labs (OpenAI/Anthropic/Google) are even more opaque, but carry greater reputational stake and face no competitive market undercutting their own inference — the "brand" remedy to the lemons problem.

## Synthesis

- The accuracy cost of efficiency tricks is borne mostly invisibly by users of third-party-hosted open models. **Detection today = independent benchmarks + statistical tests; proof today = TEEs; the pragmatic middle layer = router reputation systems.**
- Provider variance exists *because* of the price competition: converged pricing pushes differentiation into precision/engine choices users can't see, and corners get cut there — deliberately or not. Empirically, incompetence (parsers, templates, samplers, engine drift) outweighs malice.
- Akerlof's remedies map cleanly: disclosure (OpenRouter quantization labels), reputation (Exacto, provider brands), certification (K2VV/KVV, Artificial Analysis), technology (fingerprinting, TEEs, zkML). All embryonic; none mandated.
- Closed labs rest purely on reputation — the same tricks are available to them, and only aggregate evals or TEEs would catch it.
- The market hasn't fully lemon-ed, plausibly because of repeated games with sophisticated buyers: big customers run continuous evals and switch instantly, router deranking is costly, and the audit papers themselves are the market producing discipline.

## Notes & open questions

- What actually drives the per-model price sawtooth? Needs a full record of each model's providers + prices over time, not just the minimum.
- Temporal shading (degrade under peak load, restore off-peak) is strictly harder to catch with point-in-time audits — bugs and shading are indistinguishable from outside.
- No quality SLAs exist anywhere: providers contract on uptime and latency, never accuracy — quality is unverifiable, so contracts stay incomplete.
- Practical buyer playbook: canary evals on your own workload, pin providers via quantization filters, weight tool-call benchmarks over MMLU, prefer TEE endpoints where latency allows.
- Whether long-context degradation (KV eviction) can be audited cheaply remains open — short-prompt tests structurally miss it.
