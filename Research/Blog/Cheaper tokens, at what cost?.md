![[cheaper-tokens-cover.png]]

> "Virtually every commercial transaction has within itself an element of trust." — Kenneth Arrow

When a market has an information asymmetry, where suppliers can observe product quality and buyers can't, buyers purchase at the expected quality of the pool. They don't know better. Sellers of above-average goods can't get paid for the difference, so they exit; the pool degrades, buyers rationally mark down further, and the equilibrium unravels toward only 'lemons' trading. The mechanism requires no bad actors, only an information asymmetry and a pooled price. This was George Akerlof's insight in his seminal 1970 paper "The Market for 'Lemons'" and the catalyst for the field of information economics.

Third-party LLM inference often satisfies both conditions. Open weights collapse the model layer into a commodity: within weeks of DeepSeek V4 Pro's release, five of six providers converged on an identical $2.17/M blended price. And the asymmetry is structural: token buyers receive tokens but aren't privy to the computations beyond what providers claim.

Competition among providers serving supposedly "identical" weights runs on two observable metrics: cost and speed. [Artificial Analysis](https://artificialanalysis.ai/models/deepseek-v4-pro/providers) benchmarks every major endpoint on $/M tokens, throughput, and time-to-first-token, and providers [advertise their positions](https://fireworks.ai/blog/blazing-fast-inference-on-top-oss-models) on that frontier. Indeed, many of the methods that allow providers to reach the fast, cheap end of that frontier involve degrading fidelity. Quality verification is left to the buyer, must be repeated as serving stacks change, and so almost no one does it. Token buyers must hope they got the correct model.

Historically, the concern has not been hypothetical. [Gao et al. (ICLR 2025)](https://arxiv.org/abs/2410.20247) formalized the audit problem as *Model Equality Testing* and created a two-sample test that compares an API's output distribution against the reference weights, cheap enough to run for under a dollar per endpoint. Applied to 31 commercial endpoints serving four Llama models across nine providers in Summer 2024, the test found that 11 of 31 deviated statistically from Meta's reference weights. The worst case was [Perplexity](https://x.com/perplexity_ai?lang=en), whose Llama-3 8B endpoint's outputs were further from the reference distribution than an entirely different model's. Only two of nine providers disclosed any distribution-altering optimizations.

Similarly, in October 2025, [Moonshot](https://github.com/MoonshotAI/K2-Vendor-Verifier) found that on 4,000 tool-call prompts scored against the official API, third-party endpoints ranged from 100% schema conformance down to [50.6%](https://medium.com/@kimi_moonshot/announcing-the-k2-vendor-verifier-ensuring-consistent-toolcall-performance-for-kimi-k2-04c568f4a1dd).

So are we going to be left with lemons 🍋?

Akerlof predicts the degradation of such markets, but it also enumerates remedies: disclosure, reputation, certification, and technologies that make quality observable. All four now exist for model inference in embryonic form.

This post covers how & why serving costs can fall, which of the optimizations behind that decline are output-preserving and which are not, and whether the emerging verification layer is even needed.

## "LLMflation"

![[cheaper-tokens-llmflation.png]]

A [MIT FutureTech estimate](https://arxiv.org/abs/2511.23455) found that, between April 2024 and November 2025, the cost of constant-quality inference plummeted at ~5-10x per year, of which ~3x is attributable to algorithmic efficiency. The findings also show that higher-performance models dropped almost 32x per year, with less performant models only around 1.7x per year. An a16z blog by Guido Appenzeller aptly dubbed this ["LLMflation"](https://a16z.com/llmflation-llm-inference-cost/).

Interestingly, the study also found that the cost of running a model with frontier capabilities is simultaneously rising 3-18x per year. So the industry is seeing increasing costs to push frontier capabilities, but falling costs to serve those behind that frontier.

![[cheaper-tokens-frontier-costs.png|443]]

Note that these estimates hold capability fixed, i.e., progress against a benchmark like GPQA-D, not any individual model.

On a per-model basis, we see predictable behavior expected from a contestable hosting market, but not a purely smooth decline. We reconstructed the cheapest listed OpenRouter output price across the platform's 873 model listings from Sept. 2024 to July 2026. Of those, 172 could be matched by exact slug to Artificial Analysis metadata identifying open-weight status and intelligence scores (94 open-weight, 78 closed-weight). The matched sample likely skews toward prominent, benchmarked models, so the results should be read as applying to models we could classify, not all of OpenRouter. To make models with uneven pricing histories comparable, we converted the data into weekly snapshots using each model's latest observed price each Sunday.

For each model we track two floors: the cheapest price available right now, and the cheapest price ever observed. Our analysis found that open-weight models ended the week at a different output price in 10.9% of observed model-weeks, compared with 0.3% for closed models. For open models, the all-time floor fell by 42% per year (95% confidence interval: a 29%-53% decline), while the current floor increased at a pooled annualized rate of 49%. We attribute the contradiction to providers who enter and establish new floors, then exit or raise prices, causing the current minimum to rebound. Indeed, among open models observed for at least 90 days, 81% reached a lower paid floor, the median total decline was 45%, and the median model's own annualized floor slope was -25%.

![[cheaper-tokens-price-floors.png]]

One plausible story is that providers undercut, perhaps below cost, to claw together market share, causing the observed cascading price drops. If the low costs are unsustainable, a participant exits, decreasing competition and alleviating pressure. Additionally, serving different models on the same hardware introduces cold-starts and makes serving inference costlier, so providers need to be selective with which markets they contest, and abandoning one is cheap. Unfortunately, the data doesn't reveal the actual reason behind these price drops, nor for the rebounds. Finer data, perhaps a full record of a model's providers and their pricing at a given time, is needed to get a better estimate in future work.

But if competition is this fierce, what keeps participants faithful? And which of their cost-saving levers touch the qualities the buyers don't measure?

## Sneaky degradation

Most of these "levers" are some of the most important improvements in the industry. They're attributable to the cost of constant-quality inference dropping ~5-10x annually and many are responsible for pushing the Pareto frontier.

Some non-exhaustive notable examples:

- **Architecture:** [DeepSeek-V3](https://arxiv.org/abs/2412.19437)'s [MoE sparsity](https://arxiv.org/abs/1701.06538) means you only pay for active parameters, 37B of its 671B total. Its [MLA](https://arxiv.org/abs/2405.04434) compresses the KV cache, [~70 KB per token versus ~516 KB for Llama-3.1-405B](https://arxiv.org/abs/2505.09343).
- **Distillation:** frontier capabilities get [compressed](https://arxiv.org/abs/1503.02531) into cheaper mini/flash tiers. [Gemini 3 Flash](https://artificialanalysis.ai/articles/gemini-3-flash-everything-you-need-to-know) launched at half the price of 3 Pro while giving up just 2 points on Artificial Analysis's Intelligence Index.
- **Serving software:** [quantization](https://arxiv.org/abs/2306.00978), [FlashAttention](https://arxiv.org/abs/2205.14135), [continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference), [PagedAttention](https://arxiv.org/abs/2309.06180), [prefix caching](https://arxiv.org/abs/2312.07104), and [speculative decoding](https://arxiv.org/abs/2211.17192). On [InferenceMAX](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference), [NVIDIA reported](https://blogs.nvidia.com/blog/blackwell-inferencemax-benchmark-results/) B200 cost on gpt-oss-120b falling from $0.11 to $0.02 per million tokens in two months, purely from software.
- **Hardware:** H100 to B200 is roughly 4x decode throughput like-for-like, netting ~3x cheaper per million tokens. Meanwhile H100 rentals fell from ~$8/hr in 2023 to $1.20-3.50 today ([How the GPU Bubble Burst](https://www.latent.space/p/gpu-bubble)).

All of these techniques are legitimate and widely used. For many of them, there have been no historical mentions of their misuse. The problem is that they can be applied discreetly.

### Silent quantization

If you serve weights at INT4 or FP4 instead of BF16, each replica needs a quarter of the memory, which means more replicas per node, larger batches, and a materially cheaper token.

[Wu et al. (ICML 2023)](https://proceedings.mlr.press/v202/wu23k.html) found that INT4 causes a significant accuracy drop specifically for decoder-only models, and practitioner consensus is to keep INT4 away from math, code, and reasoning-heavy workloads, where the loss concentrates.

[Egashira et al. (NeurIPS 2024)](https://arxiv.org/abs/2405.18137) showed that quantization can activate adversarial behavior that is absent in the full-precision model, meaning a quantized endpoint can be behaviorally different.

Has this been found in practice? In the aforementioned [Gao et al. (ICLR 2025)](https://arxiv.org/abs/2410.20247), the paper that found 11 of 31 commercial Llama endpoints statistically deviated from reference weights, and while the cause is unknown, failures concentrated on knowledge-intensive tasks where quantization causes the most degradations. Perplexity produced suspiciously low-entropy outputs suggesting caching or a broken temperature parameter rather than low precision.

Importantly, quantization is a widely disclosed practice, used by model creators and providers alike. OpenRouter requires providers to declare precision in their models endpoint (valid values: int4, int8, fp4, fp6, fp8, fp16, bf16, fp32) as a [condition of listing](https://openrouter.ai/docs/guides/get-started/for-providers), and exposes a [filter](https://openrouter.ai/docs/features/provider-routing#quantization) so buyers can restrict routing to precisions they trust. Most models are trained for it: DeepSeek V4 was [trained largely at FP4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), so an FP4 endpoint is far less of a downgrade than the label suggests. And, in [OpenRouter and Artificial Analysis's joint benchmarks](https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/) on DeepSeek 3.1, DeepInfra, the only provider quantizing to FP4, remained competitive with full-precision hosts. Hence, quantization is only a villain if undisclosed.

### Context window truncation

Another technique could be to cap or truncate context far below the advertised window. A single 1M-token request on a model without aggressive compression can eat hundreds of gigabytes and is a massive bottleneck. A trim means that the model doesn't process part of the prompt and, unbeknownst to the buyer, the model just performs badly for longer sessions.

![[cheaper-tokens-deepinfra-context.png|400]]

*[DeepInfra's API recommendations for DeepSeek V4 Pro (Max)](https://deepinfra.com/blog/deepseek-v4-pro-max-api-benchmarks-latency-throughput-cost)*

In practice, the documented case is pooled-price divergence. DeepInfra served DeepSeek V4 Pro at a 66K context window while Fireworks, Novita, SiliconFlow, and the official API offered the full 1M, all tied at the same $2.17/M blended price ([DeepInfra's own benchmark page](https://deepinfra.com/blog/deepseek-v4-pro-max-api-benchmarks-latency-throughput-cost)). This was disclosed, but shows how much providers can vary their specs along the Pareto frontier to compete.

Less flattering was a [Hugging Face forum dispute](https://discuss.huggingface.co/t/deepinfra-deepseek-v4-pro-context-size-wrong-on-huggingface/176578) where their listing claimed 1024K while the live endpoint rejected anything past 64K. Likely a metadata bug, but at the cost of buyers.

### KV cache quantization & eviction

Providers can also further quantize the cache or evict "unimportant" tokens mid-generation ([H2O](https://arxiv.org/abs/2306.14048) keeps only heavy-hitter tokens, [StreamingLLM](https://arxiv.org/abs/2309.17453) a sliding window) to fit larger batches. The model can accept the advertised context length, but the quality is eroded and the buyer experiences greater context rot. Unfortunately, cheap short-prompt statistical tests can't catch a cache policy that only activates for long-context windows.

### Lenient speculative decoding

Standard speculative decoding is distribution-identical: a small draft model proposes tokens and the target model verifies them, and as a result you get a speed increase. But, the acceptance rule is relaxable ("typical acceptance," [Medusa](https://arxiv.org/abs/2401.10774)). It can give providers a continuous dial between free speedup and quietly serving a smaller model, invisible per-response.

### Undisclosed system prompts, fine-tuning, watermarking

A provider can also change what the model does without touching how it's served: prepend a hidden system prompt, fine-tune for bias, or watermark outputs. For example, a malicious actor could alter the model so that a request generates more tokens, milking unaware buyers.

### Incompetence

The most likely is plain incompetence: buggy tool-call parsers, wrong chat templates, bad sampler defaults, engine version drift.

As mentioned earlier, [Moonshot](https://github.com/MoonshotAI/K2-Vendor-Verifier) measured Together at 72% and Nebius at 50.6% tool-call schema accuracy against the official API's 100%, a [vLLM postmortem](https://blog.vllm.ai/2025/10/28/Kimi-K2-Accuracy.html) traced most failures to brittle tool-call parsing and chat template bugs, and [OpenRouter's telemetry](https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/) across billions of requests confirmed the variance.

![[cheaper-tokens-gpt-oss-aime.jpeg]]

After the gpt-oss-120b launch in August 2025, Artificial Analysis benchmarked the model across eleven providers in week one and AIME scores ranged from 80% to 93.3% among providers claiming to serve the model faithfully ([Simon Willison's writeup](https://simonwillison.net/2025/Aug/15/inconsistent-performance/)). Azure's 80% traced to an old vLLM commit that ignored reasoning effort, silently defaulting requests to medium. Notably, within days of the results publishing, Azure and Groq had both fixed their stacks and converged to 93.3%.

Competition is fierce, so providers rush to serve the newest model while taking shortcuts on the qualities that buyers can't see.

## Current solutions

The search space for possible degradations is massive, and as with every innovation, there usually lies a way to abuse it. The current state of inference struggles to scratch the surface for verification and the result is "black box APIs" that rely on the trust of the providers.

For example, Hugging Face's [Inference Providers router](https://huggingface.co/docs/inference-providers/en/hub-api) compares hosts of DeepSeek V4 Pro on eight fields: status, context length, price, tool support, structured output support, latency, throughput, and whether the provider is the model author. Context length and price are [self-reported by the provider's own API](https://huggingface.co/docs/inference-providers/en/register-as-a-provider). HF's [automated checks](https://huggingface.co/docs/inference-providers/en/register-as-a-provider) verify the endpoint responds, meets latency thresholds, and handles tool calls. All other dimensions live in each provider's own docs, if anywhere.

The good news is there have been attempts to fix this, each contributing Akerlof's solutions at various degrees: disclosure, reputation, certification, and technology.

> "Since the release of K2 Thinking, we have received frequent feedback from the community regarding anomalies in benchmark scores. Our investigation confirmed that a significant portion of these cases stemmed from the misuse of Decoding parameters … more subtle anomalies soon triggered our alarm. In a specific evaluation on [LiveBenchmark](https://www.reddit.com/r/LocalLLaMA/comments/1osglws/kimi_k2_thinking_scores_lower_than_gemini_25/?rdt=41412), we observed a stark contrast between third-party API and official API. After extensive testing of various infrastructure providers, we found this difference is widespread.
>
> This exposed a deeper problem in the open-source model ecosystem: The more open the weights are, and the more diverse the deployment channels become, the less controllable the quality becomes.
>
> If users cannot distinguish between "model capability defects" and "engineering implementation deviations," **trust in the open-source ecosystem will inevitably collapse.**"
> — Moonshot, [Rebuilding the "Chain of Trust": Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier)

As Moonshot put it, **"making a model open is only half the battle."** After their investigation they built the Kimi Vendor Verifier list, a ~4,000-prompt conformance suite scoring every host against the official API.

After the scare, OpenRouter shipped [Exacto](https://openrouter.ai/blog/announcements/provider-variance-introducing-exacto/) (Oct 2025), additionally routing tool-calling traffic to providers based on measured accuracy. By March 2026 they upgraded it to Auto Exacto: providers are re-scored roughly every five minutes on live tool-call telemetry and benchmarks, and statistical outliers get deranked. They found that it cut gpt-oss-120b error rates from 5.6% to 3.5%. Moreover, OpenRouter's customers put trust in their list of permissioned providers, both the provider's and router's reputations are on the line.

The deepest remedy, however, is technology that makes the computation itself checkable.

[TOPLOC](https://arxiv.org/abs/2501.16007) (Prime Intellect) commits a 258-byte locality-sensitive hash of top-k activations per 32 tokens, roughly 1000x smaller than the raw activations, and detected 100% of model, prompt, and precision substitutions in their evals while staying robust to GPU nondeterminism. [SVIP](https://arxiv.org/abs/2410.22307) verifies via a hidden-state proxy task at under 5% false negatives and 3% false positives. [Ambient's proof-of-logits](https://ambient.xyz/) and [Hyperbolic's proof-of-sampling](https://arxiv.org/abs/2405.00295) are variations of a similar idea.

For provability, [NVIDIA Confidential Computing](https://www.nvidia.com/en-us/data-center/solutions/confidential-computing/) turns H100s and later into trusted execution environments (TEEs): paired with [Intel TDX](https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/overview.html) or [AMD SEV-SNP](https://www.amd.com/en/developer/sev.html) and remote attestation, a buyer can cryptographically verify the stack that served them, down to a hash of the loaded weights. A [third-party benchmark study](https://arxiv.org/abs/2409.03992) (io.net-funded, not NVIDIA's) measured under 7% throughput overhead for typical queries and near zero for large models and long sequences, since the bottleneck is CPU-GPU PCIe encryption rather than compute. Azure confidential GPU, Near AI, Phala, and Atoma serve on it in production.

The holy grail is zkML, full cryptographic proof of inference. I have some scar tissue here: back in 2023 I built [zkMNIST-Noir](https://github.com/alexjaniak/zkMNIST-Noir), proving MNIST digit classification using a small neural net. At the time, for a 269K-parameter dense network, circuit constraints meant proving only the final layer and hand-rolling fixed-point arithmetic because the language had no signed integers. The field has come far since: [zkLLM](https://arxiv.org/abs/2404.16109) (CCS 2024) proves a complete 2048-token LLaMA-2-13B inference in about 13 minutes on an A100, with 1-3 second verification and a sub-200 kB proof. Still orders of magnitude too slow for production serving.

## Does anyone actually want this?

The issue is that we don't fully know. We also aren't sure if this is actually an issue, so we definitely aren't pointing fingers. It is, however, foolish to completely ignore it as a potential failure-mode to inference markets.

If buyers can't detect degradation, they don't know they are paying for lemons, and therefore there is no demand for better verification. Degradation in a credence good only gets demanded away if the buyers can perceive it.

The evidence is also ambiguous. Even after Gao et al. found a third of audited endpoints deviating, most provider traffic still remains routed on price and speed. However, failed tool calls have caused complaints, investigations, and subsequent rolling, albeit minimal, verifications to keep vendors in-line. Inference marketplaces are likely the best informed on buyer demand, but with tool calls failing much louder than misinformed writing or marginally worse code, perhaps the buyers have no clue.

The clearest demand signal would come from place most likely to produce *lemons*: permissionless and decentralized inference. Prime Intellect built TOPLOC for exactly this, Ambient bakes [proof-of-logits](https://ambient.xyz/) into consensus, Hyperbolic runs [proof-of-sampling](https://arxiv.org/abs/2405.00295), and Phala, Atoma, and Near AI serve from attested TEEs. Unsurprisingly, nearly all of these companies have had some ties to Crypto, the ultimate lemon factory with origins aimed at preventing lemon factories. Hopefully, if the centralized inference markets do ever unravel, the decentralized counterparts will have evolved from their embryonic form and primed to accept both buyers and providers. For now, decentralized inference remains relatively illiquid.

Lastly, the above analysis applies primarily to open weights. Closed labs have even more opaque practices, but have a greater reputational stake and no competitive market trying to undercut their inference. So hopefully inference markets don't produce lemons, and if they do, we can catch it.

---

*Originally published on [Substack](https://collectgarbage.substack.com/p/cheaper-tokens-at-what-cost).*
