## Initial Thesis

- **Lemons market.** The inference market is a lemons market: providers compete on listed price and speed while other qualities go untracked.
- **Benchmarks say parity, buyers flee.** fp4 (4-bit floating point) hosts sell the same model name at a discount and benchmarks certify parity (the [NVIDIA GLM-5.2-NVFP4 card](https://huggingface.co/nvidia/GLM-5.2-NVFP4), NVFP4 being NVIDIA's fp4 format: worst delta 0.81 points, three of five benchmarks higher under fp4), yet production users flee ([Umans killed their fp4 deployment](https://blog.umans.ai/blog/glm-5-2-nvfp4-not-worth-serving/) after a 4-day public trial).
- **Prediction.** fp4 causes degradation where benchmarks don't look, in long-horizon work, so a paired depth-swept audit of what's actually served should expose it.

## Initial Test

- **Setup.** GLM-5.2 on [BABILong](https://arxiv.org/abs/2406.10149) qa3 at 256k context. 20 paired items, [DeepInfra](https://deepinfra.com) fp4 ($0.93/$3.00 per M) vs [Z.ai](https://z.ai) first-party fp8 ($1.40/$4.40), both pinned through OpenRouter with fallbacks disabled, identical prompts from the official BABILong template, temp 0, interleaved requests, every scored response verified as served by its pinned provider. Total cost $27.
- **First run failed.** At max_tokens 8192, about 60% of responses on both arms burned the whole budget on reasoning (mean ~6.7k reasoning tokens) and returned empty answers. The nominal fp4 25% vs fp8 0% was pure truncation artifact.
- **Lesson one.** Cap your outputs carelessly and you benchmark the cap, not the model.
- **Rerun gave a clean null.** At 32,768 tokens: 20% vs 20%, McNemar p = 1.0. Both endpoints sit near the accuracy floor at this depth, so per the pre-registered rule (fund the sweep only if the gap exceeds 10 points), the deep-tier accuracy study died.
- **Finding one: token inflation.** The fp4 arm generated 29% more completion tokens than fp8 (20,716 vs 17,052 mean), hit the 32k ceiling on 12 of 20 items vs 8 for fp8 with every cap-hit unanswered, and was 2.1x slower in real time (374s vs 174s mean). Equal accuracy, more tokens, twice the wait.
- **Finding two: temp 0 is not deterministic.** Duplicate requests flipped answers on both arms with 10x swings in reasoning length, so any follow-up needs repeats or larger n.
- **Finding three: scoring is a judgment call.** fp8 lost two points to format compliance (right answer, wrong sentence position); lenient scoring would have read fp8 30% vs fp4 20%.
- **Caveats.** n = 20, the arms differ by full serving stack rather than quantization alone, and accuracy conflates wrong answers with never-answered rumination.

## Revised Thesis

- **Tokens per task is a price.** The degradation is not only in answer quality. The market lists dollars per token, nobody lists tokens per task, and effective cost is their product.
- **The discount can be fake.** A quantized endpoint can be more expensive than its discount implies while every visible metric says otherwise.
- **Structurally invisible.** [Artificial Analysis](https://artificialanalysis.ai/methodology) standardizes reasoning-token counts across providers "for fair comparison," so per-provider inflation gets laundered into "slow" by construction.
- **Mechanism is established, endpoints are not.** Two fresh papers ([CTIR](https://arxiv.org/abs/2606.25519), the chain-of-thought (CoT) token inflation ratio; [overthinking markers](https://arxiv.org/abs/2606.00206)) establish token inflation on local models. Nobody has measured it on endpoints as billed.
- **QAT vs PTQ refinement.** Inflation should appear only where serving precision diverges from training precision. GLM-5.2 is fp8-native and quantized to fp4 by providers after training (PTQ, post-training quantization), so it should inflate. DeepSeek V4 Flash had quantization in the training loop (QAT, quantization-aware training) at MXFP4 (Microscaling FP4, the open-standard 4-bit format), so it should not.

## Secondary Test

- **Design.** A pre-registered 2x2. Cell one, running now: V4 Flash on [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) x50, three arms (DeepSeek first-party, DeepInfra fp4, [Atlas Cloud](https://openrouter.ai/provider/atlas-cloud) fp4, which charges first-party prices for fp4 serving).
- **Measures.** Paired inflation ratios with CIs (confidence intervals), parity band [0.9, 1.1], effective vs listed cost per solved task as the headline number, plus billed-token cross-checks.
- **Cell two.** Same runner on GLM-5.2 arms, where inflation is predicted.
- **Predicted pattern.** Parity on the QAT'd model, inflation on the PTQ'd one. If that lands, the story upgrades from "one provider degrades" to "the fp4 label is meaningless without training-precision provenance, and the certification layer can't see the difference."

---

## Sources

- **[NVIDIA GLM-5.2-NVFP4 model card](https://huggingface.co/nvidia/GLM-5.2-NVFP4)**
	- PTQ from the BF16 (bfloat16) checkpoint with nvidia-modelopt v0.46.0. Only the linear ops inside the MoE (mixture-of-experts) experts are quantized; the shared expert is left at full precision.
	- Reported FP8 vs NVFP4 scores: GPQA Diamond (graduate-level science question answering) 89.52 vs 89.39, SciCode (scientific coding) 49.85 vs 49.04, IFBench (instruction following) 74.95 vs 75.81, AA-LCR (Artificial Analysis Long Context Reasoning) 69.38 vs 70.13, Tau2-Bench Telecom (agentic tool use) 97.9 vs 98.25. Largest drop is 0.81 points; three of five benchmarks score higher under fp4.
	- The card reports accuracy only. Token usage and long-horizon degradation go unmeasured.
- **[Umans: "What NVFP4 is, and why we chose not to serve GLM-5.2 in it"](https://blog.umans.ai/blog/glm-5-2-nvfp4-not-worth-serving/)**
	- Public experiment June 29 to July 2, 2026: NVFP4 hit 200+ tokens/sec (peaking near 250), sub-2s median TTFT, ~420 GB per copy vs 744 GB in fp8. Verdict: "the speed was real, the quality was not."
	- User reports during the trial: "chain-of-thought collapsing well before deep context," "went back to fp8 because nvfp4 was just making enough mistakes," "my workflow seems dumb with nvfp4 but was good with fp8."
	- Root cause per Umans: "the checkpoint was NOT QAT post-trained for NVFP4." Their policy: serve models only at the precision they were post-trained for. "If the tokens are not useful, they are not worth serving, no matter how efficiently we can produce them."
- **[OpenRouter: "Why OpenRouter for DeepSeek"](https://openrouter.ai/blog/insights/why-openrouter-for-deepseek/)** (7/13/26)
	- "Quality varies by provider, mainly because quantization differences across hosts can change the model's responses." The `quantizations` filter: "Filters out poorly-quantized endpoints. Fixes quality complaints."
	- Documented spread across 16 providers serving DeepSeek V4 Pro: input price $0.435 to $1.74 per M (4x), throughput 4 to 57 tokens/sec, uptime 97.44% to 99.92%.
- **[Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models](https://arxiv.org/abs/2606.25519)** (CTIR)
	- Introduces the CoT Token Inflation Ratio. INT4 (4-bit integer): +3.7% (Qwen3-30B GPTQ) to +43.0% (Qwen3-30B RTN) mean inflation with accuracy largely preserved. INT3: +19.0% up to +292.1% (Qwen3-4B HQQ); worst single case +957.4% (MBPP, a Python coding benchmark, under INT3 AWQ). GPTQ, AWQ, RTN (round-to-nearest), and HQQ (half-quadratic quantization) are all PTQ algorithms.
	- Inflation cancels the kernel speedup end to end: Qwen3-4B runs BBH (Big-Bench Hard, a reasoning suite) in 285s at BF16 but 292s at INT4 despite 1.2-1.4x faster kernels. A 1.1x CTIR raises p90 TTFT from 1050s to 1400s (+33%) and drops throughput from 0.055 to 0.047 req/s.
	- QAT is the most promising mitigation tested; prompting and decoding tricks gave inconsistent trade-offs. All measurements on locally-run models, not billed endpoints.
- **[Quantized Reasoning Models Think They Need to Think Longer, but They Do Not](https://arxiv.org/abs/2606.00206)**
	- 5 models (1.5B to 32B), 3 PTQ methods, 5 benchmarks across math, coding, and science question answering: aggressive PTQ reduces accuracy while increasing CoT length.
	- "In up to 52% of the quantized models' failures, models reach the right answer in intermediate reasoning steps but do not output it as a final answer."
	- Their fix, a training-free logit penalty on overthinking markers ("wait," "but," "alternatively"), cuts CoT length 12-23% and overthinking errors up to 58% while preserving or improving accuracy.
- **[BABILong](https://arxiv.org/abs/2406.10149)** (NeurIPS 2024)
	- 20 reasoning tasks (fact chaining, induction, deduction, counting) with facts hidden in distractor text, scalable to 10M tokens. We use qa3 (fact chaining) at 256k.
	- Headline finding: popular LLMs "effectively utilize only 10-20% of the context and their performance declines sharply with increased reasoning complexity." This is why both arms sat near the accuracy floor at 256k.
- **[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)**
	- 500-problem subset of the MATH competition benchmark (from OpenAI's "Let's Verify Step by Step" split). Secondary test dataset: short, objectively scorable, well inside every arm's context window, so token counts isolate reasoning inflation rather than context failure.
- **[Artificial Analysis methodology](https://artificialanalysis.ai/methodology)**
	- "Where the average number of reasoning tokens is not available or has not yet been calculated, we assume 2k reasoning tokens." Cost-per-task comparisons built on an assumed token count cannot register a provider that inflates tokens on the same model.
	- Speed metrics are normalized too: "We use OpenAI tokens as a standard unit of measurement across Artificial Analysis to allow fair comparisons between models." Per-provider inflation therefore surfaces only as lower throughput, not higher cost.
