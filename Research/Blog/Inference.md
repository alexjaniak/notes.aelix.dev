OpenAI's API alone processes [~6 billion tokens per minute](https://www.pymnts.com/artificial-intelligence-2/2025/openai-bests-google-in-race-for-consumer-ai-token-consumption/), roughly 8.6 trillion tokens _per day_. For comparison, [DeepSeek-V4](https://arxiv.org/abs/2606.19348), a frontier open-source model, was pre-trained on 32 trillion tokens _total_. OpenAI re-spends a frontier training run's worth of tokens on inference every four days.

And serving end-users via the API is only a part of it. The inference stacks extends to: 
- **Evals**: benchmarking a model means generating millions of tokens. Slow inference butchers iteration. 
- **RL**: modern post-training (RLHF, GRPO, agentic RL) generates rollouts on-policy; so Inference throughput directly gates training speed. With large frontier models spending [thousands of tokens *reasoning*](https://www.saastr.com/the-real-data-on-ai-agents-what-1-trillion-tokens-a-day-reveals-with-openrouters-coo/), this only compounds.

Given this, it shouldn't be surprising that the **efficiency** of model inference is dominating the economics of AI, especially when time spent on the GPU is so precious. 

Unfortunately, inference is _hard to make efficient_ in a way training fundamentally isn't, and the reason is memory, not FLOPs. If you can't move bytes fast enough, it doesn't matter how many FLOPs you're 1000 GPU Pod can do. 

## Two Phases: Prefill and Generation
Every inference request has two distinct stages with completely different performance characteristics.

![[prefill-vs-decode.png]]

**Prefill**: the prompt is encoded into vectors. Since all prompt tokens are known upfront, this is fully parallelizable, exactly like a training forward pass. The GPU chews through the whole sequence at once, saturating its compute units. Prefill is **compute-bound**.

**Generation (decode)**: the model produces new tokens one at a time. Each token depends on all previous ones, so there's no parallelizing across the sequence. Every step requires reading the _entire_ model's weights from HBM to produce a single token. Generation is **memory-bound**.

This is the core asymmetry of inference. Training lets you parallelize over the full sequence; generation forces you through it serially. Recall from the last post <need hyperlink> that GPUs are specialized in throughput: thousands of slow threads doing bulk arithmetic. Sequential token-by-token generation is close to the worst-case workload for them.

## Metrics

Three numbers characterize an inference system:

|Metric|What it measures|Who cares|
|---|---|---|
|**Time-to-first-token (TTFT)**|Delay before generation starts (roughly prefill time)|Interactive apps, chat|
|**Latency** (sec/token)|How fast tokens stream for _one_ request|Users watching output|
|**Throughput** (tokens/sec)|Aggregate tokens across _many_ requests|Batch jobs, RL rollouts, cost|

These trade off against each other. Batching more requests together improves throughput but worsens per-request latency. A chat product optimizes TTFT and latency; an eval harness or RL trainer optimizes raw throughput and doesn't care if any single sequence is slow.

## The Naive Cost: O(T³)

Attention over a sequence of length T costs O(T²) FLOPs for a single forward pass. Naively, generating each new token means re-running the forward pass over the whole prefix, recomputing every key and value projection and every attention score you already computed last step. Generating T tokens this way costs O(T³). Horrific.

The fix is to notice that the keys and values for past tokens _don't change_. Compute them once, store them, and each new token only needs to (1) compute its own K/V and append, and (2) attend over the stored cache. This is the **KV cache**, and it lives in HBM. Per-token cost drops from O(T²) to O(T), total generation from O(T³) to O(T²).

![[kv-cache.png]]

Nothing is free, though. You've converted a compute problem into a memory problem. Per token, per layer, the cache stores 2 (K and V) × n_kv_heads × head_dim values. For a Llama-2-13B-class model in FP16, that's roughly 1 MB _per token_, so a batch of 32 sequences at 4K context eats >100 GB, more than the model weights themselves. This is why so much post-2023 architecture work ([GQA](https://arxiv.org/abs/2305.13245), [MLA](https://arxiv.org/abs/2405.04434), DeepSeek-V4's [compressed attention](https://arxiv.org/abs/2606.19348) hitting 10% of V3.2's cache size) is really KV-cache-compression work in disguise.

## Why Generation Is Memory-Bound: Arithmetic Intensity

The [roofline model](https://en.wikipedia.org/wiki/Roofline_model) applies directly. **Arithmetic intensity** = FLOPs performed per byte moved. If intensity is below the hardware's ratio of compute to bandwidth (an H100 wants ~300 FLOPs/byte at BF16), you're memory-bound: the ALUs idle while bytes crawl in from HBM.

Consider generating one token at batch size B, with S tokens of context:

**MLP blocks**: the weight matrices (d² parameters each) must be read from HBM regardless of batch size, but every sequence in the batch reuses the same weights. FLOPs scale with B while bytes stay roughly fixed, so intensity ≈ **B**. Small batches are memory-bound; crank the batch up and the MLPs become compute-bound. Batching works.

**Attention**: each sequence has its _own_ KV cache. Doubling the batch doubles the FLOPs _and_ doubles the bytes read. For prefill of T tokens against context S, intensity works out to roughly **S·T / (S + T)**; during decode, T = 1, so intensity ≈ 1. One FLOP per byte, no matter the batch size.

> Aside: this is the deepest bottleneck in transformer inference. MLP inefficiency is fixable: just batch harder. Attention decode intensity is architecturally pinned near 1 for standard transformers. You cannot batch your way out, only shrink the cache (GQA, MLA, quantization) or change the architecture (sparse/linear attention, SSMs). This asymmetry is quietly steering frontier architecture design.

![[roofline-decode.png]]

At low batch sizes, then, the whole system is memory-bound and your effective FLOPS utilization craters. Single-digit MFU during decode is normal, versus 40-50% during training. Serving systems fight this with **continuous batching** (admitting new requests mid-generation to keep B high), **paged KV caches**, **speculative decoding**, and **disaggregated serving** that runs prefill and decode on separate GPU pools tuned for each regime. Those deserve their own post.

## The Open-Source Stack

You mostly don't write this machinery yourself:

|Engine|Notes|
|---|---|
|[vLLM](https://github.com/vllm-project/vllm)|The good default. Introduced [PagedAttention](https://arxiv.org/abs/2309.06180); huge community, broad model support|
|[SGLang](https://github.com/sgl-project/sglang)|Strong for agentic/structured workloads. RadixAttention reuses KV cache across shared prefixes|
|[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)|NVIDIA's engine; maximum performance on NVIDIA hardware, more operational overhead|
|[llama.cpp](https://github.com/ggml-org/llama.cpp)|C++, runs on CPUs and consumer hardware; the backbone of local inference (and of [Ollama](https://ollama.com))|
|[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)|Datacenter-scale orchestration: disaggregated prefill/decode across GPU fleets|
|[TGI](https://github.com/huggingface/text-generation-inference), [LMDeploy](https://github.com/InternLM/lmdeploy), [MLC-LLM](https://github.com/mlc-ai/mlc-llm)|Hugging Face's server; InternLM's engine; compile-to-anything (phones, browsers)|

For most people the decision tree is short: serving open models, use vLLM; heavy prefix reuse or agent loops, use SGLang; squeezing NVIDIA datacenter hardware, TensorRT-LLM (often under Dynamo); laptop, llama.cpp.

## The Takeaway

Training is a compute problem; inference is a memory problem. Prefill behaves like training and saturates the GPU; decode reads gigabytes of weights and KV cache to emit one token at a time. Everything interesting in inference (KV caches, GQA/MLA, paged attention, continuous batching, speculative decoding) is a workaround for the same fact: at generation time, arithmetic intensity collapses, and the bytes, not the FLOPs, are what you pay for.