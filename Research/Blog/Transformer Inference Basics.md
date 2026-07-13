OpenAI's API alone processes [~15 billion tokens per minute](https://aidailypost.com/news/openai-api-token-usage-rises-from-6-bn-15-bn-per-minute-straining), roughly 21.6 trillion tokens per day. For comparison, [DeepSeek-V4](https://arxiv.org/abs/2606.19348), a frontier open-source model, was pre-trained on 32 trillion tokens total. OpenAI re-spends a frontier training run's worth of tokens on inference every day and a half.

And serving end-users via the API is only a part of it. The inference stack extends to **evals**, where benchmarking a model means generating millions of tokens, and **reinforcement learning**, which requires rollouts on-policy, or letting the model play out under some policy to measure its performance 

Given this, it shouldn't be surprising that the *efficiency* of model inference is dominating the economics of AI, especially when time spent on the GPU is so precious. 

Unfortunately, inference is _hard to make efficient_ in a way training fundamentally isn't, and the reason is memory, not FLOPs. NVIDIA's new [Vera Rubin POD](https://developer.nvidia.com/blog/nvidia-vera-rubin-pod-seven-chips-five-rack-scale-systems-one-ai-supercomputer/) packs 1,152 GPUs and 60 exaFLOPS into 40 racks, but if you can't move bytes fast enough, none of those FLOPs matter.
## The nature of inference

![[inference-autoregressive.png]]
https://jax-ml.github.io/scaling-book/inference/#what-do-we-actually-want-to-optimize

Because transformers are autoregressive, each token is sampled from a distribution conditioned on everything before it. You can't compute token t+1 until you've sampled token t, since the output literally becomes the next input. 

Fortunately, all the tokens from the "prompt" or "context" are already known, so you can just stuff those into a single tensor and process the whole thing in a single parallel pass. That is, if your context is S tokens, you can immediately compute token S+1. 

Unfortunately, the following tokens (S+2, ... S+n) have to be computed sequentially, waiting on the previous token generated before you can do another forward pass to do the next. 

This is the core asymmetry of inference. Training lets you parallelize over the full sequence because you know the proceeding token; generation forces you through it serially because you only know the preceding ones. Recall from the previous post that GPUs are specialized in throughput: thousands of slow threads doing bulk arithmetic. Sequential token-by-token generation is close to the worst-case workload for them. 
## KV Cache: Prefill & Decode
![[inference-prefill-decode.png]]
https://jax-ml.github.io/scaling-book/inference/#what-do-we-actually-want-to-optimize

Naively, you could just do a full forward pass on the prefix to generate a new token, appending the sample to the prefix before each additional pass. So to generate n new tokens, you'd do n full forward passes. While simple, it's pretty inefficient. It takes O(n²) for the FFW (the feed forward layers) and O(n³) for the attention mechanism. That is, the complexity, or the amount of work needed to be done, scales cubically with the amount of tokens you want to generate — bottlenecked mostly by attention. Horrific. 

The insight to solve this involves realizing that each full forward pass is doing lots of repeated operations on previous tokens. Specifically, the attention's K & V matrices are mostly recomputed for each pass. Indeed, we could save these in a **KV Cache** and for the new token use its Q against the cached K/V, appending the new K/V to the cache for future sampling. 

Notably this splits inference into two parts: **prefill** and **decode**. Prefill, as the name suggests, fills the KV cache with the values from the prefix. Afterwards, future sampling is still done in decode, still sequentially, but now total generation for n tokens drops to O(n²). 

Nothing is free, though. The complexity is transferred from compute and into memory. Per token, per layer, the cache stores 2 (K and V) × n_kv_heads × head_dim values. For a Llama-2-13B-class model in FP16, that's roughly 1 MB _per token_, so a batch of 32 sequences at 4K context eats >100 GB, more than the model weights themselves. This is why so much post-2023 architecture work ([GQA](https://arxiv.org/abs/2305.13245), [MLA](https://arxiv.org/abs/2405.04434), DeepSeek-V4's [compressed attention](https://arxiv.org/abs/2606.19348) hitting 10% of V3.2's cache size) is really an attemp to compress the KV cache.

> Note a small free win: a forward pass produces logits (log-probabilities) for each token in the sequence, and the logits at position t are exactly the distribution for t+1, so the first "generated" token falls out of prefill itself, and the first true decode step produces token S+2. 
> 
> Concretely, if the prompt is "the brown fox", all three tokens enter at once as a [1, 3, d] tensor. One sequence with three tokens each embedded as a vector of size *d*. A causal mask ensures each position only attends backward, so the pass yields a prediction after "the", after "the brown", and after "the brown fox". The first two are thrown away and used only to fill the KV cache, discussed below. The last one is kept and sampled: " jumped". In training, discarded predictions are scored against the known token via a loss function, and the model updates via a backward pass. Predicting token S+2 requires the _sampled_ S+1 as input.

## Metrics

To progress further, it's important to know what you're looking for:

| **Time-to-first-token (TTFT)** | Delay before generation starts. Since prefill samples the first fresh token, it's roughly prefill time. |
| ------------------------------ | ------------------------------------------------------------------------------------------------------- |
| **Latency** (sec/token)        | How fast tokens stream for _one_ request. Dominated by decode speed.                                    |
| **Throughput** (tokens/sec)    | Aggregate tokens across _many_ requests.                                                                |

These trade off against each other. Batching more requests together improves throughput but worsens per-request latency. A chat product optimizes TTFT and latency; an eval harness or RL trainer optimizes raw throughput and doesn't care if any single sequence is slow.
## Why inference is memory-bound

![[roofline-model.png|472]]
https://jax-ml.github.io/scaling-book/roofline/

The roofline model shows often shows that as arithmetic intensity increases, essentially FLOP per byte transferred, utilization of the hardware increases and eventually reaches a ceiling, where we are fully utilizing the compute of our hardware. 

If the intensity is below the hardware's ratio of compute to bandwidth (an H100 wants ~300 FLOPs/byte at BF16), you're memory-bound.

Consider generating one token at batch size B, with S tokens of context:

**MLP blocks**: the weight matrices (d² parameters each) must be read from HBM regardless of batch size, but every sequence in the batch reuses the same weights. FLOPs scale with B while bytes stay roughly fixed, so intensity ≈ **B**. Small batches are memory-bound; crank the batch up and the MLPs become compute-bound. Batching works great and means that if you can deliver inference at scale, via more user requests, RL, or evals, you'll likely be utilizing the hardware more. 

**Attention**: each sequence has its _own_ KV cache. Doubling the batch doubles the FLOPs _and_ doubles the bytes read. For prefill of T tokens against context S, intensity works out to roughly **S·T / (S + T)**; during decode, T = 1, so intensity ≈ 1. One FLOP per byte, no matter the batch size.

> Aside: this is the deepest bottleneck in transformer inference. MLP inefficiency is fixable: just batch harder. Attention decode intensity is architecturally pinned near 1 for standard transformers. You cannot batch your way out, only shrink the cache (GQA, MLA, quantization) or change the architecture (sparse/linear attention, SSMs). This asymmetry is quietly steering frontier architecture design.

At low batch sizes, then, the whole system is memory-bound and your effective FLOPS utilization craters. Single-digit MFU (Model Flop Utilization) during decode is normal, versus 40-50% during training. Serving systems fight this with **continuous batching** (admitting new requests mid-generation to keep B high), **paged KV caches**, **speculative decoding**, and **disaggregated serving** that runs prefill and decode on separate GPU pools tuned for each regime.

## The Open-Source Stack

The nice thing about software is that usually someone has already done a lot of the hard work and gives it out for free. Some popular stacks that allow you to start on the shoulders of giants:

| Engine                                                                                                                                                          | Notes                                                                                                                |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [vLLM](https://github.com/vllm-project/vllm)                                                                                                                    | The good default. Introduced [PagedAttention](https://arxiv.org/abs/2309.06180); huge community, broad model support |
| [SGLang](https://github.com/sgl-project/sglang)                                                                                                                 | Strong for agentic/structured workloads. RadixAttention reuses KV cache across shared prefixes                       |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)                                                                                                          | NVIDIA's engine; maximum performance on NVIDIA hardware, more operational overhead                                   |
| [llama.cpp](https://github.com/ggml-org/llama.cpp)                                                                                                              | C++, runs on CPUs and consumer hardware; the backbone of local inference (and of [Ollama](https://ollama.com))       |
| [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)                                                                                                            | Datacenter-scale orchestration: disaggregated prefill/decode across GPU fleets                                       |
| [TGI](https://github.com/huggingface/text-generation-inference), [LMDeploy](https://github.com/InternLM/lmdeploy), [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | Hugging Face's server; InternLM's engine; compile-to-anything (phones, browsers)                                     |

## The takeaway
Serving a model at scale often demands much more compute than training it. Due to the nature of inference, it's also memory bound. This is in-part why memory is currently considered a bottleneck in the AI economy, and why inference efficiency is so important.

We can split inference into prefill, which behaves like training and saturates the GPU, and decode, which reads gigabytes of weights and KV cache to emit one token at a time. 

Everything interesting in inference (KV caches, GQA/MLA, paged attention, continuous batching, speculative decoding) is a workaround for the same issue: at generation time, arithmetic intensity collapses, and the bytes, not the FLOPs, are what are payed for.