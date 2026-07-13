# Serving Models

Research dump (July 2026) answering: *If everyone has their own model weights, how are they served? How hard is switching models? Do you need to rent GPUs separately? What do inference libraries like vLLM do, really? At a technical level, how do you serve a model?* From [[Questions]]. Related: [[Inference]], [[Inference Economics]], [[GPU Rental Markets]].

## If everyone has their own weights, how are they served?

The answer to "everyone owns a model" is **not** one GPU per person — it's **multi-LoRA multiplexing**: one shared base model per GPU + per-user LoRA adapters (~0.5–1% of base size, ~80MB for a 7B) hot-swapped per request in **2–20ms** (RAM→GPU over PCIe).

- **S-LoRA** (Berkeley, MLSys'24): unified memory pool for KV cache pages *and* adapter pages; 2,000 adapters on one GPU; 30x vs naive PEFT
- **Punica** (UW, MLSys'24): SGMV kernel batches decode across *different* LoRA models in one launch; 12x throughput
- **LoRAX** (Predibase, open source; Predibase acquired by Rubrik 2025): production multi-LoRA server — JIT adapter loading, heterogeneous continuous batching, async prefetch
- **vLLM built-in**: `--enable-lora`, GPU slots + CPU tier with LRU eviction, serves hundreds of adapters
- **Commercial**: Together Serverless Multi-LoRA and Fireworks Multi-LoRA serve hundreds of custom adapters **at base-model per-token prices**; DeepInfra BYO-LoRA
- **Production case**: OpenPipe (acquired by CoreWeave → "Serverless RL" with W&B) cut cold-start 45s→1s and GPU count −70% via S-LoRA
- Production patterns: GPU→CPU→disk adapter tiers, adapter-aware routing, merge any adapter with >80% of traffic into the base
- Research frontier: "million personal models on trillion-parameter bases" — per-user adapters as persistent local state; hypernetwork-generated instant adapters; a16z's continual-learning thesis (in-context first, adapters for personalization)

Full per-user fine-tunes (not adapters) only pay off with dedicated capacity.

### The four serving tiers

1. **Serverless per-token APIs** — shared multi-tenant pool, $0 idle, no GPU rental. Providers: **Together** (~$1B ARR, broadest open-weight catalog, downloadable fine-tuned weights), **Fireworks** (~$800M ARR, speed leader, ex-PyTorch team, day-one new-model support), **Baseten** ($13B val, BYO-model-first, Truss packaging), **DeepInfra** (cheapest floor — gpt-oss-120B ~$0.05/M blended — but quantizes to FP4), **Replicate** (50k+ community models; acquired by Cloudflare), **Modal** (~$300M ARR, Python-native), **Novita**, **Hyperbolic**, **Parasail** (early: "inference supercloud" over 25+ clouds, 500B tok/day), **Morph** (early: bf16-fidelity as the differentiator), **fal** (generative media), **Nebius Token Factory**, **Anyscale**; silicon-differentiated: **Groq**, **Cerebras**, **SambaNova** (see [[Frontier Chips]])
2. **Dedicated endpoints** — rent capacity through the platform per GPU-hour (Baseten $6.50/hr H100, Together $6.49, Fireworks $7.00), guaranteed rate limits, scale-to-zero
3. **BYO weights / custom hosting** — Baseten (core use case), Fireworks, Together (upload LoRA adapters), Replicate Cog, HF Inference Endpoints
4. **Self-host on rented GPUs** — run vLLM/SGLang yourself on Vast/RunPod/CoreWeave/Lambda (see [[GPU Rental Markets]])

**Do you need to rent GPUs separately?** No for serverless per-token and multi-LoRA; yes for dedicated/self-host. Crossover: dedicated wins above ~30% sustained utilization (~150M output tok/day for a 70B). Example: 10M tok/day on a 70B ≈ $195/mo serverless vs ~$3,600/mo for one dedicated H100.

## How hard is switching models?

**API level: trivial.** Everything is OpenAI-compatible — switching = change base URL + model string. Routers make it a config value: **OpenRouter** (400+ models, 70+ providers, 25T tok/week, $1.3B val; automatic fallbacks, `:floor`/`:nitro` suffixes), **Vercel AI Gateway**, **LiteLLM**, Portkey, Helicone. Workload-aware routing beats any single provider by 30–50% on cost.

**Infra level: hard — cold starts.** Loading a 70B from S3 naively ≈ 5 min; full vLLM cold start (container pull + engine init + CUDA graph capture + weight load) ≈ 3–5 min; TensorRT-LLM engine compile ≈ 28 min. Keeping warm replicas to dodge this can 2–3x GPU spend. The fast-switching stack attacking it:

- **Streaming loaders**: NVIDIA Run:ai Model Streamer (Llama-3-8B from S3 in 4.9s vs 47s HF loader), CoreWeave Tensorizer, vLLM instanttensor
- **GPU memory snapshots**: Modal GPU snapshots and NVIDIA Dynamo Snapshot (built on the new cuda-checkpoint driver API) — sub-second restores of loaded+compiled state
- **Early startups**: **Outerport** (YC S24 — pinned-RAM weight cache + hierarchical RAM→GPU distribution), **InferX** (CPU+GPU snapshots to NVMe; cold TTFT <2–5s, evict an idle GPU in ~10ms — "stop paying for warm pools")
- Research: ServerlessLLM, FlashServe (0.58s cold start for 7B), Foundry (Qwen3-235B init 10min→3.9s)
- Practical tricks: `enforce_eager=True` (skip CUDA graphs), pre-merge LoRAs, never gzip weight layers in images

## At a technical level, how do you serve a model?

The stack, layer by layer:

1. **Weights on disk** — safetensors is standard (zero-copy, safe); GGUF for llama.cpp; compiled TRT engines per GPU arch
2. **Load to GPU** — storage→CPU (storage-bandwidth-bound), CPU→GPU over PCIe (~25 GB/s pinned DMA); then engine init: CUDA context, torch.compile/CUDA-graph capture, KV-cache allocation
3. **Inference engine** (single node) — vLLM/SGLang/TensorRT-LLM: continuous batching + paged KV + prefix caching + speculative decoding + quantization (below)
4. **HTTP layer** — OpenAI-compatible REST + SSE streaming is the universal contract; structured output/JSON mode
5. **Orchestration** (multi-node) — **llm-d** (Red Hat/Google-backed K8s-native: prefix-cache-aware routing, tiered KV offload, KV-utilization autoscaling), **KServe** (lifecycle; its new LLMInferenceService CRD builds on llm-d), **NVIDIA Dynamo** (Triton's successor: KV-aware Smart Router, NIXL async KV transfer, tiered KV Block Manager), **Ray Serve**, **BentoML**
6. **KV infrastructure** — **LMCache** (tiered GPU→CPU→disk→remote KV reuse, up to 15x with vLLM), **Mooncake** (Moonshot's RDMA distributed KV store)
7. **Prefill/decode disaggregation** — now mainstream: separate GPU pools for compute-bound prefill vs memory-bound decode, KV shipped over RDMA; in Dynamo, SGLang, llm-d, SageMaker. **Moreh** (early, Korean) even does it across vendors — prefill on NVIDIA, decode on AMD
8. **Autoscaling** — reactive scaling is incompatible with LLM cold starts → predictive prewarming, snapshots, or paying for min-replicas

## What do inference libraries like vLLM do, really?

**The problem:** naive HuggingFace `model.generate()` serves one (padded, static) batch at a time, pre-allocates contiguous KV cache to max length — existing systems wasted **60–80% of KV memory** — and the whole batch waits for its slowest member. Decode is memory-bandwidth-bound (<1 FLOP/byte vs ~500 needed to saturate the chip; see [[Inference]]).

**vLLM is an inference engine** (Berkeley, SOSP 2023): takes weights and turns a GPU into a high-throughput OpenAI-compatible token service:

- **PagedAttention** — KV cache in fixed-size non-contiguous blocks with a block table (virtual-memory analogy); waste drops from 60–80% to <4%; copy-on-write sharing for parallel sampling
- **Continuous (iteration-level) batching** — finished requests exit and new ones join *every decode step*; the single biggest unlock (up to ~23x vs static batching)
- Optimized kernels (FlashAttention, fused paged-attention, CUDA graphs), quantization (FP8/GPTQ/AWQ), speculative decoding, prefix caching, tensor/pipeline/expert parallelism, multi-LoRA, structured output, OpenAI server built in

Headline: up to 24x throughput vs raw HF Transformers; 2–4x vs older serving stacks. One reproduction: 220 tok/s naive → 980 (+continuous batching) → 3,100 (+PagedAttention) → 4,900 (+prefix caching) ≈ 22x.

**The landscape (features converged, architectures differ):**

- **vLLM** — default choice; broadest models (~400 archs) and hardware (NVIDIA/AMD/TPU/Intel); Python scheduler overhead is its weakness. "Right answer for 80% of teams"
- **SGLang** (LMSYS; commercialized by **RadixArk**, $100M seed) — **RadixAttention**: radix tree over ALL cached KV across requests → 70–90% prefill elimination on shared-prefix/agentic workloads; best for big MoE; powers xAI Grok and Azure DeepSeek-R1
- **TensorRT-LLM** (NVIDIA) — compiled engines, +13–25% over vLLM on H100, best on Blackwell, native FP8 compute; costs 28-min compiles and NVIDIA lock-in
- **LMDeploy** — C++ TurboMind, throughput ≈ SGLang
- **TGI** (HF) — maintenance mode as of 2026
- **llama.cpp/Ollama, MLX, MLC** — local/edge tier, batch-1, not datacenter serving

On H100 Llama-70B FP8 at 50 concurrent, the big three land within ~14% of each other — the choice is operational, not raw speed.

| Technique | What it does | Typical gain |
|---|---|---|
| Continuous batching | requests join/leave every token step | up to ~23x vs static |
| PagedAttention | paged KV, <4% waste | 2–4x concurrency |
| Prefix/radix caching | reuse shared-prompt KV | up to 5x; near-zero TTFT warm |
| Speculative decoding | draft + parallel verify, output-identical | 2–3x decode |
| Quantization FP8/INT4 | fewer bytes per weight | ~2x per halving |
| Prefill/decode disaggregation | separate pools per phase | +59–498% goodput |

Framing that recurs in sources: hardware sets the roofline; the serving stack determines how close to it you live. "Inference efficiency means moving fewer bytes, reusing bytes better, or moving bytes over a better path."
