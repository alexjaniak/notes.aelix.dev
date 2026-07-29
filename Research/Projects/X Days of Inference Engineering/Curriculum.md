# X Days of Inference Engineering — Curriculum

A fluid plan for daily inference-engineering practice: 1–2 hours a day, every day ends with a post (tweet or Substack) built around one measured number or chart. Not date-bound — arcs and day-sized units, picked in whatever order the work demands. Related: [[Inference]], [[GPUs]], [[Serving Models]], [[Inference Economics]], [[Questions]].

## How this works

Two layers:

- **The skills backbone (this note).** Serving models, engine internals, GPU programming, systems papers. These compound regardless of what I'm working on and don't change when projects do.
- **A rotating current-project slot.** At any given time there's a live project (right now: [[Testing Inference Degredation]]). Projects come and go; whatever the current one is, it borrows skills from the backbone and generates the "why now" for particular units. When a project dies, the backbone doesn't.

The bet: doing real inference serving will keep surfacing problems worth chasing — cold starts, nondeterminism, scheduling weirdness, cost surprises. The curriculum's job is to make sure I keep bumping into those problems with enough depth to recognize which ones matter.

Two kinds of days, deliberately mixed:

- **Project days** — push the current project forward using skills from the backbone. Time these so results land before team syncs.
- **Depth days** — engine internals, CUDA, serving systems, a paper. Pure skill compounding, 1–2 hours, always still ends in a post.

## The daily post formula

1. One question, stated up front ("Is decode really memory-bound? My MacBook can prove it.")
2. **Predict before measuring.** Write down the expected number, then run it. Post the delta — being wrong in public is the content.
3. One chart or one number. Never a summary of reading with nothing measured. (Paper days: reproduce or sanity-check one claim from the paper.)
4. What surprised me / what breaks tomorrow.

Failed runs are posts too — "you benchmark the cap, not the model" is the genre.

## Arcs at a glance

| Arc | What |
|---|---|
| 1. Serve it yourself | Week 1: local → rented GPU → vLLM endpoint → benchmark → quantize |
| 2. Inside the engine | Build a mini inference engine (Rust-flavored) |
| 3. GPU programming | CUDA/Triton: matmul → tiling → roofline → attention |
| 4. Serving-systems internals | vLLM/SGLang source, PagedAttention, batching, spec decode, PD disaggregation |
| 5. Quantization & precision | Formats, quantize-it-yourself, what low-bit does to outputs |
| 6. Parallelism & scale | Multi-GPU, interconnects, cost-from-first-principles |

Plus a **paper track** (below) — inference-serving systems papers and surveys, one at a time, each paired with a measurement.

---

## Arc 1 — Serve it yourself (Week 1, concrete)

Goal by end of week: a model served by my own hand, benchmarked, quantized, and a "life of a token" walkthrough the team can follow.

**Day 1 — Local inference, and prove decode is memory-bound.** → [[Day 1 — Local Inference]] (each day gets its own walkthrough page like this). Run llama.cpp on the Mac with a small model (Qwen 8B-class, Q4 GGUF). Measure prefill tok/s vs decode tok/s. Then the punchline: predicted decode speed = memory bandwidth ÷ bytes-per-token (M-series bandwidth is public); compare predicted vs measured. Read: revisit own [[Inference]] notes on prefill/decode asymmetry. Post: "My laptop's memory bus predicts its token rate to within X%."

**Day 2 — Rent a GPU, stand up a real endpoint.** → [[Day 2 — Rent a GPU, Serve an Endpoint]]. Vast/RunPod/Lambda (see [[GPU Rental Markets]]); one 4090 or A100. Install vLLM, serve an 8B model, hit it with the OpenAI client from the laptop. Save a reusable image/volume so future days start in minutes, not an hour. Post: "$1 and 40 minutes to my own OpenAI-compatible endpoint" + the cold-start-time reality vs [[Serving Models]] claims.

**Day 3 — Benchmark it: watch continuous batching earn its keep.** Sweep concurrency 1→64 with `vllm bench serve` (or genai-perf): TTFT, per-request tok/s, aggregate throughput. Chart the knee where the GPU saturates. Predict the knee from roofline math first. Post: the throughput-vs-concurrency chart, predicted vs actual knee.

**Day 4 — Quantize the model myself.** llm-compressor (or AWQ/GPTQ recipes) → FP8 and INT4 variants of the same checkpoint; serve each on the same vLLM, same GPU. Measure throughput gain, memory freed, and a quick paired quality check. A core serving skill — every provider does this; now I know what it involves.

**Day 5 — The walkthrough.** Write "what happens when you call an inference API": weights on disk → PCIe → KV cache → scheduler → SSE stream, annotated with the numbers measured this week. This is the team deliverable and probably the best post of the week. Diagram it (→ `Research/Attachments/`).

## Arc 2 — Inside the engine (build one, Rust-flavored)

- **2.1 Naive generation.** HF transformers, `use_cache=False` vs `True`, time per token as context grows. Chart the O(n²)→O(n) cliff. (The math is already in [[Transformer Inference Basics]] — now measure it.)
- **2.2 Read llama2.c end-to-end.** Every tensor accounted for. Post: "700 lines is all inference is."
- **2.3 Sampling from scratch.** Greedy, temperature, top-p, min-p — implement, visualize what each does to the distribution.
- **2.4 KV cache from scratch.** Own loop, measure memory growth per token, verify the 2 × layers × kv_heads × head_dim × bytes formula against `nvidia-smi`.
- **2.5 Static batching and padding waste.** Batch 8 ragged prompts, measure wasted compute — the why of continuous batching, felt.
- **2.6 Continuous-batching simulator in Rust.** A queueing simulation (arrivals, prefill/decode service times, iteration-level scheduling) — network-engineer home turf. Measure goodput vs static batching.
- **2.7 Speculative decoding toy.** Draft model + verify, measure acceptance rate vs speedup, find where it inverts.
- **2.8 Milestone: mini engine in Rust.** candle + a small model, KV cache, streaming HTTP endpoint. Multi-day; the capstone post.

## Arc 3 — GPU programming

Pair each unit with PMPP chapters and GPU MODE lectures; practice reps on LeetGPU/KernelBench. All on the rented-GPU image from Day 2.

- **3.1 CUDA hello.** Vector add; deviceQuery. Answer the open [[GPUs]] questions concretely: how many SMs, SPs per SM, L1/L2 sizes on *this* card.
- **3.2 Coalescing.** Strided vs coalesced copy bandwidth chart — the burst-mode notes, measured.
- **3.3 Naive matmul.** GFLOP/s vs cuBLAS. Expect ~1–2% of peak; post the humiliation.
- **3.4 Tiled matmul.** Shared memory tiling; chart the climb toward cuBLAS across 3–4 optimizations.
- **3.5 Roofline.** Nsight Compute; place own kernels on the measured roofline.
- **3.6 Divergence & occupancy.** Branchy kernel vs masked kernel, measured — the control-divergence note made real.
- **3.7 Triton day.** Same matmul in Triton; effort-vs-performance comparison post.
- **3.8 Softmax → online softmax.** The kernel that makes flash attention possible.
- **3.9 Baby flash attention.** Forward pass in Triton, tiny sizes. Compare against naive attention memory traffic.
- **3.10 Low-precision on tensor cores.** How fp8/int4 matmul actually executes (answers "does the chip need special circuits?" from [[GPUs]]).

## Arc 4 — Serving-systems internals

- **4.1 vLLM source tour.** Scheduler, block manager, model runner. Post: "life of a request through vLLM," with file/line references.
- **4.2 PagedAttention measured.** KV waste with/without paging via vLLM metrics; verify the <4% claim from [[Serving Models]].
- **4.3 Prefix caching.** Agentic-shaped workload, warm vs cold TTFT; then same on SGLang RadixAttention.
- **4.4 Chunked prefill & scheduling.** The TTFT-vs-throughput dial, swept and charted.
- **4.5 Speculative decoding for real.** EAGLE/draft on vLLM, acceptance rates on my actual workload.
- **4.6 Engine bake-off.** vLLM vs SGLang vs TensorRT-LLM, same model/GPU — a mini InferenceMAX, and a check on the "within ~14%" claim.
- **4.7 Prefill/decode disaggregation.** vLLM's experimental PD / Dynamo / Mooncake. Why the most memory-bound phase gets its own fleet.
- **4.8 MoE serving.** Why experts change the batching math; expert parallelism.
- **4.9 Cold starts.** Measure the full cold-start anatomy on my own stack (container → weights → CUDA graphs → first token); try a streaming loader. The [[Serving Models]] cold-start section, reproduced.
- **4.10 Determinism.** Temp=0 on my own endpoint: vary batch size / concurrency, measure output divergence on identical prompts. Read Thinking Machines' batch-invariance post. A serving problem first; useful everywhere.

## Arc 5 — Quantization & precision

A serving skill, not a research agenda: providers quantize, engines ship fp8 paths, and low-bit kernels are where serving meets hardware.

- **5.1 Formats taxonomy.** INT8/FP8/GPTQ/AWQ/NVFP4/MXFP4 — what group scaling factors actually are (open question in [[GPUs]]), block sizes, why transposes are awkward. Deliverable: the one-diagram explainer of a quantized tensor.
- **5.2 Quantize one model four ways.** Perplexity + file size + load time + throughput for each. Chart: quality-vs-bytes frontier for one checkpoint.
- **5.3 KL divergence, not accuracy.** Per-token KL between bf16 and each quant on identical prompts — see what low-bit actually does to the output distribution.
- **5.4 KV-cache quantization.** fp8 KV cache in vLLM: memory freed vs quality, measured.
- **5.5 QAT vs PTQ.** Why training-time quantization behaves differently from post-hoc — mechanism-level understanding.

## Arc 6 — Parallelism & scale (later)

- **6.1 Tensor parallelism.** 70B on 2×GPU, measure the NCCL communication share of step time.
- **6.2 Interconnects.** NVLink vs PCIe measured bandwidth/latency — networking background, directly.
- **6.3 TP vs PP.** When each wins; measure pipeline bubbles.
- **6.4 Multi-node & KV transfer.** RDMA, Mooncake-style KV shipping — inference's networking problem.
- **6.5 Economics capstone.** Cost per M tokens from first principles (rental $/hr ÷ measured throughput) vs listed provider prices → estimated margins. Closes the loop with [[Inference Economics]].

## Paper track

One serving-systems paper at a time, in roughly this order, each read the day before (or after) the hands-on unit it pairs with. Rule: every paper day reproduces or sanity-checks one claim.

| Paper                                                                                                                                | Pairs with                  |
| ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------- |
| Orca: iteration-level scheduling (OSDI '22)                                                                                          | 2.5/2.6 continuous batching |
| vLLM / PagedAttention (SOSP '23, arXiv 2309.06180)                                                                                   | 4.1/4.2                     |
| SGLang / RadixAttention (arXiv 2312.07104)                                                                                           | 4.3                         |
| Sarathi-Serve: chunked prefill (OSDI '24)                                                                                            | 4.4                         |
| Speculative decoding (Leviathan et al., arXiv 2211.17192) + EAGLE                                                                    | 2.7/4.5                     |
| FlashAttention 1→3                                                                                                                   | 3.8/3.9                     |
| DistServe (OSDI '24) + Splitwise (ISCA '24)                                                                                          | 4.7                         |
| Mooncake (FAST '25 best paper)                                                                                                       | 4.7/6.4                     |
| ServerlessLLM (OSDI '24)                                                                                                             | 4.9                         |
| DeepSeek-V3 hardware reflections (arXiv 2505.09343)                                                                                  | 4.8/6.x                     |
| Survey: "Towards Efficient Generative LLM Serving" (arXiv 2312.15234) or "LLM Inference Serving: Recent Advances" (arXiv 2407.12391) | anytime — the map           |

## Current project slot

**Now:** [[Testing Inference Degredation]] — fp4 audits, token inflation, the 2x2. Backbone skills it draws on: Arc 1 (self-hosting removes the serving-stack confound), 5.2–5.3 (controlled precision arms), 4.10 (temp=0 nondeterminism). Project-specific reading and experiment design live in that note, not here.

**Next:** unknown, and that's fine. Candidates will fall out of problems hit while serving — cold starts, scheduling, cost modeling, something else entirely. When the slot changes, this note doesn't need to.

## Cadence heuristics

- Twice a week, sync days need a hypothesis result → schedule project-day runs to finish the evening before. Kick off long evals before bed; the GPU works while I sleep.
- Otherwise alternate: a project day, then 1–2 depth days. Depth arcs interleave fine (a CUDA unit and an engine unit share no state).
- Keep one rented-GPU template image alive; setup time is the enemy of the 1–2 hour day.
- Units are droppable and reorderable — the arc list is a menu, not a syllabus. When a problem opens a door, chase it; that's the point.

## Shelf

- JAX scaling book (inference + GPU chapters — already source of [[Inference]]/[[GPUs]])
- *Programming Massively Parallel Processors* (PMPP) + GPU MODE lectures/Discord
- Stanford CS336 (assignment 2 covers kernels/systems)
- vLLM blog + source; llama.cpp / llama2.c
- Thinking Machines, "Defeating Nondeterminism in LLM Inference"
- InferenceMAX (SemiAnalysis), Artificial Analysis provider pages
