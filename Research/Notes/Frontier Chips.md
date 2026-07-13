# Frontier Chips

Research dump (July 2026) answering: *What chips are being used at the frontier? Are SRAM machines hype?* From [[Questions]]. Related: [[GPUs]], [[Vera Rubin]], [[GPU Rental Markets]]. Key citations verified July 2026.

## What frontier labs actually train/serve on

**NVIDIA is still the default (~80% of accelerator market)**, but every hyperscaler and lab now runs a portfolio: NVIDIA + AMD + their own ASIC, with custom silicon eating *inference* first.

- **NVIDIA** — H100/H200 still workhorses (~$1.90/hr rental); Blackwell B200/GB200 NVL72 ramped mid-2025; GB300 NVL72 shipped Jan 2026 (288 GB HBM3e, claims 35x lower cost/token vs Hopper for agentic inference); Rubin ships 2026. Rubin CPX uses GDDR7 (not HBM) for prefill — an admission that inference phases want different memory tiers. Deployments: xAI Colossus (~200k GPUs, targeting 1M), OpenAI Stargate (~7 GW planned), Anthropic Claude on GB300 in Azure.
- **AMD** — MI300X → MI355X (GA on Oracle mid-2025) → MI450/Helios racks (2H 2026). Landmark deals: [**OpenAI–AMD 6 GW**](https://ir.amd.com/news-events/press-releases/detail/1260/amd-and-openai-announce-strategic-partnership-to-deploy-6-gigawatts-of-amd-gpus) (Oct 2025, warrant for 160M AMD shares), [**Meta–AMD up to 6 GW**](https://about.fb.com/news/2026/02/meta-amd-partner-longterm-ai-infrastructure-agreement/) (Feb 2026, custom MI450 variant, up to ~$60B over 5 years). Adopted by 7 of the 10 largest AI customers.
- **Google TPU** — [v7 Ironwood](https://cloud.google.com/blog/products/compute/ironwood-tpus-and-new-axion-based-vms-for-your-ai-workloads) is the first inference-first TPU (9,216-chip superpod, 1.77 PB shared HBM, GA Nov 2025). Gemini trained/served on TPUs. [**Anthropic committed up to 1M TPUs / >1 GW**](https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services) (Oct 2025, extended with a Google+Broadcom next-gen deal Apr 2026).
- **Amazon Trainium** — Trainium2 powers [Project Rainier](https://www.aboutamazon.com/news/aws/aws-project-rainier-ai-trainium-chips-compute-cluster) for Anthropic (>500k chips, scaling toward 1M); Trainium3 (3nm) used by both Anthropic and OpenAI. Notably: an [AWS–Cerebras collaboration](https://press.aboutamazon.com/aws/2026/3/aws-and-cerebras-collaboration-aims-to-set-a-new-standard-for-ai-inference-speed-and-performance-in-the-cloud) pairing **Trainium for prefill + Cerebras for decode** (Mar 2026).
- **Custom silicon** — Meta MTIA (hundreds of thousands deployed, new gen every 6 months; also [acquired RISC-V GPU startup **Rivos**](https://www.reuters.com/business/meta-buy-chip-startup-rivos-ai-effort-source-says-2025-09-30/) — price undisclosed; Rivos had been raising at ~$2B); [Microsoft Maia 200](https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/) (3nm, inference-only, claims 3x Trainium3 FP4); **OpenAI–Broadcom** 10 GW of OpenAI-designed accelerators (announced Oct 2025) — first chip ["Jalapeño" unveiled June 2026](https://openai.com/index/openai-broadcom-jalapeno-inference-chip/), design→tape-out in 9 months partly using OpenAI's own models; Huawei Ascend (DeepSeek's supply); SoftBank Izanagi (Graphcore+Ampere).

## Are SRAM machines hype? No — but they're a complement, not a GPU-killer

**The architecture in one line:** replace HBM (~8 TB/s, ~400ns) with on-chip SRAM (80 TB/s per Groq LPU; 21 PB/s aggregate per Cerebras wafer) as primary weight storage → extreme single-user decode speed, at the cost of capacity (SRAM ≈ 50–100x cost/bit vs DRAM; a 70B FP16 model needs ~574 Groq LPUs or multiple Cerebras wafers at 44 GB each).

### The receipts (2025–26 turned these from curiosities into $20B deals)

- [**NVIDIA licensed Groq's LPU technology**](https://www.cnbc.com/2025/12/24/nvidia-buying-ai-chip-startup-groq-for-about-20-billion-biggest-deal.html) (Dec 2025) — officially a *non-exclusive licensing agreement*, reported at ~$20B (investor-sourced figure, never confirmed by NVIDIA) — and acqui-hired founder Jonathan Ross and president Sunny Madra; shipped "Groq LPU 3" in an LPX rack (96 LPUs, Mar 2026) pairing with Vera Rubin. The strongest validation the SRAM-decode thesis has received. Groq itself continues independently.
- **Cerebras**: revenue $290M (2024) → $510M (2025); [IPO May 2026](https://www.cnbc.com/2026/05/14/cerebras-cbrs-stock-trade-nasdaq-ipo.html) (Nasdaq: CBRS, ~$5.6B raised, $6.4B gross with greenshoe — largest US tech IPO since Uber in 2019). [**OpenAI committed to 750 MW of Cerebras inference through 2028**](https://openai.com/index/cerebras-partnership/), [reported at >$20B](https://www.reuters.com/technology/openai-spend-more-than-20-billion-cerebras-chips-receive-equity-stake-2026-04-17/) with an equity stake. GPT-5.3-Codex-Spark runs on Cerebras at ~2,000 tok/s/user. Risks: extreme customer concentration (was 85% UAE/G42, now OpenAI), still operating-margin negative.
- **AWS + Cerebras** (above) and **d-Matrix SquadRack**: GPU-prefill/SRAM-decode disaggregated racks are becoming the consensus architecture.
- Groq context: GroqCloud 5M developers, $1.5B Saudi commitment, 19,000-LPU Dammam DC built in 51 days.

### Why they win

Decode is memory-bandwidth-bound (arithmetic intensity ~2 FLOPs/byte; an H100 runs at <1% FLOP utilization in low-batch decode — see [[Inference]]). SRAM bandwidth gives 3–20x per-user speed: Groq 280–350 tok/s vs 60–100 on H100 (Llama-70B); Cerebras ~2,100–2,700 tok/s vs ~1,000 on B200. Deterministic latency (compiler-scheduled to the clock cycle). No HBM, no CoWoS — the scarce supply-chain inputs everyone else fights over, so capacity is *additive* (a major reason OpenAI signed Cerebras). Agentic workloads (~15x token volume, latency-compounding chains) made per-user tok/s monetizable — labs now sell fast/priority/standard/batch tiers of the same weights.

### Why they lose

- **Capacity/cost per GB**: 44 GB/wafer (Cerebras) vs 288 GB HBM3e on one B300 package; Llama-70B ≈ 4 Cerebras racks ($2–3M each) vs one DGX at ~1/10 the cost.
- **Batch-throughput economics**: GPUs amortize one weight-load across huge batches; a GB300 rack has 20 TB of HBM for weights + massive KV cache. SemiAnalysis: throughput per {chip, watt, $} decisively favors HBM at high concurrency.
- **KV cache / long context**: KV must share the tiny SRAM with weights; 256k–1M contexts are hostile. Cerebras off-wafer I/O is only 150 GB/s.
- **SRAM scaling is dead**: TSMC N3E/N2 give ~zero SRAM density gains (WSE-2→3 went 40→44 GB); models grow faster than SRAM.
- **MoE awkwardness**: parking mostly-idle expert weights in premium SRAM is bad economics.
- Business-model tell: both Groq and Cerebras failed to sell chips and pivoted to selling tokens.

**Verdict circulating in credible analysis (SemiAnalysis, Gimlet Labs):** a real, monetizable niche — premium low-latency decode, ~10% of inference — inside a heterogeneous architecture: HBM/GDDR compute for training+prefill+batch, near-memory (SRAM or stacked DRAM) for decode.

## Other chip startups and their differentiators

- **d-Matrix** — [Corsair](https://www.d-matrix.ai/announcements/d-matrix-corsair-ai-inference-platform-enters-full-production-to-meet-customer-demand/) digital in-memory-compute decode chiplets (SRAM + LPDDR5, deliberately no HBM/CoWoS), full production June 2026; *cooperative* positioning next to NVIDIA prefill (SquadRack); ~$2B valuation, Microsoft-backed; next chip "Raptor" claims 3D-stacked DRAM at 10x HBM4 bandwidth
- **Etched** — Sohu, transformer-only ASIC ("architecture etched in silicon"); [claims $1B booked orders at a $5B valuation](https://techcrunch.com/2026/06/30/nvidia-competitor-etched-hits-5b-valuation-1b-in-sales-for-ai-chip/); revenue not yet proven
- **Tenstorrent** (Jim Keller) — RISC-V, GDDR6 not HBM, standard Ethernet, fully open-source stack; [Galaxy Blackhole from $110k/server](https://tenstorrent.com/en/hardware/galaxy); competes on cost/openness, sovereign-AI angle
- **Positron AI** — LPDDR5x commodity memory at near-100% utilization instead of HBM, US-fabbed; [$230M Series B at >$1B](https://techcrunch.com/2026/02/04/exclusive-positron-raises-230m-series-b-to-take-on-nvidias-ai-chips/) (Feb 2026, QIA + Arm); Asimov chip (2027) targets 2 TB memory per accelerator — memory-capacity-first bet
- **MatX** (ex-Google TPU) — hybrid: weights in SRAM, KV cache in HBM; targets frontier labs; nothing shipped publicly
- **SambaNova** — RDU dataflow + HBM; valuation whiplash $5B→$2.4B→[$11B](https://sambanova.ai/press/sambanova-completes-first-close-of-1b-financing-at-11b-valuation) ($1B Series F first close, July 2026, after near-acquisition by Intel); SN50 ships 2H26
- **Furiosa** (Korea) — RNGD tensor-contraction processor; LG validation (2.25x perf/W claim); rejected Meta's $800M acquisition offer
- **Graphcore** — commercial failure as independent (acquired by SoftBank ~$500–600M vs $2.77B peak); now captive, building Izanagi for Stargate
- **Rivos** — RISC-V AI GPU, acquired by Meta into MTIA
- **Extropic** — thermodynamic sampling units; research-stage
- **Taalas** (hardwired model-per-chip), **HyperAccel** (Korean LPU) — early
