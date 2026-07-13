# Frontier Chips

Research dump (July 2026) answering: *What chips are being used at the frontier? Are SRAM machines hype?* From [[Questions]]. Related: [[GPUs]], [[Vera Rubin]], [[GPU Rental Markets]].

## What frontier labs actually train/serve on

**NVIDIA is still the default (~80% of accelerator market)**, but every hyperscaler and lab now runs a portfolio: NVIDIA + AMD + their own ASIC, with custom silicon eating *inference* first.

- **NVIDIA** — H100/H200 still workhorses (~$1.90/hr rental); Blackwell B200/GB200 NVL72 ramped mid-2025; GB300 NVL72 shipped Jan 2026 (288 GB HBM3e, claims 35x lower cost/token vs Hopper for agentic inference); Rubin ships 2026. Rubin CPX uses GDDR7 (not HBM) for prefill — an admission that inference phases want different memory tiers. Deployments: xAI Colossus (~200k GPUs, targeting 1M), OpenAI Stargate (~7 GW planned), Anthropic Claude on GB300 in Azure.
- **AMD** — MI300X → MI355X (GA on Oracle mid-2025) → MI450/Helios racks (2H 2026). Landmark deals: **OpenAI–AMD 6 GW** (Oct 2025, warrant for 160M AMD shares), **Meta–AMD 6 GW** (Feb 2026, custom MI450 variant). Adopted by 7 of the 10 largest AI customers.
- **Google TPU** — v7 Ironwood is the first inference-first TPU (9,216-chip superpod, 1.77 PB shared HBM, GA Nov 2025). Gemini trained/served on TPUs. **Anthropic committed up to 1M TPUs / >1 GW** (Oct 2025, extended with a Google+Broadcom next-gen deal Apr 2026).
- **Amazon Trainium** — Trainium2 powers Project Rainier for Anthropic (~500k chips); Trainium3 (3nm) used by both Anthropic and OpenAI. Notably: AWS deal pairing **Trainium3 for prefill + Cerebras for decode**.
- **Custom silicon** — Meta MTIA (hundreds of thousands deployed, new gen every 6 months; also acquired RISC-V GPU startup **Rivos** ~$2B); Microsoft Maia 200 (3nm, inference-only, claims 3x Trainium3 FP4); **OpenAI–Broadcom** 10 GW of OpenAI-designed accelerators — first chip "Jalapeño" unveiled June 2026, design→tape-out in 9 months partly using OpenAI's own models; Huawei Ascend (DeepSeek's supply); SoftBank Izanagi (Graphcore+Ampere).

## Are SRAM machines hype? No — but they're a complement, not a GPU-killer

**The architecture in one line:** replace HBM (~8 TB/s, ~400ns) with on-chip SRAM (80 TB/s per Groq LPU; 21 PB/s aggregate per Cerebras wafer) as primary weight storage → extreme single-user decode speed, at the cost of capacity (SRAM ≈ 50–100x cost/bit vs DRAM; a 70B FP16 model needs ~574 Groq LPUs or multiple Cerebras wafers at 44 GB each).

### The receipts (2025–26 turned these from curiosities into $20B deals)

- **NVIDIA licensed Groq's LPU IP for ~$20B** (Dec 2025) and acqui-hired founder Jonathan Ross; shipped "Groq LPU 3" in an LPX rack (96 LPUs, Mar 2026) pairing with Vera Rubin. The strongest validation the SRAM-decode thesis has received.
- **Cerebras**: revenue $290M (2024) → $510M (2025); IPO May 2026 — largest semi IPO ever ($6.4B raised, ~$40–65B valuation). **OpenAI committed >$20B / 750 MW of Cerebras inference through 2028** (option to 2 GW). GPT-5.3-Codex-Spark runs on Cerebras at ~2,000 tok/s/user. Risks: extreme customer concentration (was 85% UAE/G42, now OpenAI), still operating-margin negative.
- **AWS + Cerebras, d-Matrix SquadRack**: GPU-prefill/SRAM-decode disaggregated racks are becoming the consensus architecture.
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

- **d-Matrix** — Corsair digital in-memory-compute decode chiplets (SRAM + LPDDR5, deliberately no HBM/CoWoS); *cooperative* positioning next to NVIDIA prefill; ~$2B valuation, Microsoft-backed; next chip "Raptor" claims 3D-stacked DRAM at 10x HBM4 bandwidth
- **Etched** — Sohu, transformer-only ASIC ("architecture etched in silicon"); claims $1B booked orders, $5B valuation; revenue not yet proven
- **Tenstorrent** (Jim Keller) — RISC-V, GDDR6 not HBM, standard Ethernet, fully open-source stack; Galaxy Blackhole $110k/server; competes on cost/openness, sovereign-AI angle
- **Positron AI** — LPDDR5x commodity memory at near-100% utilization instead of HBM, US-fabbed; Asimov chip (2027) targets 2 TB memory per accelerator — memory-capacity-first bet
- **MatX** (ex-Google TPU) — hybrid: weights in SRAM, KV cache in HBM; targets frontier labs; nothing shipped publicly
- **SambaNova** — RDU dataflow + HBM; valuation whiplash $5B→$2.4B→$11B (July 2026, after near-acquisition by Intel); SN50 ships 2H26
- **Furiosa** (Korea) — RNGD tensor-contraction processor; LG validation (2.25x perf/W claim); rejected Meta's $800M acquisition offer
- **Graphcore** — commercial failure as independent (acquired by SoftBank ~$500–600M vs $2.77B peak); now captive, building Izanagi for Stargate
- **Rivos** — RISC-V AI GPU, acquired by Meta into MTIA
- **Extropic** — thermodynamic sampling units; research-stage
- **Taalas** (hardwired model-per-chip), **HyperAccel** (Korean LPU) — early
