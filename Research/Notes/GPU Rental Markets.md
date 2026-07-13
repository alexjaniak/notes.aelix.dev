# GPU Rental Markets

Research dump (July 2026) answering: *Where can I rent GPUs? How do you technically access a rental GPU? Can you write custom kernels without owning one?* From [[Questions]]. Related: [[GPUs]], [[Frontier Chips]].

## Where can I rent GPUs?

The market has stratified into ~5 layers with a 3–12x price spread for identical silicon (H100: ~$1.29/hr on marketplaces → $7–12/hr on hyperscalers; market median ~$2.45–3/hr). SemiAnalysis's ClusterMAX rating tracks **209 providers** — it's that crowded. Newest development: compute is becoming a listed commodity, with GPU futures launching on ICE and CME in 2026.

### Hyperscalers — pay for compliance/ecosystem, not compute value

- **AWS** — P5/P6 instances, Capacity Blocks for short-term reservations; H100 ~$3.90–6.88/hr on-demand
- **Azure** — ClusterMAX Gold; also rents heavily *from* neoclouds ($19B Nebius deal, plus Nscale, Lambda, CoreWeave)
- **GCP** — A3/A4 instances, TPUs as alternative, Dynamic Workload Scheduler
- **Oracle OCI** — Gold; bare-metal superclusters (incl. AMD MI300X), OpenAI Stargate anchor

### Neoclouds (GPU-native clouds)

- **CoreWeave** — the only ClusterMAX Platinum; K8s-native, earliest Blackwell/Rubin allocations, public; reserved ~$1.45/hr beats nearly everyone
- **Nebius** (ex-Yandex) — Gold; cheapest committed H200 ($2.30/hr), $19B Microsoft contract, EU+US
- **Lambda** — developer-first, "1-Click Clusters" with InfiniBand, IPO track
- **Crusoe** — stranded-energy angle, builds Stargate Abilene for OpenAI
- **Fluidstack** — Gold; lean team assembling giant training clusters fast (Mistral, Meta)
- **Together AI** — inference API brand + rentable Slurm/K8s GPU clusters
- **Voltage Park** — nonprofit-owned (~24k H100s), H100 from $1.99/hr
- **TensorWave** — AMD-only (MI300X/325X), $350M Series B
- **Verda** (ex-DataCrunch, Finland) — cheapest H100 spot floor (~$0.80–1.14/hr), EU green energy
- Mid-tier: **Hyperstack** (~$1.90/hr), **Vultr**, **DigitalOcean/Paperspace**, **Scaleway** (won the European Commission sovereign-cloud contract), **OVHcloud**, **CUDO** (~$1.82/hr), **Civo**, **Latitude.sh** ($1.66/hr bare metal), **Hetzner**, **Massed Compute**, **GMI Cloud** (building 1GW sovereign AI infra in Japan on Vera Rubin), **Corvex** (compliance-heavy Blackwell cloud), **WhiteFiber**, **Thunder Compute** ("VMware for GPUs" — virtualized H100 $1.38/hr)
- Bitcoin-miner pivots: **IREN** (~$500M AI-cloud ARR, direct GPU cloud), **Applied Digital** ($7.5B hyperscaler lease), **TeraWulf** ($19B Anthropic lease), **Hut 8**, **Bitdeer**, **Hive**

### Marketplaces / aggregators / brokers

- **Vast.ai** — the original "Airbnb of GPUs"; per-second billing, interruptible auctions, H100 from ~$1.33/hr, consumer cards from cents/hr; cheapest floor, reliability varies by host
- **RunPod** — dev-focused hybrid: Community vs Secure tiers, serverless with sub-200ms FlashBoot cold starts, Instant Clusters
- **SF Compute** — true order-book spot market for H100/H200 blocks; buy by the hour and *resell* unused time; owns no GPUs; positions as the physical base for compute derivatives
- **Shadeform** (YC S23) — one API/console over 20–30+ providers at list prices, no markup
- **Prime Intellect** — compute exchange over 12+ clouds + decentralized training + RL environments hub; $130M Series A @ $1B (July 2026)
- **Mithril** (ex-Foundry) — spot/preemption arbitrage "omnicloud"; customers Cursor, LG, Arc Institute
- **Hydra Host** — NVIDIA-backed bare-metal brokerage (Brokkr), $100M Series A
- **TensorDock** — vetted-host marketplace, full KVM VMs
- **NVIDIA DGX Cloud Lepton** — NVIDIA now runs its own rental marketplace routing developers to partner neocloud capacity
- Price comparison: getdeploying.com, gpucloudprices.com, computeprices.com

### Financialization (the novel 2026 wave, all early-stage)

- **Ornn** — GPU compute futures with ICE ("trade AI compute like oil")
- **Silicon Data** — SDH100RT H100 rental-price index; first listed compute futures with CME Group
- **Pluto** (pluto.trade) — YC-backed, 4-person startup; "first CFTC-regulated exchange for AI infrastructure"
- **Stoa Exchange** — cash-settled futures/options on GPU hardware
- **Capacity Derivatives** — OTC forwards/swaps on Canton blockchain
- **Compute Exchange (TCEX)** — auction-based exchange, co-founded with auction theorist Paul Milgrom

### Decentralized / DePIN

**Akash** (on-chain reverse auctions), **io.net** (Solana, pivoting enterprise), **Aethir** (real enterprise revenue), **Hyperbolic** (H100 at $1.29/hr — cheapest tracked in one July 2026 index), **Salad** (gamers' idle 4090s ~$0.20/hr), **Nosana**, **Spheron**, **Render** (3D rendering more than ML), **Gensyn** (verifiable decentralized training, testnet stage). SemiAnalysis rates essentially all DePIN "Underperforming" for serious multi-node training — they compete on price for inference/batch jobs.

### Serverless GPU (rent by the second)

**Modal** (Python-native, sub-second cold starts), **Replicate** (acquired by Cloudflare), **Baseten**, **fal.ai** (generative media), **Beam**, **Cerebrium**, **Koyeb**, **Lightning.ai**, **RunPod Serverless**. Adjacent: token-priced inference (rent *outcomes*, not GPUs) — see [[Serving Models]].

### Sovereign / regional

**HUMAIN** (Saudi, ~600k GPUs planned by 2030), **Core42/G42** (UAE), **Yotta** (India, 20k+ B300s), **Sakura** (Japan), **SMC/Firmus** (Singapore, immersion-cooled), **Scaleway**/**OVHcloud** (EU), **TELUS** (Canada). Fastest-growing new category in 2025–26.

## How do you technically access a rental GPU?

Four access modes:

1. **SSH into VM/bare metal** (Lambda, hyperscalers, CoreWeave dedicated) — provider preinstalls the NVIDIA driver; Lambda images ship drivers + CUDA + Jupyter. Standard workflow is VS Code/Cursor Remote-SSH. Hyperscaler gotcha: default GPU quota is 0, must request increases.
2. **Docker container rentals** (Vast.ai, RunPod pods) — your rental is literally a `docker create` on someone's host, with an SSH daemon or Jupyter injected. Key division of labor: the **host owns the kernel driver + `libcuda.so`; your container brings the CUDA toolkit** (`nvcc`, `libcudart`). Container CUDA version must be ≤ host driver's supported version.
3. **Managed Slurm/K8s clusters** (CoreWeave SUNK, DGX Cloud, reserved clusters) — SSH to a login node (no GPU), shared NFS home, `srun`/`sbatch` for GPU jobs, containers via Pyxis/Enroot.
4. **Serverless** (Modal, RunPod Serverless, Replicate) — never see a machine; decorate a Python function with `gpu="H100"`. Modal's stack: gVisor sandbox, lazy-loaded image FS, CUDA checkpoint/restore for sub-second cold starts.

## Can you write custom kernels without owning a GPU? Yes.

Compiling and running CUDA/Triton kernels works on **every** rental type, including serverless — people run full CUTLASS/B200 kernel labs on Modal. Timing-based benchmarking (CUDA events, `torch.profiler`, Nsight Systems timelines) also generally works.

The one real constraint is **hardware-counter profiling** (Nsight Compute `ncu`): since driver 418, GPU perf counters are admin-restricted (`ERR_NVGPUCTRPERM`). You need bare metal, a dedicated VM, or `--cap-add=SYS_ADMIN`:

- Bare metal / dedicated VMs (Lambda, CoreWeave dedicated, Vast VM-mode): full `ncu`/`nsys` works
- Container marketplaces (RunPod, Vast Docker mode): hit-or-miss; Vast's answer is "use a VM instance"
- Serverless: structurally blocked (no SYS_ADMIN in multi-tenant sandboxes)
- Avoid MIG/vGPU/fractional GPUs for kernel work — CUPTI profiling APIs unsupported, no P2P across slices. Rent a whole physical GPU.

### Cheap/free options for learning kernel dev

- **Google Colab** free T4 — nvcc preinstalled, `nvcc4jupyter` cell magic, srush's GPU-Puzzles; Kaggle similar
- **Modal** — $30/month free credits, per-second H100/B200 billing
- **Lightning AI** — free monthly GPU hours, SSH + any IDE
- **LeetGPU** — browser CUDA problems + free simulator for zero-cost correctness checks
- **Tensara** — competitive kernel challenges (CUDA/Triton/Mojo) on real T4/A100/H100, leaderboards vs PyTorch baselines
- **GPU MODE** Discord (~20k members) — free kernel submissions via popcorn-cli to donated T4→B200/MI300 hardware, ~15s turnaround, weekly lectures; all 60k+ submitted kernels open-sourced
- Raw SSH floor: RunPod/Vast spot RTX 3090/4090 from ~$0.10–0.40/hr
