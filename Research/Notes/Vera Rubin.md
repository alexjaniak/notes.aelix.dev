* Announced at CES 2026, full production as of Q1 2026, availability 2H 2026. 
* A suite of 6 platform products: Rubin GPU, Vera CPU, NVLink 6 Switch, ConnectX-9, BlueField-4, and Spectrum-6.
* NVIDIA have been focusing more on system & rack level design. They hope to make racks a unit of compute, with the hardware and software in extreme co-design. 

| Chip                           | What it is                                                                                    | Why it exists                                                                                               |
| ------------------------------ | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Rubin GPU**                  | Flagship datacenter GPU (dual-die, HBM4)                                                      | The core compute engine; ~5x Blackwell inference                                                            |
| **Groq 3 LPU**                 | Language Processing Unit. SRAM-based inference chip (from NVIDIA's Dec 2025 Groq acquisition) | 500MB on-chip SRAM @ 150 TB/s per chip. Fights the HBM memory wall for decode. The 7th chip of the platform |
| **Vera CPU**                   | 88-core custom Arm CPU                                                                        | Power-efficient host CPU, tightly coupled to GPUs via NVLink-C2C (chip-to-chip interconnect)                |
| **NVLink 6 Switch**            | Scale-**up** interconnect                                                                     | Makes 72 GPUs in a rack act as one memory/compute domain                                                    |
| **ConnectX-9 SuperNIC**        | Scale-**out** Network Interface Card                                                          | Node-to-node networking across racks                                                                        |
| **BlueField-4 DPU**            | Data processing unit                                                                          | Offloads networking/storage/security from CPUs; software-defined infra                                      |
| **Spectrum-6 Ethernet Switch** | Scale-out Ethernet (co-packaged optics)                                                       | Connects racks into pods/clusters efficiently                                                               |

| Board               | What it is                  | Why it exists                                                                                                         |
| ------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **HGX Rubin NVL8**  | 8-GPU baseboard with NVLink | For orgs that want Rubin GPUs but their own servers/CPUs — pairs with x86 (Intel/AMD) or Vera. The OEM building block |
| **Vera Rubin NVL4** | 4 GPUs + 2 Vera CPUs module | Smaller HPC/scientific-computing form factor for MGX servers                                                          |

| Rack                 | What it is                                               | Why it exists                                                                                                                                                                               |
| -------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Vera Rubin NVL72** | Flagship rack: 72 GPUs + 36 Vera CPUs, one NVLink domain | Max scale-up — whole rack runs as a single coherent AI engine, no model partitioning. Fully liquid-cooled, cable-free modular trays                                                         |
| **Groq 3 LPX**       | 256 Groq 3 LPUs  in a rack; deployed alongside NVL72     | Decode is bandwidth-bound — LPX offloads token generation (FFN/MoE) to SRAM while Rubin GPUs handle prefill + attention. ~35x throughput/MW claim. Supersedes the earlier Rubin CPX concept |
| **Vera CPU rack**    | 256 Vera CPUs, no GPUs                                   | CPU capacity for RL and agentic AI — sandboxes, tool calls, evals, orchestration (~22,500 concurrent sandboxes)                                                                             |

| TurnKey                  | What it is                                                    | Why it exists                                                                                                               |
| ------------------------ | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **DGX Vera Rubin NVL72** | The NVL72 rack as a full NVIDIA appliance                     | Hardware + Mission Control software + support, zero integration work                                                        |
| **DGX Rubin NVL8**       | NVL8 in a liquid-cooled x86 system                            | On-ramp to Rubin for orgs staying on x86                                                                                    |
| **DGX SuperPOD**         | Multi-rack pod: 14x NVL72 (1,008 GPUs) or 64x NVL8 (512 GPUs) | The "AI factory in a box" — networking, storage, software included. Also the reference blueprint for hyperscale deployments |

 When will the first models trained on either be released?

## Rubin GPU



