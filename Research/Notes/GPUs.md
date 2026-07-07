- Assumption that bigger matrix multiplies are always better — not always
- Need to understand how GPUs work to make fast algorithms
- Need to understand hardware model to make an effective model. Big part of scaling is using resources effectively.
- Compute leads to more predictable performance gains for language models. Aka scaling

* Dennard scaling, clock speed increase and smaller transitors decreaseso compute scaled. 2000s is when this tapped out. # of transistors has increased, but smaller transistors doesn't mean they go faster
* Parallel scaling with GPUs has scaled 1000x in 10 years. Instead of serially increasing speed via clock speed, we do horizontal scaling
* Tensor cores, structured sparsity, FP8 ... hardware improvements push parallel scaling

How is a GPU different from a CPU? 
* CPUs optimize for few fast threads while GPU for many
* CPUs has big control units for complex branching logic and control flows
* Control Unit (CU) = directs operation, routes data, and handles instructions
* Arithemtic Logic Unit (ALUs) = execution engine, arithemtic, logic and bitwise operations. basically hte math
* GPU is a about throughput not speed for a single thread. CPU optimize for latency while GPU for throughput
* SM, streaming multiprocessor, sorta like a core / compute unit that independantly execute 'blocks' jobs. Each SM has many SPs (streaming processors) that can execute 'threads in parallel'. 
* The closer the memory to the SM, the faster. L1 and shared memory is inside the SM. L2 is on die,and global is next to the GPU
* The "GB" memory on the GPU are talking abouy global memory
* Why don't you build a whole chip out of a shared memory, its expensive and power hungry. 100x more epensive but only 8x faster
* Groq = giant amount of SRAM
* are L1 L2 and Shared all in an SM? How many SMs instead a GPU? how many SPs in an SM? 


Thread: do the work, in parallel. Different input but the same instructions
Blocks: groups of threads, run on a SM w/ its own shared memory
Warp: Threads always execute in a 'warp' of 32 consequtively numbered treads each. Scheduling unit. Decreasing overhead of a scheduling unit

Scheduler decides which warp executes next

Device code can R/w per-thread registers, local memory, per-block shared memory. R/W only per-grid constant memory.
Host can transfer data to/from per grid global and constant memory.

So blocks use shared memory. Threads don't share registers aka local memory. All share global / constant memory

What are TPUs? Tensor Processor Unit
* Simpler, more optimized for machine learning. lighter control units and much biger Matrix Multiple Unit (MXU) (the driver of FLOP/s. 
* Scaler Unit acts like a CPU 'dispatching' instructions
* VPU performs elementwise ops (activations) loads data into MXU
* HBM is kinda the same
* GPU has more SMs and TPUS have fewer TCs
* VPU is like the warp scheduler
* Vmem, vector memory is fast on-chip cache memory aka L1 SMEM
* Main diff, 
* Tensor core is SM in TPU land, but matrix mult for GPU land
* GPU more flexible, smaller more many TCs, but TPU much bigger MXU but less flexible, has to big matmuls.

GPU s have easy scalibility because just use more SMs
programming is deceptively easy
use SIMT model. wtf is that? single instruction multiple threads. huh
threads are lightweight and can be stopped and started easily. its quick to swap out warps

early days, programmable shaders hacked to do matmuls. different rending gave faster matmuls

as of v100, matmul aka tensor core became specialized. 10x faster than other flop
GPUs are getting faster and faster, overal compute throughput is getting faster FLOP. Memory bandwidth aka how fast you can transfer data and parallellism (multiple GPUs) moving really slowly. 
	because of this, much bigger gap between compute & memory. Most optimizations are therefore memory optimizations

SRAM vs DRAM? what is SRAM and DRAM? 
L2 Cache on TPUs is much faster

prefill decode disaggregation. aka different chips for each. Even different layers might use different chips. See this kind of stuff in inference the most because its the most memory bound.

why has memory grown so much more slowly? no answer

bigger matrixs dne more FLOP/s in all cases. 
	up untill a certain point, you'll be memory limited
		at some point you'll have enough work that compute units get completed saturated. at some point adding more work doesn't benefit you

How do you make GPUs go fast
1. control divergence
	1. Every thread needs to execute the same instruction, so for a conditional. if you have a if-else, you need to rerun the conditional twice for each part of the if-else. 
	2. You'll want to do masks instead of conditionals.
	3. Not necessary memory but just utilizing computations more
2. low precision computation
	1. decreasing precision means moving less bits so means less memory bottleneck lol
	2. not all operations can be low precisio
	3. how does the chip do this? do you need special circuits?
	4. FP8 and lower, you'll have many different types of pericison formats.
	5. Blackwell has MXFP8 have many scaling factors
		1. What are scaling factors lol?
		2. issues with transposing?
	6. what is quantizing?
	7. is low precision for training or testing?
3. operator fusion
	1. torch.compile automatically fuse base things into a single cuda call
4. recomputation
	1. instead of storing activations, throw them out and re-compute everything for the backwards pass 
5. coalescing memory
	1. DRAM reads in 'burst mode', which means each read igives you many bytes
	2. organize matrices to use this
6. tiling
	1. group and order threads to minimize global memory access
	2. group submatrices and operated them. 
	3. Tile sizes need to divide matrix size or low utliziation
	4. tiles are fast if bursts align with the matrix
	5. larger matrices need to dvide well for the tiling / bursts, and this woukd better for when the shape is divizible by K, the more alingned with the burst the better. high utilization
		1. also matters how many SMs an you fit in. this is wave quantizations

Flash attention
* Tile matrix mult
* online softmax, so compute as you go
* fuse exponential
* recompute on the backwards pass
