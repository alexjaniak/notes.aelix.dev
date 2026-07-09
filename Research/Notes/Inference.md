* Relevant for actual use, model eval, and RL
	* What? 
* Training is a one-time cost, inference is repeated many times. Efficiency matters the most here. 
* OpenAI processes ~8.6T tokens per day, while DeepSeek v4 was trained on 32T tokens. 

Open-source packages: 
* vLLM, good default
* SGLang, good for agentic workloads
* TensorRT-LLM, NVIDIA, optimized for GPUs
* llama.cpp, C++, CPU inference, runs locally
* Anymore? 

Metrics:
* Time-to-first-token (TTFT): period before generation
* Latency (seconds/token): host fast tokens appear for one query
* Throughput (tokens/second): how fast tokens appear for many queries (batch processing)

Unfortunately, inference is sequential and so you can't parallelize over generation. So its harder to utilize compute. Training allows you to parallelize over the entire sequence.

At low batch sizes, the system tends to be memory bound. This is basically what happens with inference. You're batch dimension is smaller and then more time is spend transferring bytes than computing FLOPs. 

For each token, you need O(T^2) FLOPS for the forward pass. So to generate T tokens, it takes O(T^3). Horrific

This is in part because you're recomputing the same probabilities and recreating the K & V matrices O(T^2). You can instead store a KV cache in HBM. How big is this? 

For MLP, its roughly BxT for MLP. 
For Attention its SxT / (S + T)

Two stages of inference: 
Prefill: given a prompt encode into vectors (parallelizable like in training)

Generation: Generate new response tokens (sequential)

Prefill is compute bound, generation is memory bound. In fact, the biggest bottleneck is generation attention intesity, which is impossible to improve for transformers.

Grouped Quert attention groups query matrices into less heads of keys and values. Effectively reducing KV cache size, decresasing memory bottleneck

Multi-head latent attention MLA
Compress the vector, then apply KV. not compatible with RoPE so add extra dimension

Cross layer attetnion, only compute KVs for subset of the layers and share them

Sliding window attention, look at the last K tokens, KV cache becomes indp of sequence length, good for long contexts, but reduces effectivenss

Compressed sparse attention: compress every m tokens into 1
Deepseek sparse atteiton selects top k
Heavily compressed attention does even more compression

quantization reduces the percision of numbers
quantization aware training uses quantization forward pass to sim quant errs

























