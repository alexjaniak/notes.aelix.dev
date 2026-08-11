Day 9/45 of Inference Engineering: Benchmarking my from-scratch inference server against vLLM Missed a few days, it was my birthday and had friends visiting so was thoroughly occupied. In the meantime, I built nano-vllm: a tiny LLM inference server written from scratch to learn how vLLM actually works. The generation loop is ~370 lines of readable Python. [https://github.com/alexjaniak/nano-vllm](https://t.co/QdNnJMJ6BG) What's in it so far: - FastAPI server, model worker in its own process - SSE token streaming - Per-sequence KV cache (prefill once, then one token per forward pass) - Interactive terminal client that streams tokens live Then I put it head-to-head against real vLLM on my GTX 1660 Super, 6GB, Qwen3-1.7B fp16, 8 prompts, & 50 max tokens. A quick but weak setup. Time to first token: 430.3ms vs 335.6ms (vLLM 1.3x ahead) Single stream: 22.9 vs 35.7 tok/s (vLLM 1.6x ahead) 8 concurrent prompts: 22.5 vs 15.4 tok/s (nano-vllm took this one) The entire test has a big asterisk. To get vLLM running on a 1660 at all, I had to disable its sampling kernel and turn off CUDA kernels. Also, the 1660 doesn't have bf16 support, so I was forced to use fp16. 6 GB is not much memory! vLLM was definitely not in its natural habitat. Despite all that, I'm actually quite shocked by how close the results were. I'll likely keep building the nano-vllm server and learn by adding more features. Guess we'll have to see how close I can get. Pictured: (1) the benchmark scoreboard (2) an architecture diagram


**Model:** DeepSeek-V4-Flash-Base — 284B total parameters, ~13B active (MoE), FP8 weights (~295GB), the pretrained base checkpoint (no instruct tuning). Served with the fp8_ds_mla KV cache format V4's hybrid attention requires.

**Hardware:** 4× H200 SXM-class (140GB each, 564GB total VRAM) on a Vast.ai marketplace box in Canada — $15.71/hr. Weights sharded across the 4 GPUs via tensor parallelism (~75GB/GPU), with vLLM pre-reserving the rest for KV cache.

**Stack:** vLLM v0.26.0, `-tp 4 --kv-cache-dtype fp8 --max-model-len 16384`, OpenAI-compatible `/v1/completions` endpoint, exposed over HTTPS via a Cloudflare tunnel. Client: Cotabby on Mac hitting it for tab-autocomplete, 10-token greedy completions.

Day **10-11**/45 of Inference Engineering: **Self-Hosted System-wide Intelligent Tab-Autocomplete**

Took a detour in the last two days to tinker on something with @oogway: an intelligent system-wide autocomplete. That is, tab-autocomplete (cursor-like next-word prediction) in *every* text box on my computer.

I've tested both Cotabby and CoTypist for a few days and found both products interesting but somewhat lacking. What kept me from removing them was that *sometimes* the autocomplete was good enough that I felt like my use of a computer was completely transformed. I'd just be holding tab and watching the text *I wanted* appear.

Unfortunately, the models that ran locally on my 24GB M4 Mac (Gemma 4 E2B, Qwen 3 1.7B), while responsive, were incredibly stupid — especially on anything slightly technical. I'm an engineer, so that's a problem.

When I booted up smarter models via third-party inference providers (GPT 5.6 Luna, K3, DeepSeek v4-Flash), the complete was too slow and responded like a chat-bot (Q&A-style). Horrific.

My guess was that If I used a smart base-model, (a model not fine-tuned for chat), the autocomplete would be more useful more often.

I couldn't find a single inference provider serving a base model so I hosted my own: **DeepSeek V4-Flash-Base on 4xH200s SXM using vLLM**. 

I used the autocomplete to write this post and I naturally tabbed >350 times! I got a mean TTFT of ~150ms during natural use. Apparently 100-300ms is considered perceptible but still responsive for humans, which is consistent with my experience. Regarding intelligence, it fairs well and can somewhat handle the technical stuff. It's not perfect, but leagues better than the previous setup. My hypothesis was correct and I find myself tabbing way more-often 😎!

For the "front-end"—the "ghost-text" and context collection—I used CoTabby and plugged my OpenAI-compatible endpoint into it. It was nice not having to build it myself but both are pretty rough. The context that's fed in gets cut-off and doesn't include things from other tabs/windows I have up. This is definitely the bottleneck. Potentially @screenpipe to the rescue? 

For cost, it's quite expensive. I'm paying ~$15/hr for compute which is ~$10k/month 😵‍💫. I'm basically paying for the VRAM since the GPUs are idle most of the time.

This is a proof- of-concept so I'm not too worried about it. I'm sure there are ways to optimize it














ery soon today's frontier intelligence will be dirt-cheap, and we might even have Fable-level intelligence running locally. System-wide autocomplete is a killer app for me, and with this new setup, really reminds me of the early days of Copilot & Cursor. Imagine if this was fine-tuned on your own data! 

Till next time!

