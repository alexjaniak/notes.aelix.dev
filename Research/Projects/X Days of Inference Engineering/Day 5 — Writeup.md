# Day 5 Writeup — Counting dots (a learning day)

*Learning day — no GPU, no new measurements; every number cited below was measured on Days 1–4. Sources: the JAX scaling book ("Counting Dots" / rooflines) and Strang, *Linear Algebra and Its Applications* (multiplication by rows, columns, columns-times-rows, and blocks). Follows [[Day 4 — Writeup]]; quantization moves to Day 6.*

**The question:** Day 4's roofline predicted batch-1 decode within 6% and missed batch-64 by 2×. Both numbers come from the same three-row FLOP table. Today is about actually learning that table — how to count the FLOPs and bytes of a matmul, and why the *same* multiplication can be computed four different ways that cost identical FLOPs but wildly different memory traffic.

## Counting dots

Everything in a transformer forward pass reduces to dot products. Count those and you've counted everything:

- A **dot product** of two P-vectors is P multiplies and P adds: **2P FLOPs**.
- A **matrix-vector product** Ax (A is [N, P]) is one dot product per row: **2NP FLOPs**.
- A **matrix-matrix product** AB ([N, P] × [P, M]) is one matvec per column of B: **2NPM FLOPs**.

Now count the bytes each one has to touch (bf16, 2 bytes/element, ignoring the output write):

| Operation | FLOPs | Data (elements) | Intensity (FLOPs/byte) |
|---|---|---|---|
| x · y | 2P | 2P | ~0.5 |
| Ax | 2NP | NP + P | **~1** |
| AB | 2NPM | NP + PM | grows with shape |

The first two rows are stuck: a matvec does 2 FLOPs for every 2 bytes of weights it streams, forever, no matter how big the matrix. **Batch-1 decode is the middle row.** That's Day 1 in one line: the FLOPs are irrelevant, the cost is streaming NP weight bytes, so tok/s = bandwidth ÷ bytes — and the M4's spec sheet predicted 20.8 tok/s within 13%.

The third row is different in kind. For a square N×N matmul, compute scales **O(N³)** while data scales **O(N²)** — make the problem bigger and the FLOPs-per-byte ratio *improves*. Matmul is nearly the only important operation with this property, and it's a fair summary of why the entire industry builds architectures out of it: it's the op that can actually saturate a chip's compute instead of its memory bus.

## Where the batch dimension hides

The serving-relevant shape is [B, P] × [P, M] — B tokens hitting a P×M weight matrix. FLOPs: 2BPM. Bytes: ~2(PM + BP + BM), dominated by the PM weights when B is small. So:

> **arithmetic intensity ≈ B** — the batch size, in FLOPs per byte.

Every machine has a ridge point where it stops being bandwidth-starved: peak FLOPs ÷ bandwidth. The week's numbers, re-derived from this one idea:

- **M4:** 4.4 TFLOPS ÷ 120 GB/s ≈ **37**. Decode (B=1, intensity ~1) sits 37× below the ridge — hence Day 1's bandwidth-only prediction working. Prefill batches 512 prompt tokens → intensity ~512 → compute-bound, hence the *other* roofline (270 tok/s ceiling, measured 85% of it).
- **A100:** ~161 (Day 3's derivation). Day 3's saturation constant B ≈ 111 is this same ratio wearing KV-cache clothes.
- **Day 4's 2× miss:** every request's 1,000-token prompt is a [1000, P] × [P, M] matmul — intensity ~1000, pure compute — that a decode-bandwidth model priced at zero. "Serving is not decode" is the statement that a real endpoint mixes intensity-1 work (decode) with intensity-1000 work (prefill) and you have to bill both.

## Four ways to compute the same product

Strang's observation, and the part I hadn't appreciated: AB has (at least) four equally valid computation orders, all costing exactly 2NPM FLOPs.

1. **Dot-product form (the definition).** C[i,j] = (row i of A) · (column j of B). NM little dot products of length P. This is how everyone learns it and how almost nothing computes it.
2. **Column form.** Column j of C = A × (column j of B) — every column of the output is a *combination of the columns of A*. M matvecs.
3. **Row form.** Row i of C = (row i of A) × B — every row of the output is a combination of the rows of B. N vector-matrix products.
4. **Column-times-row (outer product) form.** AB = Σₖ (column k of A)(row k of B) — a sum of P rank-1 matrices. Each term touches one column of A and one row of B and updates the *entire* output.

Same arithmetic, same 2NPM — but each form walks memory in a completely different order, and which entries get *reused* while resident in fast memory differs radically. FLOPs are fixed by the shapes; **memory traffic is a property of the loop order you choose.** That asymmetry is the entire subject of kernel engineering.

## Blocks: the form hardware actually uses

Strang's last multiplication rule is the license for all of it: **partition A and B into blocks, and block multiplication follows the same rules as if the blocks were scalars.**

> [A₁₁ A₁₂; A₂₁ A₂₂] × [B₁₁ B₁₂; B₂₁ B₂₂] → C₁₁ = A₁₁B₁₁ + A₁₂B₂₁, etc.

Composed with form 4, this is literally a tiled GPU kernel: loop over the contraction dimension in chunks, load one block-column of A and one block-row of B into fast memory, accumulate their (outer) product into the output tile, repeat. Each loaded block gets reused across the whole tile before being evicted — that's how the O(N³)/O(N²) reuse that exists *on paper* gets *realized* through a cache hierarchy that's far too small to hold the matrices. Tensor cores go one level deeper: the hardware primitive is itself a tiny block matmul (16×16-ish), so the whole computation is Strang's block rule applied recursively — blocks of blocks of blocks.

The theoretical punchline: the FLOP count was never the interesting number. It's fixed. Everything Arc 3 will be about — coalescing, tiling, shared memory, the climb from 1% of cuBLAS to 90% — is choosing *which of these equivalent orderings* to walk, to move the fewest bytes per FLOP. The math says the orderings are interchangeable; the memory hierarchy says they're not even close.

## What surprised me

- Arithmetic intensity of a batched matmul is just… B. The saturation constant Day 3 derived from KV-cache accounting (B ≈ 111) and the ridge point from the spec sheet (~161) are the same idea approached from two directions.
- The dot-product *definition* of matmul is the worst of the four forms for a machine — no operand gets meaningful reuse. The form nobody teaches first (columns × rows, rank-1 updates) is the one hardware actually resembles.
- "Compute scales cubically, data quadratically" is an unreasonably deep sentence: it's the reason big matmuls are the one workload that can outrun a memory bus, and therefore the reason models are towers of matmuls in the first place. Hardware and architecture co-evolved around this single asymmetry.

## X post — final draft (Alex's version, edited)

> Day 5/45 of Inference Engineering: Numerical Notes from Linear Algebra
>
> A dot product between two vectors of size [P] takes 2P FLOPs and requires a transfer of 2P items of memory. Why?
>
> By definition, each pair of items is multiplied together element-wise, and everything is summed together. That's P multiplications and P additions.
>
> A mult. between a matrix of size [N, P] and a vector of [P] is N dot products, so N(2P) operations and NP+P item transfers. (N is the output dimension, not a batch. The "batch" in serving is how many *vectors* you push through at once, and a matvec is batch 1.)
>
> A mult. between a matrix of size [N, P] and another matrix of [P, M] is M matrix-vector operations, so MN(2P) operations and NP+PM item transfers. Stack B vectors instead of one and that's what a batch actually is: [B, P] × [P, M], with B sitting where N was.
>
> For [N, N] matrices, this operation has cubic complexity and quadratic memory transfer, and that's the unusual part. For most operations (matvec included), moving the bytes costs more than doing the math. Matmul is the rare op where growing the problem makes the compute pull *ahead* of the transfer. That's why big matmuls can saturate a chip instead of its memory bus, and a decent one-line explanation of why models are towers of matmuls.
>
> Ok, but what about a much bigger multi-dimensional tensor? 👀
>
> Same rule, dressed up. When two tensors share dimensions, each shared dim is either CONTRACTING (summed over, like the P above) or BATCHING (it just labels independent copies of the problem). FLOPs = 2 × the product of every dimension involved, counting batch and contracting dims only once.
>
> Why it collapses so cleanly: any tensor contraction is a batched matmul in disguise. Flatten the batch dims into B, the contracting dims into P, whatever's left into N and M, and you're back at B·2NPM. It's still just 2 FLOPs per multiply-add; the shapes only decide how many dot products you owe.
>
> (Attention is exactly this: QKᵀ contracts over head_dim while batch and heads ride along as batch dimensions.)
>
> A fun bonus:
>
> It's been 5 years since my Linear Algebra course, and despite approaching it from Abstract Algebra, I've never found it satisfying. I could never get a good intuition for it and it always felt very invented rather than discovered. The good news is that the Linear Algebra needed to do this stuff is relatively simple and mostly targeted at parallelization, at least from what I've seen so far!
>
> If you're interested in this I recommend Chapter 2 of Lay's Linear Algebra and Its Applications (https://www.amazon.com/dp/032198238X) and Part 1 ("All About Rooflines") of the JAX scaling book, How to Scale Your Model: https://jax-ml.github.io/scaling-book/roofline/

## Reply tweet — why batch dims only count once (draft)

> Why do batch dims only count once in the FLOP rule? Because an output dimension and a batch dimension are the same thing arithmetically but different things memory-wise.
>
> Case 1: [B, P] × [P, M] (einsum bp,pm→bm). B only appears on the activation side, so every one of the B tokens multiplies the *same* [P, M] weight matrix. The weights get reused B times. Free dimensions create reuse of the other operand, and that reuse is exactly why batching raises arithmetic intensity.
>
> Case 2: bp,bpm→bm. Now each token brings its *own* [P, M] matrix. Token b's vector only ever touches token b's matrix. Nothing is shared, nothing is reused, and bytes scale linearly with B. Batch dimensions create no reuse, so FLOPs and bytes grow together and intensity never improves.
>
> Decode attention is exactly case 2: each request's query reads only its own KV cache. The weights of the model are case 1, attention is case 2. My Day 4 throughput model had both terms sitting in the denominator: 16.38 GB of weights (amortized over B) plus 0.147 GB of KV per request (not amortized). Same formula, two kinds of dimensions.

## X post — earlier draft (Claude, single post)

> Day 5/45 of Inference Engineering: a learning day. No GPU today — just the FLOP-counting math that explains every number I've measured this week.
>
> Counting FLOPs is easy: a P-dim dot product is 2P (multiply + add). A matvec is N of those: 2NP. A matmul is M matvecs: 2NPM. Done — that's a transformer forward pass.
>
> The interesting part is the *bytes*. A matvec does 2NP FLOPs but streams NP weights: ~1 FLOP per byte, always, no matter the size. That's batch-1 decode — it's why my MacBook's memory bus predicted its token rate on Day 1.
>
> A matmul is different in kind: compute grows O(N³), data only O(N²). Bigger problem = better FLOPs-per-byte. Matmul is nearly the only op with this property — it's *the* op that can saturate a chip instead of its memory bus, which is a decent one-line explanation of why models are towers of matmuls at all.
>
> For serving, the shape is [batch B] × [weights]: intensity ≈ B FLOPs/byte. Ridge on my M4 ≈ 37, on an A100 ≈ 160. Decode sits at B=1 (bandwidth-bound, Day 1); a 1,000-token prefill sits at ~1,000 (compute-bound — the term my Day 4 model priced at zero and missed 2× for).
>
> Then the part I'd never appreciated, via Strang: there are four ways to compute the same AB — dot products, columns, rows, or a sum of rank-1 (column × row) outer products. Identical FLOPs. Completely different memory traffic. And his block-partition rule (blocks multiply like scalars) is literally what a tiled GPU kernel is: load a block-column and block-row into fast memory, accumulate rank-1 updates into the output tile, recurse down to tensor cores.
>
> The FLOP count is fixed by the shapes. Every trick in kernel engineering is just choosing which mathematically-equivalent ordering moves the fewest bytes. The math says the four forms are interchangeable; the memory hierarchy says they're not even close.
>
> What surprised me: the matmul form nobody teaches first — columns times rows — is the one hardware actually resembles.
>
> Day 6: quantizing the model myself.

## X post — shorter alternate

> Day 5/45: learning day. One table explains my whole week:
>
> dot product: 2P FLOPs / 2P bytes
> matvec: 2NP FLOPs / NP bytes → intensity ~1, forever
> matmul: 2NPM FLOPs / (NP+PM) bytes → intensity grows with batch
>
> Decode is the matvec row (bandwidth-bound — Day 1). Prefill is the matmul row at intensity ~1000 (compute-bound — the term my Day 4 roofline priced at zero, and missed 2×). Compute scales cubically, data quadratically: matmul is the one op that can outrun a memory bus, which is why models are made of it.
>
> Bonus from Strang: there are 4 equivalent ways to compute AB (dots, rows, columns, sum of rank-1 outer products) — same FLOPs, radically different memory traffic. Tiled GPU kernels are his block-multiplication rule applied recursively. The FLOPs were never the interesting number; the loop order is.
>
> Day 6: quantizing the model myself.
