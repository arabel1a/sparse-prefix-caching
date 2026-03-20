# Recurrent State Prefix Caching

**TL;DR:** For LLMs with linear attention (e.g. Mamba2, Qwen3.5), full per-token prefix caching is inefficient: GDN recurrent state is $d_\text{head}^2$ per head vs $d_\text{head}$ for classical GQA KV. One solution is [per-block caching](https://github.com/vllm-project/vllm/pull/26807), which stores GDN states at every cache block boundary (e.g. every 16 tokens), and fully avoids recomputation on cache hit.

This work proposes an intermediate solution — it exchanges _some_ recomputation as a price for sublinear (in length of cached sequence) checkpoint size.

## Background: prefix caching today

Prefix caching avoids redundant computation when several requests share the same prompt prefix. When a new request matches a cached prefix, the corresponding KV blocks are reused, skipping recomputation. For standard transformers this works well: if 900 out of 1000 prefix tokens are cached, you reuse 900 and recompute only 100.

### The hybrid model challenge

While what follows applies to **any state-space or hybrid model**, I illustrate with the [Qwen3.5](https://qwen.ai/blog?id=qwen3.5) model family.

In Qwen3.5, consecutive layers are grouped by 4: 3 **Gated DeltaNet** (GDN) layers followed by 1 full attention layer. For GDN layers, the recurrent state is the result of sequential in-place updates — the state at position $t$ is independent of states at $s<t-1$ given state at time $t-1$. **Caching state for every token** is prohibitive: GDN state is $d_\text{head}^2$ per head per token vs $d_\text{head}$ for attention KV.

However, GDN only needs the state at position $t-1$ to produce position $t$ (unlike full attention, which needs the entire KV history). So if we store **some** states, we can always resume from the closest previous cached checkpoint.

**Current vLLM status** ([tracking issue](https://github.com/vllm-project/vllm/issues/26201), Mar 2026): Mamba1 prefix full caching merged. [GatedDeltaNet support](https://github.com/vllm-project/vllm/pull/26807) is pending review with per-block-boundary caching. Other linear attention models not yet supported.

## Checkpoint placement strategies

All strategies store attention KV cache for the full cached prefix alongside GDN state checkpoints at selected positions. On cache hit, the full model (all layers: GDN, attention, FFN) recomputes from the min(longest stored kvcache, best GDN checkpoint position onward).

**Why KV cache alone cannot skip any compute:** to produce GDN input at layer $i$, we need attention output at layer $i-1$. Attention output requires Q projected from current hidden states (produced by preceding GDN+FFN layers), so no layer's forward pass can be skipped. KV cache for positions before the checkpoint is still necessary so that attention can attend to the full history without re-encoding old tokens into K,V.

| Strategy |  Checkpoint positions | Tail tokens to recompute | GDN checkpoint count | Cache size per group per sequence |
| --- |  --- | --- | --- | --- |
| No cache | — | $L$ | 0 | 0 |
| KV only |  — (no GDN ckpts) | $L$ (equivalent to no cache) | 0 | $2 L n_{kv} d_a$ |
| Block (fixed $B$) |  $B, 2B, \ldots$ | $L \bmod B$ | $\lfloor L/B \rfloor$ | $2 L n_{kv} d_a + 3\lfloor L/B \rfloor n_v d_h^2$ |
| Balanced ($k$ blocks) |  $\lfloor L/k \rfloor, 2\lfloor L/k \rfloor, \ldots$ | $L \bmod \lfloor L/k \rfloor$ | $k$ | $2 L n_{kv} d_a + 3 k \, n_v d_h^2$ |
| Sqrt |  $\sqrt{L}$-spaced | $L \bmod \lfloor\sqrt{L}\rfloor$ | $\lfloor\sqrt{L}\rfloor$ | $2 L n_{kv} d_a + 3\lfloor\sqrt{L}\rfloor n_v d_h^2$ |
| Diadic |  $1, 2, 4, \ldots, 2^{\lfloor\log_2 L\rfloor}$ | worst $L/2$, avg $\sim L/4$ | $\lfloor\log_2 L\rfloor + 1$ | $2 L n_{kv} d_a + 3(\log_2 L) n_v d_h^2$ |

Currently vLLM uses dense $O(L/B)$ checkpoints (linear in sequence length). We propose exchanging some tail recomputation for asymptotically less memory (e.g. $O(\log L)$ vs $O(L/B)$ GDN states), leading to higher cache hit rates under fixed memory budgets.

## Case study: Qwen3.5

![image](assets/qwen_3_5_27b.jpg)
> Qwen3.5 architecture image from [CalvinXKY/InfraTech](https://github.com/CalvinXKY/InfraTech/tree/main/models/qwen3_5)

### Baseline FLOPs

Since prefill is compute-bound, FLOP is a good proxy for latency. **Per token per layer group:**

| Module | Formula | 27B | 0.8B | Comment |
| --- | --- | --- | --- | --- |
| FFN (×4) | $4 \times 3 \times d \times d_\text{ff}$ | 1.07e9 | 4.40e7 | 3 projections per FFN, 4 FFN per group |
| GA Q,O,Gate proj | $3 \times d \times n_q \times d_a$ | 9.44e7 | 6.29e6 | 27B: $n_q=24$, 0.8B: $n_q=8$; $d_a=256$ |
| GA KV proj | $2 \times d \times n_\text{kv} \times d_a$ | 1.05e7 | 1.05e6 | 27B: $n_\text{kv}=4$, 0.8B: $n_\text{kv}=2$ |
| GA quadratic | $n_q \times d_a \times N$ | 6.1e3$N$ | 2.05e3$N$ | Attention cost |
| GDN V,Gate,O proj (×3) | $3 \times 3 \times d \times n_v \times d_h$ | 2.83e8 | 1.89e7 | 27B: $n_v=48$, 0.8B: $n_v=16$; $d_h=128$ |
| GDN Q,K proj (×3) | $3 \times 2 \times d \times n_{qk} \times d_h$ | 6.29e7 | 1.26e7 | $n_{qk}=16$ |
| GDN $\alpha, \beta$ proj (×3) | $3 \times 2 \times d \times n_v$ | 1.47e6 | 9.83e4 | |
| GDN recurrence (×3) | $3 \times n_v \times d_h^2$ | 2.36e6 | 7.86e5 | State shape $d_h \times d_h$ |
| **Total** | | **1.52e9 + 6.1e3$N$** | **8.37e7 + 2.05e3$N$** | |

## Experiments

All experiments run a **single Qwen3.5-0.8B layer group** (3 GDN + 1 attention, 4 layers total) with random weights, using the HuggingFace Transformers implementation. Vocab size is reduced to 512 to fit in memory. Prefill uses chunked processing (4096-token chunks) with SDPA attention backend.

### Single-sequence benchmark (`benchmark_single.py`)

Measures prefill latency for a single sequence across a sweep of lengths (1K–32K tokens). For each length and strategy, GDN checkpoints are captured via a full forward pass, then `prefill_from_checkpoint` is timed over 3 runs. Reports wall time and cache memory per strategy.

> TODO: sweep block size

![empirical_0_8b](assets/benchmark_baselines_0_8b.png)

Theoretical (dashed) lines correspond to number of floating-point operations according to the FLOP table above.

### End-to-end benchmark on ShareGPT traces (`benchmark_e2e.py`)

Replays multi-turn conversations from the [tucnguyen/ShareGPT](https://huggingface.co/datasets/tucnguyen/ShareChat) dataset through the model with a FIFO prefix cache (CPU offloading, configurable budget).

**Data filtering** (`prepare_data.py`):
- Source: `chatgpt_results_final_language_filtered.csv` (pre-filtered for language)
- Tokenized with `Qwen/Qwen3.5-0.8B` tokenizer, formatted as `<|role|> text`
- Conversations truncated to first 36 messages (95th percentile; raw data has up to 800 rounds per conversation, which would create a quadratic blowup in total tokens since each turn re-prefills the full history)
- Messages dropped where cumulative token count exceeds 16384
- Tokenization done in chunks of 2048 conversations, converting to `np.int32` immediately to reduce memory (28 bytes per Python int vs 4 bytes)

**Request simulation:**
- Each user message generates a prefill request containing all preceding messages (simulating multi-turn serving)
- Conversations are interleaved with Poisson arrival times (exponential inter-arrival) to simulate concurrent load. Without interleaving, the crowdsourced timestamps almost never overlap, leading to trivial 100% cache hits
- Turn order within each conversation is preserved

![e2e_speedup](assets/e2e_time_vs_length.png)

> Left panel — Prefill time vs context length. Right panel — per-request relative speedup.
