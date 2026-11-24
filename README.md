# RocketKV-cpp

RocketKV-cpp runs [RocketKV](https://arxiv.org/abs/2502.14051), a training-free two-stage KV-cache compression method, inside a pinned and patched `llama.cpp` on Apple silicon. Both stages run on the GPU as custom Metal kernels:

- **Stage 1** (once, after prefill) scores every prompt token with the queries of the last observation window and permanently evicts the unimportant ones from the KV cache.
- **Stage 2** (every decode step, every layer) is hybrid sparse attention (HSA): it estimates chunk importance in a reduced space, selects the top-$k$ tokens per KV group, and computes exact attention over the selection only.

With Llama-3.2-3B-Instruct on an Apple M5 Ultra and a 1024-token budget, Stage 1+2 decodes at 7.1–7.5 ms/token from 2K to 128K context while Full KV grows to 21.9 ms/token (2.91× at 128K). The full results are in [`report/report.pdf`](report/report.pdf).

[Architecture](#architecture) · [Request lifecycle](#request-lifecycle) · [Implementation](#implementation) · [Repository layout](#repository-layout) · [Build and run](#build-and-run) · [Results and resume](#read-and-resume-a-run) · [Development](#development)

## Architecture

```text
┌───────────────────────────── Python (eval_suite/, scripts/eval/*.sh) ─────────────────────────────┐
│ stage2.py (LongBench/RULER accuracy)   latency.py (timing)   report.py (tables)   metrics.py       │
│        │  one runner process per sample: prompt in a JSON file (--batch), metrics out (--output)   │
└────────┼───────────────────────────────────────────────────────────────────────────────────────────┘
         ▼
┌──────────────────────────────────── build/runner (src/runner.cpp) ────────────────────────────────┐
│ run_inference(): prefill → Stage 1 eviction → Stage 2 init → decode loop → metrics                 │
│                                                                                                    │
│  Stage 1                               Stage 2                               CPU reference         │
│  eval_callback() captures Q/K   ──▶    RocketAdapter                         src/rocket.cpp        │
│  RocketMetal::capture_prefill()        (src/rocket_adapter.cpp)              compute_prefill_      │
│  RocketMetal::finish_prefill()         builds a custom GGML op per layer,    eviction(),           │
│  select_prefill_tokens() (CPU)         maps positions → KV slots             compute_hybrid_       │
│  llama_memory_seq_rm() (CPU)                  │                              sparse_decode()       │
│                                               ▼                                                    │
│                        RocketMetal (src/metal/rocket_metal.mm, rocket_metal.metal)                 │
│                        prefill_* kernels (Stage 1), score + attend kernels (Stage 2)               │
└────────────────────────────────────────────────┼───────────────────────────────────────────────────┘
                                                 ▼
┌──────────────── llama.cpp @ patches/llama.cpp-revision + 3 patches (patches/*.patch) ───────────────┐
│ llama-stage2.patch      decode attention hook (llama_set_rocket_attention), KV position readout     │
│ llama-metal-buffer.patch  MTLBuffer + offset behind a tensor → kernels read the KV cache in place   │
│ llama-metal-custom.patch  custom op encodes into GGML's own Metal encoder (same command buffer)      │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Layer | Code | Responsibility |
|---|---|---|
| Experiment drivers | `eval_suite/stage2.py`, `eval_suite/latency.py`, `scripts/eval/*.sh`, `configs/*.json` | Build prompts, run the runner once per sample, score, resume, write `manifest.json` / `summary.json` |
| Runner | `src/runner.cpp` (`run_inference`, `run_main`) | Load the model, run prefill and decode, call Stage 1 and Stage 2, measure time and memory |
| Algorithm (CPU reference) | `include/rocket.h`, `src/rocket.cpp` | Budget derivation, Stage 1 scoring and selection, Stage 2 HSA; also the test oracle |
| Stage 2 integration | `include/rocket_adapter.h`, `src/rocket_adapter.cpp` | Replace each layer's decode attention with a custom GGML op and keep the candidate/slot map |
| Metal backend | `include/rocket_metal.h`, `src/metal/rocket_metal.mm`, `src/metal/rocket_metal.metal` | Buffers, pipelines and dispatch (host side); Stage 1 and Stage 2 kernels (GPU side) |
| llama.cpp patches | `patches/llama-stage2.patch`, `patches/llama-metal-buffer.patch`, `patches/llama-metal-custom.patch` | The minimal hooks the adapter and Metal backend need |

### Modes and backends

| `--rocket-mode` | Stage 1 eviction | Decode attention | Report ID |
|---|---|---|---|
| `off` | — | native `llama.cpp` | E0 Full KV |
| `stage1` | yes (`--stage1-backend cpu\|metal`) | native `llama.cpp` over the retained tokens | E1 (CPU) / E2 (Metal) |
| `hsa` | yes | HSA (`--stage2-backend cpu-reference\|metal`) | E3 (CPU) / E4 (Metal, main method) |
| `stage2` | — (whole prompt kept) | HSA over the whole prompt | E5 ablation |

The model's own layers always run on Metal (`--gpu-layers 99`); the backend flags only choose where RocketKV's own computation runs. HSA requires an F16 KV cache; weights can be quantized (Q8_0 in all experiments).

## Request lifecycle

What happens for one sample in `hsa` mode with Metal backends (E4), in order:

| # | Phase | Device | What happens | Code |
|---|---|---|---|---|
| 1 | Budget | CPU | Derive the Stage 1 prompt budget, Stage 2 top-$k$, chunk size and $r$ from the token budget and prompt length | `compute_rocket_params` (`src/rocket.cpp`) |
| 2 | Prefill | Metal | `llama_decode` over the prompt with native attention | `run_inference` (`src/runner.cpp`) |
| 3 | Q/K capture | Metal | GGML's eval callback sees each post-RoPE `Qcur-*`/`Kcur-*` of the last layer; the Metal backend copies them GPU-to-GPU (0 host bytes) | `eval_callback` → `RocketMetal::capture_prefill`; kernel `prefill_capture` |
| 4 | Stage 1 scoring | Metal | Window scores $QK^\top$, softmax, GQA reduce, max-pool (kernel 63) | `RocketMetal::finish_prefill` → `score_prefill`; kernels `prefill_pack/dot/softmax/reduce/pool` |
| 5 | Stage 1 selection | CPU | Top-$k$ over the importance vector, shared by all layers; the observation window is always kept | `select_prefill_tokens` (`src/rocket.cpp`) |
| 6 | Eviction | CPU | Delete the evicted ranges from the KV cache | `llama_memory_seq_rm` in `run_inference` |
| 7 | Stage 2 init | CPU | Store the retained positions as candidates and install the attention hook | `RocketAdapter::initialize` → `llama_set_rocket_attention` |
| 8 | Decode step | CPU | Append the new position to the candidates | `RocketAdapter::begin_step` |
| 9 | Graph build | CPU | For every layer, `llama.cpp` calls the hook, which returns a custom op instead of native attention | `RocketAdapter::build_attention` (`ggml_custom_4d` + `RocketMetal::attach`) |
| 10 | Slot map | CPU | Once per step, map candidate positions to live KV-cache slots | `RocketAdapter::refresh_slots` (`llama_rocket_kv_positions`) |
| 11 | HSA scoring + selection | Metal | Top-$r$ query dims, incremental chunk min/max, approximate scores, per-head softmax, GQA sum, token-weighted top-$k$ | `RocketMetal::encode` → `run`; kernel `score` |
| 12 | Exact attention | Metal | Split-K online softmax over the selected tokens; heads of a GQA group share one K/V read | kernel `attend` |
| 13 | Rest of the layer | Metal | Output projection, MLP, logits, in the same command buffer as steps 11–12 | `llama.cpp` (unchanged) |
| 14 | Checks | CPU | Every layer ran exactly once; Metal error word is clear | `RocketAdapter::finish_step`, `RocketMetal::check_errors` |

`stage1` mode stops after step 6 and decodes natively. `stage2` mode skips steps 3–6 and keeps the whole prompt as candidates. `off` runs only steps 2, 8 and 13.

## Implementation

### Budget derivation

`compute_rocket_params` follows the RocketKV paper. With prompt length $N$ and token budget $B$, the initial compression ratio is $\mathrm{CR}=N/B$ and $\alpha=\min(0.2+0.06\log_2\mathrm{CR},\,0.8)$. The total capacity is $\lfloor N/\mathrm{CR}^{\alpha}\rfloor$; Stage 1 keeps that capacity minus the expected output length, and Stage 2 uses $k=B/2$, chunk size $\min(\lfloor \mathrm{CR}_2\rfloor, \lceil\sqrt{\mathrm{CR}_2}\rceil)$ and $r=\mathrm{round}(d\cdot \text{chunk}/\mathrm{CR}_2)$, where $\mathrm{CR}_2$ = capacity / $B$. In `stage2` mode the capacity is the whole prompt. At 32K and $B=1024$, Stage 1 keeps about 5.7K tokens and Stage 2 uses chunk 3, $r=68$.

### Stage 1: prefill eviction

- **Capture without host copies.** `run_inference` registers `eval_callback` for the prefill `llama_decode`. For each post-RoPE Q/K tensor of the target layer the callback calls `RocketMetal::capture_prefill`, which binds the tensor's own `MTLBuffer` (via `llama-metal-buffer.patch`) and copies the observation-window queries and all keys into RocketKV's buffers with `prefill_capture`. The runner records `stage1_capture_host_bytes`, which is 0 for Metal runs.
- **Scoring on Metal.** `RocketMetal::score_prefill` runs five kernels: `prefill_pack` (layout), `prefill_dot` (window queries × all keys), `prefill_softmax` (per query row), `prefill_reduce` (sum over window queries and the query heads of each GQA group) and `prefill_pool` (1D max-pool). Only the final importance vector (one float per token) is read back.
- **Selection and deletion on the CPU.** `select_prefill_tokens` keeps the top tokens plus the observation window; `run_inference` deletes contiguous evicted ranges with `llama_memory_seq_rm`. This is a one-time cost of 0.5–8 ms per request.
- **CPU path.** With `--stage1-backend cpu` the callback copies Q/K to host memory and `compute_prefill_eviction` (`src/rocket.cpp`) does the same computation.

### Stage 2: hybrid sparse attention

- **Hook into `llama.cpp`.** `llama-stage2.patch` adds `llama_set_rocket_attention(ctx, builder, user_data)`. While a builder is set, the graph builder calls it for every layer of a single-token decode instead of building native attention, and passes Q, the KV-cache views, the mask and the scale. It also adds `llama_rocket_kv_positions`, which reports the position stored in every KV-cache slot.
- **Custom op per layer.** `RocketAdapter::build_attention` validates shapes and types (F32 Q, F16 K/V, scale $1/\sqrt{d}$) and returns a `ggml_custom_4d` op. On Metal it calls `RocketMetal::attach`, which (through `llama-metal-custom.patch`) makes GGML call `RocketAdapter::encode` with GGML's own compute encoder, and replaces the op's K/V sources with 1-element views so the scheduler keeps the dependency without copying the KV cache to the CPU.
- **Candidates and slots.** Candidates are the retained prompt positions plus every generated position (`begin_step` appends one per step). `refresh_slots` reads the position of every KV slot once per step and builds the candidate → slot map that the kernels use to gather K/V in place.
- **Two kernels per layer.** `RocketMetal::run` encodes, per layer:
  - `score`: a wide grid where each threadgroup owns a range of chunks of one KV head. It recomputes the group's top-$r$ query dimensions, refreshes the min/max summaries of chunks that received new tokens (summaries persist across steps, so only new tokens cost work), and writes approximate scores. The last threadgroup to finish (atomic ticket) runs per-head softmax, the GQA sum and the top-$k$: an 8-bit radix select with simdgroup prefix sums that counts each chunk by its number of tokens and emits chronological token IDs.
  - `attend`: split-K exact attention with online softmax over the selected tokens. The query heads of a GQA group share every K/V read; simdgroups merge in threadgroup memory and the last block merges the partials.
  Cross-threadgroup data uses `coherent(device)` buffers with acquire/release fences. Because both kernels are encoded into GGML's encoder, Stage 2 runs inside the same command buffer as the rest of the layer, with no extra synchronization.
- **CPU path.** With `--stage2-backend cpu-reference`, `RocketAdapter::compute_layer` gathers Q/K/V on the host and calls `compute_hybrid_sparse_decode` (`src/rocket.cpp`). It is 33–115× slower and exists as the reference for tests and for the E3 baseline.
- **Optimization history.** Stage 2 started as 9 kernels per layer with a barrier after each (5.30 ms for 28 layers at 8K) and ended at 2 kernels (2.48 ms). Every kept and rejected change is in [`HSA_OPTIMIZATION_LOG.md`](HSA_OPTIMIZATION_LOG.md).

### llama.cpp patches

| Patch | Adds | Used by |
|---|---|---|
| `llama-stage2.patch` | `include/llama-rocket.h`: `llama_set_rocket_attention`, `llama_rocket_kv_positions`, `llama_rocket_set_f32_reference`; the decode-attention hook in the graph builder | `RocketAdapter` |
| `llama-metal-buffer.patch` | `ggml_backend_metal_get_tensor_buffer`: the `MTLBuffer` and offset behind a GGML tensor | `RocketMetal::capture_prefill`, `can_bind`, `run` (zero-copy KV access) |
| `llama-metal-custom.patch` | `ggml_backend_metal_set_custom` + `ggml_metal_encoder_get_obj`: a custom op encodes its own kernels into GGML's Metal encoder | `RocketMetal::attach`, `RocketAdapter::encode` |

`scripts/stage2.sh build` checks out the revision in `patches/llama.cpp-revision` into `llama.cpp/`, applies the three patches in order, builds `llama`, then `make runner`.

### Correctness

- `tests/test_rocket.cpp`: the CPU reference (budget derivation, Stage 1 selection, HSA output against a dense-attention oracle).
- `tests/test_metal.cpp`: Metal kernels against the CPU reference: selection recall (≥ 0.98 on average) and attention output against fp64 attention over the tokens Metal selected; `build/test_metal bench` times the 28-layer HSA chain.
- `tests/test_stage2_integration.py`: end-to-end runs of the runner in every mode.
- At run time the runner checks that every layer executed HSA exactly once per step (`stage2_layer_calls == model_layers × decode_forward_calls`), that Metal Stage 1 copied 0 host bytes, and that every output is finite.
- On 60 RULER samples each at 2K/4K/8K, the CPU and Metal versions of Stage 1 and of Stage 1+2 produce identical generations (`results/a106/bench/ruler_cpu/`).

## Repository layout

```text
RocketKV-cpp/
├── include/                  rocket.h (algorithm), rocket_adapter.h (llama.cpp integration), rocket_metal.h (Metal backend)
├── src/
│   ├── runner.cpp            Model loading, prefill → Stage 1 → Stage 2 → decode loop, metrics (run_inference)
│   ├── rocket.cpp            CPU reference: budget derivation, Stage 1 eviction, Stage 2 HSA
│   ├── rocket_adapter.cpp    Custom GGML attention op per layer; candidate positions → KV slots
│   └── metal/                rocket_metal.metal (kernels), rocket_metal.mm (buffers and dispatch), non-Metal stub
├── patches/                  Pinned llama.cpp revision and the three patches
├── scripts/                  Shell entry points only
│   ├── stage2.sh             Build, test, and demo-run the runner
│   ├── data/                 Download the model and LongBench; generate official RULER inputs
│   ├── eval/                 One script per experiment (ruler.sh, longbench.sh, latency*.sh, stage2_only.sh, benchmark.sh)
│   └── bench/                Timing benchmarks
├── configs/                  Model paths, task lists, sample counts, lengths, budgets
├── eval_suite/               Dataset loaders, prompt protocol, scoring, and the drivers the scripts call
├── tests/                    CPU/Metal kernel tests and model integration tests
├── results/a106/bench/       Raw results of the reported experiments (one directory per experiment)
├── report/                   Experiment report (report.tex, report.pdf) and figure scripts (report/figures/)
├── eval.py                   Legacy LongBench/RULER evaluator (no Stage 2 HSA)
├── HSA_OPTIMIZATION_LOG.md   Every Stage 2 kernel experiment with A/B numbers and the decision
├── Makefile                  Incremental runner and test builds
└── environment.yml           Python evaluation environment
```

Local inputs live outside the tracked source: `llama.cpp/` is the patched upstream checkout, `build/` holds the compiled runner, and `models/` and `data/` hold benchmark inputs.

## Build and run

From the repository root on a Metal-capable Mac (Git, CMake, Make, Python 3, a C++17 compiler):

```bash
conda env create -f environment.yml    # First setup only
conda activate rocketkv
bash scripts/stage2.sh build
bash scripts/data/download_model.sh     # Q8_0 GGUF weights + HF tokenizer in models/
bash scripts/data/download_longbench.sh # official JSONL files in data/longbench/
bash scripts/data/prepare_ruler.sh      # official RULER inputs in data/ruler/<len>/<task>/
```

`prepare_ruler.sh` runs the NVIDIA RULER generator vendored in [NVlabs/RocketKV](https://github.com/NVlabs/RocketKV) (`eval/ruler_utils`), cloned to `../RocketKV` unless `ROCKETKV_DIR` points elsewhere.

A single request (`--batch` takes a JSON array of prompts; `out.json` holds the generations and timings):

```bash
echo '[{"prompt": "Summarize: ...", "max_tokens": 128}]' > in.json
build/runner --model models/Llama-3.2-3B-Instruct-Q8_0.gguf --batch in.json --output out.json \
    --rocket-mode hsa --stage1-backend metal --stage2-backend metal --budget 1024 \
    --kv-type f16 --gpu-layers 99
```

The experiments in the report (results in `results/a106/bench/<directory>`):

| Experiment | Command | Result directory | Inputs |
|---|---|---|---|
| RULER | `bash scripts/eval/ruler.sh OUT/ruler` | `ruler/` | `niah_single_2`, `niah_multikey_2`, `vt`, `fwe`, `qa_1`; 10 samples each at 2K–32K tokens |
| LongBench | `bash scripts/eval/longbench.sh OUT/longbench --budget 1024 --num-samples 20` | `longbench/` | 16 English tasks × 20 samples, middle-truncated to 32K tokens |
| CPU/Metal consistency | see `scripts/eval/benchmark.sh` | `ruler_cpu/` | E1, E3 on the first 4 samples per task at 2K–8K |
| Budget sweep (accuracy) | see `scripts/eval/benchmark.sh` | `ruler_budget/` | E4 at budgets 256 and 4096 tokens, 16K/32K |
| Latency | `bash scripts/eval/latency.sh OUT/latency` | `latency/` | E0–E4, RULER prompts at 2K–32K tokens, 128 output tokens (`--ignore-eos`), interleaved repeats |
| Long decode | `bash scripts/eval/latency_long.sh OUT/latency_long` | `latency_long/` | E0, E2, E4; 1024 output tokens at 8K and 32K |
| 64K/128K | `bash scripts/eval/latency_128k.sh OUT`; `bash scripts/eval/ruler.sh OUT/ruler_128k --config configs/ruler_128k.json` | `latency_128k/`, `latency_long_128k/`, `ruler_128k/` | Latency and 1024-token decode at 64K/128K; RULER 5 tasks × 3 samples |
| Stage 2 ablation | `bash scripts/eval/stage2_only.sh OUT` | `ruler_stage2/`, `ruler_128k_stage2/`, `latency_stage2/` | E5 (`--rocket-mode stage2`): RULER 2K–128K and latency against E4 |
| Budget sweep (latency) | `python -m eval_suite.latency --config configs/latency_budget.json --output OUT/latency_budget` (and `latency_budget_128k.json`) | `latency_budget/`, `latency_budget_128k/` | E0 and E4 at budgets 256–4096 tokens, 32K and 128K |
| All | `bash scripts/eval/benchmark.sh OUT` | — | Everything at the scale of the report, then `report.md` (`scripts/eval/report.sh`) |

The accuracy commands run full KV once, then `stage1` and `hsa` for every budget in the config (512, 1024, 2048 tokens), with Metal Stage 1 scoring and Metal HSA. After the output directory, add `--mode stage1 hsa` (or `stage2`), `--budget 1024`, `--stage1-backend cpu`, `--stage2-backend cpu-reference`, `--num-samples N`, `--context-lens ...`, or `--dry-run` (load and format every input, print the plan, run nothing).

Prompts and scores follow the RocketKV paper code. LongBench prompts are middle-truncated with the HF tokenizer and wrapped in the llama3 chat template, except `trec`, `triviaqa`, `samsum`, `lcc`, and `repobench-p`; scoring is LongBench's `scorer()`. RULER inputs are used as generated (llama3 template and answer prefix included); every task generates up to 128 tokens and is scored with RULER's `string_match_all` (`string_match_part` for `qa_*`). As in RocketKV, chat-formatted prompts start with two BOS tokens. Differences from the paper are recorded in each config's `_comment`.

## Read and resume a run

- `status.json` shows progress; `summary.json` reports scores (accuracy in %) and timings (ms) per run (`off`, `stage1_b1024`, ...) and task (`<len>/<task>` for RULER).
- Each sample has a scored checkpoint and runner log. Wall time includes model startup; `ttft_ms`, `prefill_ms`, `evict_ms`, `stage2_init_ms`, `first_decode_ms` and `decode_step_ms` (one entry per decode forward, ms) measure inference. `kv_cache_mb` is the logical KV size in MB; `peak_rss_mb` is measured.
- Re-run the same command to resume. The driver checks `manifest.json` against the inputs, settings, binary, driver hash, and Git commit. Use a new directory when these change.

## Development

| Entry point | Purpose |
|---|---|
| `make runner` | Rebuild `build/runner` after C++ changes. |
| `ROCKET_TEST_METAL=1 bash scripts/stage2.sh test` | Run CPU and Metal checks, integration tests, and UBSan. |
| `bash scripts/stage2.sh run MODEL.gguf` | Compare `off`, `stage1`, and `hsa` on a demo batch (CPU by default). |
| `bash scripts/bench/stage1.sh` | Measure Stage 1 capture and scoring separately from task scoring. |
| `python3 eval.py configs/longbench.json` | Run the legacy evaluator; its `rocket-kv` mode uses Stage 1 only. |

Further reading: [RocketKV paper](https://arxiv.org/abs/2502.14051) · [Experiment report](report/report.pdf) · [HSA optimization log](HSA_OPTIMIZATION_LOG.md).
