# CudaForge StreamMA Protocol Plan

This workspace is for reproducing the communication-layer idea from
Streaming Communication in Multi-Agent Reasoning on top of CudaForge.

Hard boundary: change only the LLM communication layer. The existing
compile/test/Nsight Compute harness must remain behaviorally unchanged.

## Goal

Replicate StreamMA-style streaming as semantic-frame communication between
agents, then compare it with CudaForge serial baselines on the same task set,
same compile/test harness, same NCU profiling path, and same scoring logic.

The protocol must not stream partial CUDA code. Streaming units are semantic
frames. CUDA/Python code is produced only at the final Coder step.

## Experimental Arms

P0 `serial_full`
- CudaForge original behavior.
- Full prompt -> full answer.
- No semantic frame layer.

P1 `serial_segmented`
- Planner A generates all semantic steps A1..A4 first.
- Judge B waits until A is fully done, then produces B1..B4.
- Coder C waits until B is fully done, then generates C1..C4 and FinalCode.
- No interleaved streaming.

P2 `stream_nl`
- StreamMA-spirit natural-language semantic frames.
- A emits one semantic frame at a time.
- B validates/revises each frame as it arrives.
- C maintains an accepted-frame context but still emits code only in FinalCode.

P3 `stream_json_gate`
- A emits JSON semantic frames.
- B validates each frame against schema and semantic constraints.
- Rejected frames are not visible to C except as rejection metadata.
- C consumes accepted frames only and produces exactly one FinalCode payload.

## Semantic Frame Order

The communication unit order is fixed:

1. `ContractFrame`
   - Input/output contract, dtype, shape, tolerance, API preservation.
2. `ScheduleFrame`
   - Operator replacement/fusion schedule and launch-level intent.
3. `MemoryFrame` or `ReductionFrame`
   - Memory layout/access/coalescing/shared-memory plan, or reduction axes,
     associativity, block strategy, numerical strategy.
4. `OptimizationHypothesisFrame`
   - Metrics-driven hypotheses, expected effect, validation signal, risk.
5. `FinalCode`
   - Complete `ModelNew` Python source only.

No `ContractFrame`, `ScheduleFrame`, `MemoryFrame`, `ReductionFrame`, or
`OptimizationHypothesisFrame` may contain partial CUDA source.

## CudaForge Insertion Points

CudaForge currently has three natural LLM decision points:

1. Round 0 seed path
   - `build_seed_prompt -> _llm_to_kernel -> compile/test`
2. Repair round
   - Judge correctness `problem_identify -> repair_prompt -> _llm_to_kernel`
3. Optimization round
   - `NCU metrics -> Judge optimization -> build_optimization_prompt -> _llm_to_kernel`

The StreamMA implementation should wrap prompt construction and LLM calls around
these three points, then pass one complete final code block to the existing
`_llm_to_kernel` saving/evaluation path.

## Planned Code Shape

Keep the change narrow:

- Add a protocol selector such as `--comm_protocol`.
- Add a communication adapter module, for example `streamma/protocol.py`.
- Add schema validation utilities for P3, for example `streamma/schema_gate.py`.
- Route only LLM prompts/replies through the selected protocol.
- Do not alter `compare_and_bench`, `_bench_and_score`, `profile_bench`,
  `load_ncu_metrics`, or score aggregation.

The experiment-side gate prototype is:

```bash
/data/workspace/airulan/conda_envs/kernelbench_py310_cu124/bin/python \
  experiments/streamma/validate_frame.py semantic \
  experiments/streamma/examples/contract_frame.example.json
```

P3 should reuse the same logic in the runtime protocol adapter: JSON Schema
first, then non-final code-leak rejection, then Judge B semantic acceptance.

Expected call boundary:

```text
existing prompt builder
  -> protocol adapter
  -> agent calls A/B/C
  -> complete FinalCode text
  -> existing _llm_to_kernel code extraction/save
  -> existing compile/test/NCU harness
```

## Artifact Requirements

Each run should write:

- Original base prompt.
- Per-agent prompt and raw reply.
- Parsed semantic frames.
- P3 schema validation result for every frame.
- Accepted/rejected frame log.
- FinalCode raw response.
- Existing CudaForge metrics and usage CSV.

The artifact path should stay under the task `evaluation/llm_io` directory so
the existing run layout remains easy to compare.

## Model/API Policy

Do not store API keys in git. Use environment variables only.

Allowed base URLs for cloud API runs are documented in `API_AND_ENV.md`.
Local model candidates under `/data1/hf_models` include coder-oriented models
such as `DeepSeek-Coder-V2-Lite-Instruct`, `Qwen2.5-Coder-7B-Instruct`, and
`Qwen3-Coder-30B-A3B-Instruct`.

## Initial Workspace State

The GitHub pack clone was unstable, so this workspace was created from a
`main` tarball fetched through `ghfast.top`, then initialized as a local git
repository. The remote metadata is still present:

- `origin`: `https://github.com/arlisgreat/CudaForge.git`
- `ghfast`: `https://ghfast.top/https://github.com/arlisgreat/CudaForge.git`

When GitHub connectivity is stable, replace or reconcile this tarball import
with a real clone before pushing long-lived branches.
