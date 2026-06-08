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
- This is the unmodified CudaForge one-agent baseline. It is not claimed to be
  the paper's `Single`; it is the project control for the original system.

P1 `serial_segmented`
- Causal control for segmented generation, multiple LLM calls, JSON schema,
  gate/self-conditioning, and semantic-frame prompting.
- Use the same agent roles, step count, frame order, output format, schema gate
  mode, retry budget, and Coder final-code prompt as the matching stream arm.
- Planner A generates all semantic steps A1..A4 first.
- Judge B starts only after A4 is complete, then produces/validates B1..B4.
- Coder C starts only after B4 is complete, then generates C1..C4 and FinalCode.
- No interleaved message passing is allowed.

P2 `stream_nl`
- StreamMA-spirit natural-language semantic frames.
- A emits one semantic frame at a time.
- B validates/revises each frame as it arrives.
- C maintains an accepted-frame context but still emits code only in FinalCode.
- This arm tests natural-language step streaming and should be reported as the
  paper-spirit version, not the CUDA-specific main method.

P3 `stream_json_gate`
- A emits JSON semantic frames.
- B validates each frame against schema and semantic constraints.
- Rejected frames are not visible to C except as rejection metadata.
- C consumes accepted frames only and produces exactly one FinalCode payload.
- This is the main CUDA-oriented method.

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

## Paper-Faithfulness Rules

The original StreamMA protocol compares `Single`, `Serial`, and `Stream`.
According to arXiv:2606.05158, Serial waits for each upstream agent's complete
response before calling the next agent, while Stream runs agents concurrently,
pushes each completed reasoning step downstream immediately, and calls
downstream agents step-by-step with prior steps in context.

For this CudaForge adaptation:

- `paper_strict_stream=true` requires actual overlapping execution:
  `B_s` or `C_s` must start before the upstream agent has completed all later
  steps. Artifact timestamps must prove this.
- If the implementation uses one independent LLM call per semantic frame
  because the provider cannot yield step boundaries from a live stream, label it
  `stream_emulated`, not paper-strict StreamMA.
- P1 is mandatory whenever P2/P3 use multiple calls. Otherwise any observed
  gain could come from segmented generation, JSON constraints, repeated
  self-conditioning, or retry/gating rather than streaming communication.
- For P3 vs P1, P1 must also use the same JSON schema/gate/retry rules but with
  strict serial barriers. The only causal difference should be message timing.
- For P2, steps are natural language. For P3, steps are JSON semantic frames
  with hard schema validation. Neither may stream partial CUDA code.

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
- Per-agent/per-step `call_start_ts`, `first_token_ts` if available,
  `step_complete_ts`, and `call_end_ts`.
- Protocol metadata: `paper_strict_stream`, `stream_emulated`,
  `call_count_by_agent`, `schema_gate_enabled`, and `barrier_policy`.

The artifact path should stay under the task `evaluation/llm_io` directory so
the existing run layout remains easy to compare.

## Remote Maintenance

After each clean stage:

```bash
git status --short
scripts/streamma_checkpoint.sh <stage-name>
scripts/streamma_sync.sh
```

The sync script pushes the current branch and tags. It does not store
credentials; use an existing git credential helper or a temporary `GIT_ASKPASS`
wrapper outside the repository when HTTPS auth is needed.

## Phase Matrix

Use the matrix script to start controlled runs:

```bash
export OPENAI_API_KEY="<set in shell>"
export OPENAI_BASE_URL="https://aigc.x-see.cn/v1"
experiments/streamma/run_matrix.sh KernelBench/level1/19_ReLU.py
```

For a full Level1 run, pass the directory and force the task picker to take all
100 sorted tasks. `main.py` otherwise defaults to sampling one task from a
directory:

```bash
export OPENAI_API_KEY="<set in shell>"
export OPENAI_BASE_URL="https://open.xiaojingai.com/v1"
export PATH="/usr/local/cuda-12.4/bin:/data/workspace/airulan/conda_envs/kernelbench_py310_cu124/bin:$PATH"
PHASES=phaseA FIRST_N=100 NUM_TASKS=0 OUT_ROOT=run/streamma_level1_phaseA \
  experiments/streamma/run_matrix.sh KernelBench/level1
```

By default the matrix runs the four requested arms: P0, P1, P2, and P3. Set
`INCLUDE_P1_NL=1` only when you want the extra P2-matched serial natural-language
control.

If an API connection drops during a long matrix, the OpenAI-compatible client
retries transient connection/timeout/rate-limit/server errors. Defaults:
`LLM_API_MAX_ATTEMPTS=5`, `LLM_API_RETRY_BASE_SECONDS=2`,
`OPENAI_TIMEOUT_SECONDS=120`, and `OPENAI_CLIENT_MAX_RETRIES=2`.

For manual resume, use `SKIP_FIRST` with the same sorted task picker. Example:

```bash
PHASES=phaseA FIRST_N=100 SKIP_FIRST=12 OUT_ROOT=run/streamma_level1_phaseA_resume \
  experiments/streamma/run_matrix.sh KernelBench/level1
```

Directory runs save `summary.json` after every completed task, so partial
progress remains summarizable after an interrupted run.

Run Phase B separately so seed streaming and optimization streaming remain
distinct:

```bash
PHASES=phaseB FIRST_N=100 NUM_TASKS=0 OUT_ROOT=run/streamma_level1_phaseB \
  experiments/streamma/run_matrix.sh KernelBench/level1
```

Summarize after each matrix:

```bash
/data/workspace/airulan/conda_envs/kernelbench_py310_cu124/bin/python \
  experiments/streamma/summarize_results.py run/streamma_level1_phaseA
```

Phase A uses `--round 1` and applies the protocol only to seed generation.
The primary outcome is compile/correctness failure reduction from early semantic
frames.

Phase B uses `--round 2` and applies the protocol only to optimization
generation. Round 0 remains the original seed path; round 1 reaches the
optimization protocol only if the seed kernel is runnable and NCU succeeds. The
primary outcome is speedup improvement from streamed bottleneck/optimization
decisions.

Gate settings:

- `balanced` is the default for P3 and JSON P1 controls. It enforces JSON
  Schema, rejects code leakage in non-final frames, and requires Judge B
  acceptance.
- `strict` is for stress testing gate sensitivity; do not use it as the default
  headline setting.
- `off` is only for natural-language P1/P2 controls where JSON schema is not
  part of the arm.
- Report gate rejection rate. A gate that accepts everything is too weak; a gate
  that prevents most rounds from reaching Coder is too strong.

Do not use end-to-end wall time as the main claim. Every task summary records:

- `llm_wall_time`
- `llm_api_time_sum`
- `protocol_wall_time`
- `compile_test_wall_time`
- `ncu_profile_wall_time`
- `total_wall_time`

Compile/test and NCU can dominate total runtime, so communication claims must
use the decomposed timing fields.

Performance follows CudaForge/KernelBench: the score is the generated kernel
execution speed relative to the PyTorch reference on the same GPU. For iterative
methods, report each task's fastest correct candidate across all generated
rounds.

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
