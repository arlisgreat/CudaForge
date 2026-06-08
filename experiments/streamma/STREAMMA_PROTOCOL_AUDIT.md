# StreamMA Protocol Audit

This note records what must be true before we call a CudaForge run a faithful
StreamMA reproduction.

Source checked: [arXiv:2606.05158](https://arxiv.org/abs/2606.05158),
submitted 2026-06-03.

## Original Protocol Facts

The paper compares three modes:

- `Single`: a single agent receives only the query.
- `Serial`: each agent waits for its predecessor to finish and receives the
  predecessor's complete output.
- `Stream`: agents run concurrently; each completed reasoning step is sent to
  direct successors immediately.

The paper's chain analysis uses `A` agents, each producing `S` reasoning steps.
For adjacent agents, the downstream step condition is:

- Single: own previous steps only.
- Stream: own previous steps plus upstream prefix through step `s`.
- Serial: own previous steps plus all upstream steps `1:S`.

The method section also states that downstream agents in Stream are called `S`
times, while Serial makes a blocking full-response call per agent. Therefore,
if a reproduction compares Stream multi-call against Serial one-call only, the
result confounds streaming with segmentation, self-conditioning, and call-count
effects.

## CudaForge Mapping

CudaForge has three LLM decision boundaries:

1. seed: `build_seed_prompt -> _llm_to_kernel -> compile/test`
2. repair: correctness judge -> repair prompt -> `_llm_to_kernel`
3. optimization: NCU metrics -> optimization judge -> optimization prompt ->
   `_llm_to_kernel`

The communication protocol must wrap these boundaries and still hand exactly one
complete FinalCode response to the existing kernel save/evaluation path.

For the CUDA adaptation, we use a three-agent chain:

- Planner A: semantic analysis frames.
- Judge B: validates, rejects, or revises frames.
- Coder C: consumes accepted frames and writes FinalCode.

The fixed step count is `S=4` semantic steps plus one final code step:

1. `ContractFrame`
2. `ScheduleFrame`
3. `MemoryFrame` or `ReductionFrame`
4. `OptimizationHypothesisFrame`
5. `FinalCode`

Only step 5 may contain CUDA/Python source.

## Required Four Arms

P0 `serial_full`
- Unmodified CudaForge behavior.
- One full prompt and one full final response per original LLM decision point.
- Tracks the original system, not StreamMA's multi-agent Serial.

P1 `serial_segmented_control`
- Same semantic steps, agent roles, schema mode, retry budget, and call count as
  the stream arm under comparison.
- Strict barrier after every upstream agent: A finishes all steps before B
  starts; B finishes all steps before C starts.
- This is the required control for distributed output, multiple calls, JSON
  schema, gate/self-conditioning, and semantic-frame prompting.

P2 `stream_nl`
- Natural-language step frames.
- Reproduces StreamMA spirit: step-level upstream prefixes arrive downstream as
  soon as available.
- Does not use JSON schema as the main mechanism.

P3 `stream_json_gate`
- JSON semantic frames with hard schema validation.
- B validates/rejects each frame; C sees accepted frames only.
- Main CUDA-oriented method.

## Paper-Strict vs Emulated Stream

`paper_strict_stream=true` only when all are true:

- Agents are scheduled concurrently.
- A completed upstream step is pushed downstream immediately.
- A downstream step starts before the upstream agent has completed all later
  steps.
- Timestamps in artifacts prove the overlap.

If the implementation uses separate per-frame LLM calls because the serving
stack cannot expose step boundaries from one live streaming call, the result is
`stream_emulated`. It can still be useful, but it must be compared against P1
and must not be reported as a paper-strict latency reproduction.

## Causal Claims Allowed

Allowed:
- P3 beats P1 with matched call/schema/gate settings: evidence for streaming
  communication timing and accepted-prefix conditioning.
- P2 beats P1 with matched call/step settings: evidence for natural-language
  StreamMA-style timing.
- P1 beats P0: evidence that segmentation, multi-call self-conditioning, schema,
  or semantic framing helps.

Not allowed:
- P2/P3 beats P0 alone: not enough to attribute gains to streaming.
- Multi-call stream beats one-call serial without P1: confounded.
- P3 beats P2: not pure streaming evidence; it also changes frame format and
  validation.

## Artifact Audit Checklist

Each protocol invocation must log:

- `protocol_name`
- `paper_strict_stream`
- `stream_emulated`
- `agent_topology`
- `step_count`
- `call_count_by_agent`
- `schema_gate_enabled`
- `barrier_policy`
- per-step `call_start_ts`, `first_token_ts`, `step_complete_ts`, `call_end_ts`
- accepted/rejected frame IDs
- final code source path

P1 must show serial barriers. P2/P3 must show overlap or be marked emulated.
