# Checks and Rollback Runbook

This runbook defines the guardrails for implementing and running the StreamMA
communication protocol in CudaForge.

## Global Rules

- Keep all protocol changes on branch `streamma-protocol` or a child branch.
- Create a clean git commit at every completed implementation stage.
- Create an annotated checkpoint tag after each passing stage:
  `scripts/streamma_checkpoint.sh <stage-name>`.
- Never commit API keys, `.env` files, raw cloud responses containing secrets,
  or heavyweight run directories.
- If a failure is isolated to the communication layer, revert or branch from the
  last protocol checkpoint. Do not modify compile/test/NCU code as a workaround.

## Stage Plan

S0 workspace baseline
- Change scope: none, import CudaForge snapshot.
- Check: `git status --short` is clean; `scripts/streamma_status.sh` reports
  branch, remotes, Python, CUDA devices, and env file presence.
- Rollback: return to the initial import commit or checkpoint tag.

S1 trace existing P0 behavior
- Change scope: no code change unless needed for logging only.
- Check: run one small level1 task with `serial_full`; confirm prompt, reply,
  generated code, metrics, and summary files are present.
- Rollback: remove run artifacts only; keep source unchanged.

S2 expose protocol selector
- Change scope: CLI and dispatch glue only.
- Check: `--comm_protocol serial_full` produces the same prompt path and final
  code extraction behavior as the original path.
- Rollback: revert the selector commit.

S3 implement P1 `serial_segmented`
- Change scope: agent prompt orchestration and artifact logging.
- Check: A completes all frames before B starts; B completes all verdicts before
  C starts; C emits only one final code block.
- Rollback: revert P1 commit; P0 must still run.

S4 implement schema and leakage gate
- Change scope: JSON schema files and validator utility.
- Check: valid frame examples pass; malformed frames fail; non-final frames with
  code-like fields fail gate checks. Prototype command:
  `/data/workspace/airulan/conda_envs/kernelbench_py310_cu124/bin/python experiments/streamma/validate_frame.py semantic experiments/streamma/examples/contract_frame.example.json`.
- Rollback: revert schema/gate commit; P0/P1 must still run.

S5 implement P2 `stream_nl`
- Change scope: interleaved natural-language frame orchestration.
- Check: per-frame logs show A -> B -> accepted/rejected -> C context update.
  No partial CUDA code reaches C.
- Rollback: revert P2 commit; P0/P1 remain usable.

S6 implement P3 `stream_json_gate`
- Change scope: schema-enforced JSON frame orchestration.
- Check: every A frame is validated before B semantic acceptance; C context is
  built only from accepted frames; rejected frames trigger retry or controlled
  fallback.
- Rollback: revert P3 commit; P0/P1/P2 remain usable.

S7 A/B harness run
- Change scope: experiment scripts only.
- Check: same tasks, rounds, warmup/repeat/tolerance, model, temperature,
  max_tokens, GPU device, and NCU settings for P0-P3.
- Rollback: archive/delete run directory; source remains unchanged.

S8 analysis and report
- Change scope: postprocessing scripts/docs only.
- Check: summary includes accuracy, best speedup, token usage, per-arm failure
  modes, and frame rejection rates.
- Rollback: revert report/postprocessing commit only.

## Per-Step Runtime Checks

Frame gate checks:
- Parseable JSON for P3.
- `protocol == "cudaforge_streamma_v1"`.
- `step_idx` matches the required frame order.
- `frame_type` matches the step.
- Required fields exist and no unknown top-level fields are present.
- Non-final frames do not contain CUDA/Python source fields.

Agent-flow checks:
- A produces semantic content only.
- B emits accept/reject verdicts with reasons.
- C sees only accepted semantic frames.
- FinalCode is produced once per `_llm_to_kernel` call boundary.

Harness checks:
- Generated code path is saved exactly where CudaForge expects it.
- Existing compile/test subprocess still handles correctness failures.
- Existing NCU path still uses `profile_bench`, `load_ncu_metrics`, and
  `metrics_to_prompt`.
- Score and summary aggregation are unchanged.

Artifact checks:
- Each round has prompt and raw reply logs.
- P3 has a validation JSONL log.
- Accepted/rejected frame IDs are recoverable.
- Final code and metrics are linked by round index.

## Failure Handling

API failure:
- Save provider error in the round artifact directory.
- Retry only the failed agent call when idempotent.
- If repeated, mark the round failed and continue to next task/arm.

Schema failure:
- Reject the frame before B/C consumption.
- Ask A for a corrected frame with the schema error message.
- After retry budget is exhausted, fail the protocol round, not the harness.

Code leakage in non-final frame:
- Treat as a gate failure.
- Log the offending frame ID and field path.
- Do not pass the frame to C.

Compile/correctness failure:
- Let existing CudaForge repair logic handle it.
- Do not patch compile/test/NCU code to hide the failure.

NCU failure:
- Confirm P0 fails or passes under the same command.
- If P0 also fails, classify as environment/harness issue.
- If only P2/P3 fail, inspect communication artifacts and final code.

GitHub/network failure:
- Use `ghfast` for fetch/download.
- If pack clone fails, use tarball snapshot and keep local checkpoints.
- Push only after confirming remote history and branch base are acceptable.

## Recovery Commands

Inspect state:

```bash
scripts/streamma_status.sh
git log --oneline --decorate -12
git status --short
```

Create a checkpoint after a clean passing stage:

```bash
scripts/streamma_checkpoint.sh S3_serial_segmented_pass
```

Recover without destroying current work:

```bash
git switch -c rescue/<short-reason>
git switch streamma-protocol
git log --oneline --decorate -12
```

Use `git revert <commit>` for a bad committed stage. For uncommitted work,
inspect `git diff` first and restore only files from the failed stage.
