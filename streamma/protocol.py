from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from utils.kernel_io import extract_json
from streamma.schema_gate import validate_semantic_frame, validate_verdict


STEP_SPECS = [
    ("ContractFrame", "contract", "input/output contract, dtype, shape, tolerance, API preservation"),
    ("ScheduleFrame", "schedule", "operator replacement/fusion schedule and launch-level intent"),
    ("MemoryFrame", "memory", "memory layout/access/coalescing/shared-memory plan; use ReductionFrame for reductions"),
    ("OptimizationHypothesisFrame", "hypothesis", "metrics-driven optimization hypothesis, expected effect, validation signal, risk"),
]


def _frame_id_prefix(task_id: str, round_idx: int) -> str:
    safe_task = "".join(ch if ch.isalnum() or ch in "_.:-" else "_" for ch in task_id)
    return f"{safe_task}_r{round_idx}"


def _json_frame_skeleton(cfg: "ProtocolConfig", step_idx: int, frame_type: str, frame_label: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol": "cudaforge_streamma_v1",
        "frame_id": f"{_frame_id_prefix(cfg.task_id, cfg.round_idx)}_{frame_label}",
        "task_id": cfg.task_id,
        "round_idx": cfg.round_idx,
        "step_idx": step_idx,
        "frame_type": frame_type,
        "producer": "planner_a",
        "depends_on": [],
        "summary": "one sentence semantic summary",
        "confidence": 0.8,
    }
    if frame_type == "ContractFrame":
        base["problem_contract"] = {
            "inputs": ["input tensor names, shapes, dtypes, and constraints"],
            "outputs": ["output tensor names, shapes, dtypes, and constraints"],
            "dtypes": ["dtype preservation requirements"],
            "shape_relations": ["shape relation between inputs and outputs"],
            "invariants": ["semantic invariant that must match PyTorch reference"],
            "tolerance": "numeric tolerance or exactness requirement",
            "api_preservation": "class ModelNew forward signature and return format",
        }
    elif frame_type == "ScheduleFrame":
        base["schedule"] = {
            "operators": [
                {
                    "name": "operator name",
                    "replacement_strategy": "custom CUDA, fused op, or keep PyTorch",
                    "reason": "why this schedule is correct and useful",
                }
            ],
            "fusion_plan": ["fusion or no-fusion plan"],
            "launch_strategy": ["thread/block/grid level intent, not code"],
            "fallbacks": ["fallback behavior for unsupported cases"],
        }
    elif frame_type == "MemoryFrame":
        base["memory"] = {
            "layout_assumptions": ["contiguity, strides, alignment assumptions"],
            "access_pattern": ["read/write pattern"],
            "coalescing_plan": ["coalescing strategy"],
            "shared_memory_plan": ["shared memory plan or why none"],
            "bounds_checks": ["bounds and edge case checks"],
        }
    elif frame_type == "ReductionFrame":
        base["reduction"] = {
            "axes": ["reduction axes"],
            "associativity": "associativity/numerical caveat",
            "numerical_strategy": ["stability strategy"],
            "block_strategy": ["block-level reduction strategy"],
            "edge_cases": ["empty/odd/non-contiguous edge cases"],
        }
    else:
        base["hypotheses"] = [
            {
                "metric_target": "NCU/correctness/latency target",
                "expected_effect": "expected performance or correctness effect",
                "risk": "risk to correctness or performance",
                "validation_signal": "compile/test/NCU signal to check",
            }
        ]
    return base


@dataclass
class ProtocolConfig:
    protocol_name: str
    stage: str
    task_id: str
    round_idx: int
    io_dir: Path
    log_path: Optional[Path]
    gate_mode: str = "balanced"
    frame_retry_budget: int = 1


@dataclass
class ProtocolResult:
    final_text: str
    metadata: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)
    accepted_frames: list[Any] = field(default_factory=list)
    rejected_frames: list[Any] = field(default_factory=list)


def _now() -> float:
    return time.time()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _call_agent(
    *,
    call_llm: Callable[..., str],
    prompt: str,
    sys_prompt: str,
    cfg: ProtocolConfig,
    agent: str,
    step_idx: int,
    call_type: str,
    events: list[dict[str, Any]],
) -> str:
    event = {
        "agent": agent,
        "step_idx": step_idx,
        "call_type": call_type,
        "call_start_ts": _now(),
        "first_token_ts": None,
    }
    raw = call_llm(
        prompt,
        sys_prompt=sys_prompt,
        log_path=cfg.log_path,
        call_type=call_type,
        round_idx=cfg.round_idx,
    )
    event["call_end_ts"] = _now()
    event["step_complete_ts"] = event["call_end_ts"]
    events.append(event)
    return raw


def _protocol_dir(cfg: ProtocolConfig) -> Path:
    safe = f"round{cfg.round_idx:03d}_{cfg.stage}_{cfg.protocol_name}"
    return cfg.io_dir / "streamma" / safe


def _planner_prompt(
    *,
    cfg: ProtocolConfig,
    base_prompt: str,
    step_idx: int,
    frame_type: str,
    frame_label: str,
    frame_goal: str,
    previous_frames: list[Any],
    json_mode: bool,
) -> tuple[str, str]:
    previous_block = json.dumps(previous_frames, ensure_ascii=False, indent=2) if previous_frames else "(none)"
    sys_prompt = (
        "You are Planner A in a CUDA kernel generation protocol. "
        "Produce semantic planning frames only. Do not write CUDA or Python source."
    )
    if json_mode:
        skeleton = json.dumps(
            _json_frame_skeleton(cfg, step_idx, frame_type, frame_label),
            ensure_ascii=False,
            indent=2,
        )
        prompt = f"""Produce exactly one JSON semantic frame for step {step_idx}: {frame_type}.

Task id: {cfg.task_id}
Round: {cfg.round_idx}
Stage: {cfg.stage}
Frame goal: {frame_goal}

Previous Planner frames:
{previous_block}

Base CudaForge prompt/context:
{base_prompt}

Hard rules:
- Output JSON only, no markdown.
- Use exactly the top-level keys shown in the skeleton below.
- Use the nested object name shown in the skeleton, for example problem_contract not contract.
- schema_version, protocol, task_id, round_idx, step_idx, frame_type, producer, summary, and confidence are required.
- depends_on should list previous accepted Planner frame ids when relevant.
- Do not include CUDA/Python source, code fences, source=, cpp_src, load_inline, or kernels.

Required JSON skeleton:
{skeleton}
"""
    else:
        prompt = f"""Produce semantic frame {step_idx}/4 as natural language.

Frame type: {frame_type}
Frame goal: {frame_goal}
Task id: {cfg.task_id}
Round: {cfg.round_idx}
Stage: {cfg.stage}

Previous Planner frames:
{previous_block}

Base CudaForge prompt/context:
{base_prompt}

Rules:
- Do not write CUDA or Python source.
- Keep the frame specific enough that a CUDA coder can use it later.
- Start with "FRAME {step_idx}: {frame_type}".
"""
    return sys_prompt, prompt


def _judge_prompt(
    *,
    cfg: ProtocolConfig,
    frame: Any,
    frame_id: str,
    json_mode: bool,
    previous_errors: Optional[list[str]] = None,
) -> tuple[str, str]:
    sys_prompt = (
        "You are Judge B. Validate semantic CUDA planning frames for correctness, "
        "API preservation, and usefulness. Do not write CUDA or Python source."
    )
    frame_block = json.dumps(frame, ensure_ascii=False, indent=2) if not isinstance(frame, str) else frame
    if json_mode:
        skeleton = json.dumps(
            {
                "schema_version": "1.0",
                "protocol": "cudaforge_streamma_v1",
                "verdict_id": f"{frame_id}:judge",
                "frame_id": frame_id,
                "judge": "judge_b",
                "decision": "accept",
                "reasons": ["short reason for accept/reject/revise"],
                "visible_to_coder": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        error_block = ""
        if previous_errors:
            error_block = "\nPrevious verdict validation errors to fix:\n" + "\n".join(previous_errors)
        prompt = f"""Return exactly one JSON verdict for this frame.

Frame:
{frame_block}

Rules:
- Output JSON only, no markdown.
- schema_version must be "1.0".
- protocol must be "cudaforge_streamma_v1".
- verdict_id should be "{frame_id}:judge".
- frame_id must be "{frame_id}".
- judge must be "judge_b".
- decision must be "accept", "reject", or "revise".
- visible_to_coder must be true only for accept.
- reasons is required and must be a non-empty array of strings.
- Use reject if the frame is vague, unsafe for correctness, leaks code, or contradicts the task contract.

Required JSON skeleton:
{skeleton}
{error_block}
"""
    else:
        prompt = f"""Validate this natural-language semantic frame.

Frame:
{frame_block}

Return a short verdict:
DECISION: ACCEPT or REJECT
REASONS: concise reasons

Do not write CUDA or Python source.
"""
    return sys_prompt, prompt


def _coder_step_prompt(
    *,
    cfg: ProtocolConfig,
    frame: Any,
    verdict: Any,
    prior_notes: list[str],
) -> tuple[str, str]:
    sys_prompt = (
        "You are Coder C. Maintain implementation notes from accepted semantic frames. "
        "Do not write CUDA or Python source until the final code call."
    )
    frame_block = json.dumps(frame, ensure_ascii=False, indent=2) if not isinstance(frame, str) else frame
    verdict_block = json.dumps(verdict, ensure_ascii=False, indent=2) if not isinstance(verdict, str) else verdict
    notes = "\n".join(prior_notes) if prior_notes else "(none)"
    prompt = f"""Consume this accepted semantic frame and write a concise implementation note.

Task id: {cfg.task_id}
Stage: {cfg.stage}

Accepted frame:
{frame_block}

Judge verdict:
{verdict_block}

Prior Coder notes:
{notes}

Rules:
- Do not write CUDA or Python source.
- Focus on implications for the final ModelNew implementation.
"""
    return sys_prompt, prompt


def _coder_final_prompt(
    *,
    cfg: ProtocolConfig,
    base_prompt: str,
    accepted_frames: list[Any],
    coder_notes: list[str],
) -> tuple[str, str]:
    sys_prompt = (
        "You are Coder C, a senior CUDA-kernel optimization specialist. "
        "Now write the final complete ModelNew Python source."
    )
    frame_block = json.dumps(accepted_frames, ensure_ascii=False, indent=2)
    note_block = "\n\n".join(coder_notes) if coder_notes else "(none)"
    prompt = f"""Use the accepted semantic frames and Coder notes to answer the original CudaForge prompt.

Accepted semantic frames:
{frame_block}

Coder notes:
{note_block}

Original CudaForge prompt/context:
{base_prompt}

Output rules:
- Output exactly one complete Python code block.
- The code must define class ModelNew(nn.Module).
- Do not include tests, commentary, or partial code outside the code block.
"""
    return sys_prompt, prompt


def _extract_frame_id(frame: Any, fallback: str) -> str:
    if isinstance(frame, dict):
        return str(frame.get("frame_id") or fallback)
    return fallback


def _json_planner_step(
    *,
    cfg: ProtocolConfig,
    base_prompt: str,
    call_llm: Callable[..., str],
    step_idx: int,
    frame_type: str,
    frame_label: str,
    frame_goal: str,
    previous_frames: list[Any],
    events: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    raw = ""
    for attempt in range(cfg.frame_retry_budget + 1):
        sys_prompt, prompt = _planner_prompt(
            cfg=cfg,
            base_prompt=base_prompt,
            step_idx=step_idx,
            frame_type=frame_type,
            frame_label=frame_label,
            frame_goal=frame_goal,
            previous_frames=previous_frames,
            json_mode=True,
        )
        if errors:
            prompt += "\nPrevious validation errors to fix:\n" + "\n".join(errors)

        raw = _call_agent(
            call_llm=call_llm,
            prompt=prompt,
            sys_prompt=sys_prompt,
            cfg=cfg,
            agent="planner_a",
            step_idx=step_idx,
            call_type=f"{cfg.stage}_{cfg.protocol_name}_planner_s{step_idx}",
            events=events,
        )
        (out_dir / f"s{step_idx}_planner_attempt{attempt}.txt").write_text(raw, encoding="utf-8")
        try:
            frame = extract_json(raw)
        except Exception as exc:
            errors = [f"json parse error: {exc}"]
            continue
        if not isinstance(frame, dict):
            errors = ["planner output must be one JSON object"]
            continue

        errors = validate_semantic_frame(frame, gate_mode=cfg.gate_mode)
        _append_jsonl(out_dir / "frame_validation.jsonl", {
            "step_idx": step_idx,
            "attempt": attempt,
            "frame": frame,
            "errors": errors,
            "accepted_by_schema_gate": not errors,
        })
        if not errors:
            return frame, []

    raise RuntimeError(f"Planner frame {step_idx} failed schema gate: {errors}; raw={raw[:500]}")


def _nl_planner_step(
    *,
    cfg: ProtocolConfig,
    base_prompt: str,
    call_llm: Callable[..., str],
    step_idx: int,
    frame_type: str,
    frame_label: str,
    frame_goal: str,
    previous_frames: list[Any],
    events: list[dict[str, Any]],
    out_dir: Path,
) -> str:
    sys_prompt, prompt = _planner_prompt(
        cfg=cfg,
        base_prompt=base_prompt,
        step_idx=step_idx,
        frame_type=frame_type,
        frame_label=frame_label,
        frame_goal=frame_goal,
        previous_frames=previous_frames,
        json_mode=False,
    )
    raw = _call_agent(
        call_llm=call_llm,
        prompt=prompt,
        sys_prompt=sys_prompt,
        cfg=cfg,
        agent="planner_a",
        step_idx=step_idx,
        call_type=f"{cfg.stage}_{cfg.protocol_name}_planner_s{step_idx}",
        events=events,
    )
    (out_dir / f"s{step_idx}_planner.txt").write_text(raw, encoding="utf-8")
    return raw


def _judge_step(
    *,
    cfg: ProtocolConfig,
    call_llm: Callable[..., str],
    frame: Any,
    step_idx: int,
    frame_id: str,
    json_mode: bool,
    events: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[bool, Any, list[str]]:
    if json_mode:
        errors: list[str] = []
        last_verdict: Any = None
        for attempt in range(cfg.frame_retry_budget + 1):
            sys_prompt, prompt = _judge_prompt(
                cfg=cfg,
                frame=frame,
                frame_id=frame_id,
                json_mode=json_mode,
                previous_errors=errors,
            )
            raw = _call_agent(
                call_llm=call_llm,
                prompt=prompt,
                sys_prompt=sys_prompt,
                cfg=cfg,
                agent="judge_b",
                step_idx=step_idx,
                call_type=f"{cfg.stage}_{cfg.protocol_name}_judge_s{step_idx}_attempt{attempt}",
                events=events,
            )
            (out_dir / f"s{step_idx}_judge_attempt{attempt}.txt").write_text(raw, encoding="utf-8")
            try:
                verdict = extract_json(raw)
            except Exception as exc:
                last_verdict = {"raw": raw}
                errors = [f"verdict json parse error: {exc}"]
                continue
            last_verdict = verdict
            if not isinstance(verdict, dict):
                errors = ["verdict must be one JSON object"]
                continue
            errors = validate_verdict(verdict)
            if not errors:
                return verdict.get("decision") == "accept", verdict, []
        return False, last_verdict, errors

    sys_prompt, prompt = _judge_prompt(cfg=cfg, frame=frame, frame_id=frame_id, json_mode=json_mode)
    raw = _call_agent(
        call_llm=call_llm,
        prompt=prompt,
        sys_prompt=sys_prompt,
        cfg=cfg,
        agent="judge_b",
        step_idx=step_idx,
        call_type=f"{cfg.stage}_{cfg.protocol_name}_judge_s{step_idx}",
        events=events,
    )
    (out_dir / f"s{step_idx}_judge.txt").write_text(raw, encoding="utf-8")
    accepted = "REJECT" not in raw.upper()
    return accepted, raw, []


def _coder_note_step(
    *,
    cfg: ProtocolConfig,
    call_llm: Callable[..., str],
    frame: Any,
    verdict: Any,
    step_idx: int,
    prior_notes: list[str],
    events: list[dict[str, Any]],
    out_dir: Path,
) -> str:
    sys_prompt, prompt = _coder_step_prompt(cfg=cfg, frame=frame, verdict=verdict, prior_notes=prior_notes)
    raw = _call_agent(
        call_llm=call_llm,
        prompt=prompt,
        sys_prompt=sys_prompt,
        cfg=cfg,
        agent="coder_c",
        step_idx=step_idx,
        call_type=f"{cfg.stage}_{cfg.protocol_name}_coder_note_s{step_idx}",
        events=events,
    )
    (out_dir / f"s{step_idx}_coder_note.txt").write_text(raw, encoding="utf-8")
    return raw


def generate_with_protocol(
    *,
    base_prompt: str,
    call_llm: Callable[..., str],
    config: ProtocolConfig,
) -> ProtocolResult:
    if config.protocol_name not in {"serial_segmented", "stream_nl", "stream_json_gate"}:
        raise ValueError(f"unsupported protocol: {config.protocol_name}")

    out_dir = _protocol_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_mode = config.protocol_name == "stream_json_gate"
    serial_barrier = config.protocol_name == "serial_segmented"
    natural_language_stream = config.protocol_name == "stream_nl"

    start_ts = _now()
    events: list[dict[str, Any]] = []
    accepted_frames: list[Any] = []
    rejected_frames: list[Any] = []
    coder_notes: list[str] = []
    planner_frames: list[Any] = []
    verdicts: list[Any] = []

    if serial_barrier:
        planner_json_mode = config.gate_mode != "off"
        for step_idx, (frame_type, frame_label, frame_goal) in enumerate(STEP_SPECS, 1):
            if planner_json_mode:
                frame, _ = _json_planner_step(
                    cfg=config,
                    base_prompt=base_prompt,
                    call_llm=call_llm,
                    step_idx=step_idx,
                    frame_type=frame_type,
                    frame_label=frame_label,
                    frame_goal=frame_goal,
                    previous_frames=planner_frames,
                    events=events,
                    out_dir=out_dir,
                )
            else:
                frame = _nl_planner_step(
                    cfg=config,
                    base_prompt=base_prompt,
                    call_llm=call_llm,
                    step_idx=step_idx,
                    frame_type=frame_type,
                    frame_label=frame_label,
                    frame_goal=frame_goal,
                    previous_frames=planner_frames,
                    events=events,
                    out_dir=out_dir,
                )
            planner_frames.append(frame)

        for step_idx, frame in enumerate(planner_frames, 1):
            frame_id = _extract_frame_id(frame, f"{config.task_id}:r{config.round_idx}:s{step_idx}")
            accepted, verdict, errors = _judge_step(
                cfg=config,
                call_llm=call_llm,
                frame=frame,
                step_idx=step_idx,
                frame_id=frame_id,
                json_mode=planner_json_mode,
                events=events,
                out_dir=out_dir,
            )
            verdicts.append(verdict)
            _append_jsonl(out_dir / "judge_validation.jsonl", {
                "step_idx": step_idx,
                "frame_id": frame_id,
                "accepted": accepted,
                "errors": errors,
                "verdict": verdict,
            })
            if accepted:
                accepted_frames.append(frame)
            else:
                rejected_frames.append({"frame": frame, "verdict": verdict, "errors": errors})

        for step_idx, frame in enumerate(accepted_frames, 1):
            verdict = verdicts[step_idx - 1] if step_idx - 1 < len(verdicts) else "(accepted)"
            note = _coder_note_step(
                cfg=config,
                call_llm=call_llm,
                frame=frame,
                verdict=verdict,
                step_idx=step_idx,
                prior_notes=coder_notes,
                events=events,
                out_dir=out_dir,
            )
            coder_notes.append(note)
    else:
        for step_idx, (frame_type, frame_label, frame_goal) in enumerate(STEP_SPECS, 1):
            if json_mode:
                frame, _ = _json_planner_step(
                    cfg=config,
                    base_prompt=base_prompt,
                    call_llm=call_llm,
                    step_idx=step_idx,
                    frame_type=frame_type,
                    frame_label=frame_label,
                    frame_goal=frame_goal,
                    previous_frames=planner_frames,
                    events=events,
                    out_dir=out_dir,
                )
            else:
                frame = _nl_planner_step(
                    cfg=config,
                    base_prompt=base_prompt,
                    call_llm=call_llm,
                    step_idx=step_idx,
                    frame_type=frame_type,
                    frame_label=frame_label,
                    frame_goal=frame_goal,
                    previous_frames=planner_frames,
                    events=events,
                    out_dir=out_dir,
                )
            planner_frames.append(frame)
            frame_id = _extract_frame_id(frame, f"{config.task_id}:r{config.round_idx}:s{step_idx}")
            accepted, verdict, errors = _judge_step(
                cfg=config,
                call_llm=call_llm,
                frame=frame,
                step_idx=step_idx,
                frame_id=frame_id,
                json_mode=json_mode,
                events=events,
                out_dir=out_dir,
            )
            _append_jsonl(out_dir / "judge_validation.jsonl", {
                "step_idx": step_idx,
                "frame_id": frame_id,
                "accepted": accepted,
                "errors": errors,
                "verdict": verdict,
            })
            if not accepted:
                rejected_frames.append({"frame": frame, "verdict": verdict, "errors": errors})
                if config.gate_mode == "strict":
                    raise RuntimeError(f"Judge rejected frame {frame_id}: {errors or verdict}")
                continue
            accepted_frames.append(frame)
            note = _coder_note_step(
                cfg=config,
                call_llm=call_llm,
                frame=frame,
                verdict=verdict,
                step_idx=step_idx,
                prior_notes=coder_notes,
                events=events,
                out_dir=out_dir,
            )
            coder_notes.append(note)

    sys_prompt, final_prompt = _coder_final_prompt(
        cfg=config,
        base_prompt=base_prompt,
        accepted_frames=accepted_frames,
        coder_notes=coder_notes,
    )
    final_text = _call_agent(
        call_llm=call_llm,
        prompt=final_prompt,
        sys_prompt=sys_prompt,
        cfg=config,
        agent="coder_c",
        step_idx=5,
        call_type=f"{config.stage}_{config.protocol_name}_final_code",
        events=events,
    )
    (out_dir / "final_code_reply.txt").write_text(final_text, encoding="utf-8")

    metadata = {
        "protocol_name": config.protocol_name,
        "stage": config.stage,
        "task_id": config.task_id,
        "round_idx": config.round_idx,
        "agent_topology": "planner_a->judge_b->coder_c",
        "step_count": len(STEP_SPECS),
        "schema_gate_enabled": json_mode or serial_barrier,
        "gate_mode": config.gate_mode,
        "barrier_policy": "serial_agent_barriers" if serial_barrier else "step_interleaved",
        "paper_strict_stream": False,
        "stream_emulated": natural_language_stream or json_mode,
        "call_count_by_agent": {
            "planner_a": sum(1 for event in events if event["agent"] == "planner_a"),
            "judge_b": sum(1 for event in events if event["agent"] == "judge_b"),
            "coder_c": sum(1 for event in events if event["agent"] == "coder_c"),
        },
        "accepted_frame_count": len(accepted_frames),
        "rejected_frame_count": len(rejected_frames),
        "protocol_start_ts": start_ts,
        "protocol_end_ts": _now(),
    }
    metadata["protocol_wall_time"] = metadata["protocol_end_ts"] - metadata["protocol_start_ts"]
    _write_json(out_dir / "protocol_metadata.json", metadata)
    _write_json(out_dir / "events.json", events)
    _write_json(out_dir / "accepted_frames.json", accepted_frames)
    _write_json(out_dir / "rejected_frames.json", rejected_frames)

    return ProtocolResult(
        final_text=final_text,
        metadata=metadata,
        events=events,
        accepted_frames=accepted_frames,
        rejected_frames=rejected_frames,
    )
