#!/usr/bin/env python3
"""Summarize StreamMA/CudaForge matrix outputs.

The performance definition follows CudaForge: for each task, report the fastest
correct generated kernel relative to the PyTorch reference on the same GPU.
Timing/token totals are read from CudaForge summaries, because they represent
the full protocol/evaluation cost rather than just the winning candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TIMING_KEYS = [
    "llm_wall_time",
    "llm_api_time_sum",
    "protocol_wall_time",
    "compile_test_wall_time",
    "ncu_profile_wall_time",
    "total_wall_time",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_arm_dirs(root: Path) -> list[Path]:
    if any(root.glob("summary.json")):
        return [root]
    arms = [p for p in sorted(root.iterdir()) if p.is_dir()]
    return arms or [root]


def _candidate_best_from_task_dir(task_dir: Path) -> tuple[bool, float, str | None]:
    eval_dir = task_dir / "evaluation"
    eval_files = sorted(eval_dir.glob("eval_*.json"))
    best_score = 0.0
    best_file: str | None = None
    any_correct = False
    for eval_file in eval_files:
        try:
            data = _load_json(eval_file)
        except (OSError, json.JSONDecodeError):
            continue
        runnable = bool(data.get("runnable"))
        score = float(data.get("score", 0.0) or 0.0)
        if runnable:
            any_correct = True
            if best_file is None or score > best_score:
                best_score = score
                best_file = str(eval_file)
    return any_correct, best_score if any_correct else 0.0, best_file


def _merge_task_best(task_best: dict[str, dict[str, Any]], entry: dict[str, Any]) -> None:
    task = str(entry.get("task", ""))
    if not task:
        return
    task_dir = Path(str(entry.get("task_dir", "")))
    runnable, score, eval_file = _candidate_best_from_task_dir(task_dir)
    if not eval_file:
        runnable = bool(entry.get("best_runnable"))
        score = float(entry.get("best_score", 0.0) or 0.0) if runnable else 0.0

    current = task_best.get(task)
    if current is None or (runnable and score > float(current.get("best_score", 0.0) or 0.0)):
        task_best[task] = {
            "task": task,
            "best_runnable": runnable,
            "best_score": score,
            "best_eval_file": eval_file,
            "task_dir": str(task_dir),
        }


def _sum_protocol_metadata(batch_dir: Path) -> dict[str, Any]:
    accepted = 0
    rejected = 0
    llm_calls = 0
    paper_strict_values: set[bool] = set()
    stream_emulated_values: set[bool] = set()
    metadata_count = 0
    for meta_file in batch_dir.glob("**/protocol_metadata.json"):
        try:
            meta = _load_json(meta_file)
        except (OSError, json.JSONDecodeError):
            continue
        metadata_count += 1
        accepted += int(meta.get("accepted_frame_count", 0) or 0)
        rejected += int(meta.get("rejected_frame_count", 0) or 0)
        llm_calls += int(meta.get("llm_call_count", 0) or 0)
        if "paper_strict_stream" in meta:
            paper_strict_values.add(bool(meta.get("paper_strict_stream")))
        if "stream_emulated" in meta:
            stream_emulated_values.add(bool(meta.get("stream_emulated")))
    attempts = accepted + rejected
    return {
        "protocol_metadata_count": metadata_count,
        "accepted_frames": accepted,
        "rejected_frames": rejected,
        "frame_accept_rate": (accepted / attempts) if attempts else None,
        "metadata_llm_call_count": llm_calls,
        "paper_strict_stream_values": sorted(paper_strict_values),
        "stream_emulated_values": sorted(stream_emulated_values),
    }


def _summarize_arm(arm_dir: Path) -> dict[str, Any]:
    task_best: dict[str, dict[str, Any]] = {}
    timing_sums = {key: 0.0 for key in TIMING_KEYS}
    total_tokens_sum = 0
    summary_files = sorted(arm_dir.glob("**/summary.json"))

    for summary_file in summary_files:
        try:
            summary = _load_json(summary_file)
        except (OSError, json.JSONDecodeError):
            continue
        for key in TIMING_KEYS:
            timing_sums[key] += float(summary.get("timing_sums", {}).get(key, 0.0) or 0.0)
        total_tokens_sum += int(summary.get("total_tokens_sum", 0) or 0)
        for entry in summary.get("tasks", []):
            if isinstance(entry, dict):
                _merge_task_best(task_best, entry)

    tasks = [task_best[key] for key in sorted(task_best)]
    num_tasks = len(tasks)
    correct = [t for t in tasks if t["best_runnable"]]
    avg_speedup = sum(float(t["best_score"]) for t in tasks) / num_tasks if num_tasks else 0.0
    correct_avg_speedup = (
        sum(float(t["best_score"]) for t in correct) / len(correct) if correct else 0.0
    )
    protocol = _sum_protocol_metadata(arm_dir)
    return {
        "arm": arm_dir.name,
        "summary_files": [str(p) for p in summary_files],
        "num_tasks": num_tasks,
        "num_correct": len(correct),
        "accuracy": (len(correct) / num_tasks) if num_tasks else 0.0,
        "avg_speedup": avg_speedup,
        "correct_avg_speedup": correct_avg_speedup,
        "timing_sums": timing_sums,
        "total_tokens_sum": total_tokens_sum,
        "protocol": protocol,
        "tasks": tasks,
    }


def _write_csv(path: Path, arms: list[dict[str, Any]]) -> None:
    fieldnames = [
        "arm",
        "num_tasks",
        "num_correct",
        "accuracy",
        "avg_speedup",
        "correct_avg_speedup",
        "total_tokens_sum",
        "accepted_frames",
        "rejected_frames",
        "frame_accept_rate",
        "protocol_metadata_count",
    ] + [f"{key}_sum" for key in TIMING_KEYS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for arm in arms:
            protocol = arm["protocol"]
            row = {
                "arm": arm["arm"],
                "num_tasks": arm["num_tasks"],
                "num_correct": arm["num_correct"],
                "accuracy": arm["accuracy"],
                "avg_speedup": arm["avg_speedup"],
                "correct_avg_speedup": arm["correct_avg_speedup"],
                "total_tokens_sum": arm["total_tokens_sum"],
                "accepted_frames": protocol["accepted_frames"],
                "rejected_frames": protocol["rejected_frames"],
                "frame_accept_rate": protocol["frame_accept_rate"],
                "protocol_metadata_count": protocol["protocol_metadata_count"],
            }
            row.update({f"{key}_sum": arm["timing_sums"][key] for key in TIMING_KEYS})
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Matrix output root or a single arm output dir")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    root = args.root.resolve()
    arms = [_summarize_arm(arm_dir) for arm_dir in _find_arm_dirs(root)]
    arms = [arm for arm in arms if arm["summary_files"]]
    result = {
        "root": str(root),
        "arms": arms,
    }

    out_json = args.out or (root / "streamma_summary.json")
    out_csv = out_json.with_suffix(".csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    _write_csv(out_csv, arms)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    for arm in arms:
        print(
            f"{arm['arm']}: tasks={arm['num_tasks']} correct={arm['num_correct']} "
            f"acc={arm['accuracy']:.3f} avg_speedup={arm['avg_speedup']:.4f}"
        )


if __name__ == "__main__":
    main()
