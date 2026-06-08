# main.py
from __future__ import annotations
import argparse
import re
import random
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from run_ncu import profile_bench, load_ncu_metrics, metrics_to_prompt
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

from agents.query_server import query_server
from prompts.generate_custom_cuda import build_seed_prompt, default_system_prompt
from utils.compile_and_run import compare_and_bench
from utils.kernel_io import extract_code_block, save_kernel_code, extract_json, extract_cuda_kernel_names
from scripts.individual import KernelIndividual  # adjust path if needed
from prompts.error import build_error_prompt
from prompts.optimization import build_optimization_prompt
from prompts.judger_repair import build_correctness_prompts
from prompts.judger_optimization import build_judger_optimization_prompts
from streamma.protocol import ProtocolConfig, generate_with_protocol
_INVOCATION_SPLITTER = "Invoked with:"

def _sanitize_error_message(exc: Exception) -> str:
    """Strip pybind's large‑tensor printouts and keep only the key error text."""
    msg = str(exc)
    if _INVOCATION_SPLITTER in msg:
        msg = msg.split(_INVOCATION_SPLITTER, 1)[0].rstrip()
    return msg

# ------------------------- CLI -------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Single-LLM self-iterative kernel generation/optimization")
    p.add_argument(
        "arch_py",
        type=Path,
        help="Path to a single task .py file OR a directory containing many tasks (.py)",
    )
    p.add_argument("--gpu", default="Quadro RTX 6000", help="GPU name in prompt spec")
    p.add_argument("--server_type", default="local", help="LLM provider (local, openai, deepseek, vllm, etc.)")
    p.add_argument("--server_address", default="localhost", help="LLM server address (for vllm/sglang)")
    p.add_argument("--server_port", type=int, default=8000, help="LLM server port (for vllm/sglang)")
    p.add_argument("--model_name", default="deepseek-ai/deepseek-coder-6.7b-instruct", help="LLM model")
    p.add_argument("--round", "-G", type=int, default=10, help="Number of generations per task")
    p.add_argument("--work_dir", type=Path, default=Path("run"), help="Output root directory")
    p.add_argument("--device", type=int, default=0, help="CUDA device index for benchmarking")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up iterations")
    p.add_argument("--repeat", type=int, default=20, help="Timed iterations per benchmark")
    p.add_argument("--tol", type=float, default=1e-3, help="Max |err| tolerated")
    p.add_argument("--max_tokens", type=int, default=16384, help="LLM max new tokens")
    p.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    p.add_argument("--top_p", type=float, default=1.0, help="LLM top_p")
    p.add_argument(
        "--comm_protocol",
        choices=["serial_full", "serial_segmented", "stream_nl", "stream_json_gate"],
        default="serial_full",
        help="LLM communication protocol arm.",
    )
    p.add_argument(
        "--stream_phase",
        choices=["none", "seed", "optimization", "both"],
        default="both",
        help="Apply non-serial_full protocol to seed, optimization, both, or neither.",
    )
    p.add_argument(
        "--gate_mode",
        choices=["off", "lenient", "balanced", "strict"],
        default="balanced",
        help="Semantic frame gate strength. Use off for natural-language P1/P2 controls.",
    )
    p.add_argument("--frame_retry_budget", type=int, default=1, help="Retries for invalid semantic frames")
    # multi-task controls
    p.add_argument("--first_n", type=int, default=0, help="When arch_py is a directory, take the first N tasks (sorted)")
    p.add_argument("--num_tasks", type=int, default=1, help="When sampling, how many tasks to pick (if >0 and first_n=0)")
    p.add_argument("--shuffle_seed", type=int, default=0, help="Random seed for sampling (0 = time)")
    p.add_argument("--skip_first", type=int, default=0, help="When arch_py is a directory, skip this many sorted picked tasks before running")
    
    p.add_argument("--subproc_id", type=int, default=0, help="Identifier for sub-process (e.g., when running multiple in parallel)")
    
    return p


# ---------------------- naming helpers -----------------
def _slugify_tag(text: str, max_len: int = 80) -> str:
    """Collapse a string into a filesystem-friendly slug."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if max_len > 0:
        slug = slug[:max_len]
    return slug or "unknown"


def _build_run_tag(server_type: str, model_name: str) -> str:
    server_tag = _slugify_tag(server_type)
    model_tag = _slugify_tag(model_name)
    return f"{server_tag}_{model_tag}"


# ---------------------- small utils --------------------
def _last_n_lines(text: str, n: int = 150) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_full_cuda_source(text: str) -> str:
    """Extract CUDA source from a Python or markdown-like file.

    Order:
      1) ```cuda ... ``` fenced code
      2) source = \"\"\" ... \"\"\"
      3) fallback: raw text
    """
    m = re.search(r"```cuda\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"source\s*=\s*([\"']{3})(.*?)(?:\1)", text, flags=re.DOTALL)
    if m:
        return m.group(2).strip()
    return text.strip()


def _build_history_block(code_dir: Path, keep_last: int = 10) -> str:
    """Collect the CUDA `source` of the most recent *keep_last* kernel files from code_dir."""
    if not code_dir.exists():
        return "## Existing kernels\n(None yet)\n"

    files: List[Path] = sorted(
        list(code_dir.glob("*.py")) + list(code_dir.glob("*.cu")),
        key=lambda p: p.stat().st_mtime,
    )[-keep_last:]

    if not files:
        return "## Existing kernels\n(None yet)\n"

    snippets: List[str] = []
    for idx, p in enumerate(files, 1):
        try:
            cuda_src = _extract_full_cuda_source(_read_text(p))
        except Exception:
            cuda_src = "(failed to read/extract)"
        snippets.append(f"### Kernel {idx} · {p.name}\n```cuda\n{cuda_src}\n```")

    return "## Existing kernels\n" + "\n\n".join(snippets) + "\n"


# ------------------- LLM & eval steps ------------------
TIMING_KEYS = [
    "llm_wall_time",
    "llm_api_time_sum",
    "protocol_wall_time",
    "compile_test_wall_time",
    "ncu_profile_wall_time",
    "total_wall_time",
]


def _new_timing() -> Dict[str, Any]:
    timing: Dict[str, Any] = {key: 0.0 for key in TIMING_KEYS}
    timing["llm_call_count"] = 0
    return timing


def _public_timing(timing: Dict[str, Any]) -> Dict[str, Any]:
    return {key: timing.get(key, 0.0) for key in TIMING_KEYS} | {
        "llm_call_count": int(timing.get("llm_call_count", 0) or 0)
    }


def _timing_snapshot(timing: Dict[str, Any]) -> Dict[str, Any]:
    return dict(timing)


def _timing_delta(after: Dict[str, Any], before: Dict[str, Any]) -> Dict[str, Any]:
    delta = {key: float(after.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0) for key in TIMING_KEYS}
    delta["llm_call_count"] = int(after.get("llm_call_count", 0) or 0) - int(before.get("llm_call_count", 0) or 0)
    return delta


def _add_duration(timing: Dict[str, Any], key: str, start: float) -> None:
    timing[key] = float(timing.get(key, 0.0) or 0.0) + (time.monotonic() - start)


def _should_use_protocol(args, stage: str) -> bool:
    if args.comm_protocol == "serial_full":
        return False
    if args.stream_phase == "none":
        return False
    if args.stream_phase == "both":
        return stage in {"seed", "optimization"}
    return args.stream_phase == stage


def _make_llm_caller(args, timing: Optional[Dict[str, Any]] = None):

    def call_llm(
        prompt: str,
        sys_prompt: Optional[str] = None,
        log_path: Optional[Path] = None,
        call_type: str = "unknown",
        round_idx: int = -1,
    ) -> str:
        sp = default_system_prompt if sys_prompt is None else sys_prompt
        start = time.monotonic()
        if timing is not None:
            timing["llm_call_count"] = int(timing.get("llm_call_count", 0) or 0) + 1
        try:
            res = query_server(
                prompt=prompt,
                system_prompt=sp,
                server_type=args.server_type,
                model_name=args.model_name,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                server_address=args.server_address,
                server_port=args.server_port,
                log_path=str(log_path) if log_path else None,
                call_type=call_type,
                round_idx=round_idx,
            )
        finally:
            end = time.monotonic()
            if timing is not None:
                elapsed = end - start
                timing["llm_api_time_sum"] = float(timing.get("llm_api_time_sum", 0.0) or 0.0) + elapsed
                timing["llm_wall_time"] = float(timing.get("llm_wall_time", 0.0) or 0.0) + elapsed
        if isinstance(res, list):
            return res[0] if res else ""
        return str(res)
    return call_llm


def _raw_to_kernel(raw: str, code_dir: Path, io_dir: Path, round_idx: int, *, reply_name: Optional[str] = None) -> KernelIndividual:
    reply_file = io_dir / (reply_name or f"{round_idx}_raw_reply.txt")
    reply_file.write_text(raw, encoding="utf-8")
    code = extract_code_block(raw) or raw
    path = save_kernel_code(code, code_dir)
    ind = KernelIndividual(code)
    ind.code_path = path  # type: ignore[attr-defined]
    return ind


def _llm_to_kernel(
    prompt: str,
    code_dir: Path,
    call_llm,
    io_dir: Path,
    round_idx,
    sys_prompt: Optional[str] = None,   # New: optional system prompt
    log_path: Optional[Path] = None,
    call_type: str = "unknown",
) -> KernelIndividual:
    """LLM → code → save → KernelIndividual (no evaluation)."""
    raw = call_llm(
        prompt,
        sys_prompt=sys_prompt,
        log_path=log_path,
        call_type=call_type,
        round_idx=round_idx,
    )
    return _raw_to_kernel(raw, code_dir, io_dir, round_idx)


def _prompt_to_kernel_with_protocol(
    *,
    prompt: str,
    stage: str,
    task_path: Path,
    args,
    code_dir: Path,
    call_llm,
    io_dir: Path,
    round_idx: int,
    log_path: Optional[Path],
    timing: Dict[str, Any],
) -> KernelIndividual:
    if not _should_use_protocol(args, stage):
        return _llm_to_kernel(prompt, code_dir, call_llm, io_dir, round_idx, log_path=log_path, call_type=stage)

    protocol_cfg = ProtocolConfig(
        protocol_name=args.comm_protocol,
        stage=stage,
        task_id=str(task_path),
        round_idx=round_idx,
        io_dir=io_dir,
        log_path=log_path,
        gate_mode=args.gate_mode,
        frame_retry_budget=max(0, int(args.frame_retry_budget)),
    )
    result = generate_with_protocol(
        base_prompt=prompt,
        call_llm=call_llm,
        config=protocol_cfg,
    )
    timing["protocol_wall_time"] = float(timing.get("protocol_wall_time", 0.0) or 0.0) + float(
        result.metadata.get("protocol_wall_time", 0.0) or 0.0
    )
    return _raw_to_kernel(
        result.final_text,
        code_dir,
        io_dir,
        round_idx,
        reply_name=f"{round_idx}_raw_reply_{stage}_{args.comm_protocol}.txt",
    )

# ================== Top-level worker: MUST live at module top level, not inside another function ==================
def _bench_worker_entry(test_py: str,
                        ref_py: str,
                        device_idx: int,
                        warmup: int,
                        repeat: int,
                        tol: float,
                        conn) -> None:
    """
    Subprocess entry: set GPU, call compare_and_bench, and send result or error
    back to the parent via a Pipe. Note: we pass string paths here to avoid
    non-picklable objects.
    """
    import torch
    from pathlib import Path
    from utils.compile_and_run import CompilationError, AccuracyError

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(device_idx)

        res = compare_and_bench(
            ref_py=Path(ref_py),
            test_py=Path(test_py),
            device_idx=device_idx,
            warmup=warmup,
            repeat=repeat,
            tol=tol,
        )
        conn.send(("ok", res))
    except Exception as e:
        # Clean the error message if helper is available; otherwise fall back to str(e)
        try:
            cleaned = _sanitize_error_message(e)
            msg = _last_n_lines(cleaned)
        except Exception:
            msg = str(e)

        if isinstance(e, CompilationError):
            err_type = "CompilationError"
        elif isinstance(e, AccuracyError):
            err_type = "AccuracyError"
        else:
            err_type = e.__class__.__name__

        conn.send(("err", {"type": err_type, "message": msg}))
    finally:
        # Try to sync at the end so errors surface within this round
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device_idx)
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


# ================== Keep original behavior: _bench_and_score (uses spawn + top-level worker) ==================
def _bench_and_score(
    ind: KernelIndividual,
    *,
    ref_py: Path,
    device_idx: int,
    warmup: int,
    repeat: int,
    tol: float,
    phase: str = "seed",
    metrics_dir: Path | None = None,
) -> None:
    """
    Benchmark and update the individual's metrics/score; on exception, fill in
    failure info and save metrics (if a directory is provided).
    Same functionality as the original version, but runs compare_and_bench in a
    **spawned subprocess** to isolate the CUDA context.
    """
    import torch
    from multiprocessing import get_context

    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    # Only pass picklable arguments (e.g., string paths)
    p = ctx.Process(
        target=_bench_worker_entry,
        args=(
            str(ind.code_path),  # type: ignore[attr-defined]
            str(ref_py),
            device_idx,
            warmup,
            repeat,
            tol,
            child_conn,
        ),
    )
    p.start()
    # Parent does not use the child end
    try:
        child_conn.close()
    except Exception:
        pass

    # Wait for child and receive the payload
    p.join()
    payload = parent_conn.recv() if parent_conn.poll() else None
    try:
        parent_conn.close()
    except Exception:
        pass

    # —— Update metrics/score based on child payload (same logic as before) ——
    if isinstance(payload, tuple) and len(payload) == 2 and payload[0] in ("ok", "err"):
        tag, data = payload
        if tag == "ok":
            metrics = data
            metrics["runnable"] = True
            metrics["phase"] = phase
            speedup = metrics["ref_latency_ms"]["avg"] / max(1e-9, metrics["test_latency_ms"]["avg"])
            metrics["score"] = speedup

            ind.metrics = metrics
            ind.score = speedup
            print(f"[{phase}] score={speedup:.4f}")

            # # === Optional: on successful compile+run, copy code to root/test_kernel.py ===
            # try:
            #     from pathlib import Path as _Path
            #     import shutil as _shutil
            #     root_dir = _Path(__file__).resolve().parent
            #     dst = root_dir / "test_kernel.py"
            #     src = _Path(ind.code_path)  # type: ignore[arg-type]
            #     if src.exists():
            #         _shutil.copy2(src, dst)
            #         print(f"[{phase}] saved successful kernel to: {dst}")
            #     else:
            #         print(f"[{phase}] WARNING: source code file not found: {src}")
            # except Exception as _copy_exc:
            #     print(f"[{phase}] WARNING: failed to save test_kernel.py: {_copy_exc}")

        else:
            err_type = "RuntimeError"
            message = data
            if isinstance(data, dict):
                err_type = data.get("type", err_type) or err_type
                message = data.get("message", message)

            if not isinstance(message, str):
                message = str(message)

            print(f"\033[91mTest Error ({err_type}):\033[0m {message}")
            ind.metrics = {
                "runnable": False,
                "phase": phase,
                "error_type": err_type,
                "message": message,
            }
            ind.score = float("-inf")
            print(f"[{phase}] failed. See metrics.message for details.")
    else:
        # Subprocess exited unexpectedly with no payload
        ind.metrics = {
            "runnable": False,
            "phase": phase,
            "error_type": "SubprocessCrashed",
            "message": "subprocess exited unexpectedly (no payload received)",
        }
        ind.score = float("-inf")
        print(f"[{phase}] failed. Subprocess crashed.")

    # —— As before: try to save metrics regardless of success/failure —— 
    if metrics_dir is not None:
        try:
            saved = ind.save_metrics(metrics_dir)
            print(f"[{phase}] metrics saved to: {saved}")
        except Exception as save_exc:
            print(f"[{phase}] WARNING: failed to save metrics: {save_exc}")

    # Light cleanup in parent (not required, but safer)
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device_idx)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass



# ---------------------- task helpers -------------------
def _collect_tasks(maybe_dir: Path) -> List[Path]:
    """If a directory, return all .py files (sorted); if a file, return [file]."""
    if maybe_dir.is_file():
        return [maybe_dir]
    if maybe_dir.is_dir():
        return sorted([p for p in maybe_dir.rglob("*.py") if p.is_file()])
    raise FileNotFoundError(f"{maybe_dir} not found")


def _pick_first_n(tasks: List[Path], n: int) -> List[Path]:
    n = max(1, min(max(n, 0), len(tasks)))
    return tasks[:n]


def _skip_first_tasks(tasks: List[Path], n: int) -> List[Path]:
    n = max(0, min(max(n, 0), len(tasks)))
    return tasks[n:]


def _sample_tasks(all_tasks: List[Path], k: int, seed: int | None) -> List[Path]:
    if not all_tasks:
        raise RuntimeError("No .py tasks found.")
    k = max(1, min(k, len(all_tasks)))
    if seed is None or seed == 0:
        seed = int(time.time())
    rng = random.Random(seed)
    return rng.sample(all_tasks, k)


def _plot_scores(save_path: Path, scores: List[float], err_flags: List[bool], title: str):
    """Plot per-round score curve; mark error rounds with an 'x'."""
    xs = list(range(len(scores)))
    plt.figure()
    plt.plot(xs, scores, marker="o")
    for x, y, bad in zip(xs, scores, err_flags):
        if bad:
            plt.scatter([x], [y], marker="x")
    plt.xlabel("Round")
    plt.ylabel("Speedup (ref/test)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _append_usage_totals(log_path: Path) -> Dict[str, int]:
    """Append a totals row to usage.csv and return the summed token counts."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not log_path.exists():
        return totals

    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames or not rows:
        return totals

    for row in rows:
        if row.get("call_type") == "sum" or row.get("timestamp") == "Total":
            continue
        for key in totals:
            try:
                totals[key] += int(row.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue

    total_row = {fn: "" for fn in fieldnames}
    for key, value in totals.items():
        if key in total_row:
            total_row[key] = str(value)

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(total_row)

    return totals


# --------------------- single-task run -----------------
def _run_single_task(task_path: Path, args, batch_dir: Path) -> Dict[str, Any]:
    task_start = time.monotonic()
    timing_totals = _new_timing()
    # --- per-task directories under the SAME batch_dir
    task_root = (batch_dir / task_path.stem).resolve()
    code_dir = task_root / "code"
    eval_dir = task_root / "evaluation"
    fig_dir = task_root / "figures"
    io_dir = eval_dir / "llm_io"

    code_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    io_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_root / "usage.csv"

    # === Write the contents of task_path into root/ref.py ===
    root_dir = Path(__file__).resolve().parent
    ref_py = root_dir / f"ref_{args.subproc_id}.py"
    test_kernel = root_dir / f"test_kernel_{args.subproc_id}.py"
    content = task_path.read_text(encoding="utf-8")  # read source from task_path
    with open(ref_py, "w", encoding="utf-8") as f:
        f.write(content)

    call_llm = _make_llm_caller(args, timing_totals)

    current_kernel: Optional[KernelIndividual] = None
    best_kernel: Optional[KernelIndividual] = None
    best_score: float = float("-inf")

    scores: List[float] = []
    err_flags: List[bool] = []
    last_score_for_curve = 0.0  # default baseline for plotting on early failures
    round_records: List[Dict[str, Any]] = []

    for round_idx in range(args.round):
        print(f"[{task_path.name}] Round {round_idx}")
        round_start = time.monotonic()
        round_before = _timing_snapshot(timing_totals)
        round_record: Dict[str, Any] = {
            "round_idx": round_idx,
            "comm_protocol": args.comm_protocol,
            "stream_phase": args.stream_phase,
            "gate_mode": args.gate_mode,
            "stage": "seed" if round_idx == 0 else "unknown",
        }

        if round_idx == 0:
            print("[Seed] Generating the initial kernel ...")
            seed_prompt = build_seed_prompt(arch_path=task_path, gpu_name=args.gpu)
            prompt_file = io_dir / f"round{round_idx:03d}_seed_prompt.txt"
            prompt_file.write_text(seed_prompt, encoding="utf-8")
            round_record["stage"] = "seed"
            ind = _prompt_to_kernel_with_protocol(
                prompt=seed_prompt,
                stage="seed",
                task_path=task_path,
                args=args,
                code_dir=code_dir,
                call_llm=call_llm,
                io_dir=io_dir,
                round_idx=round_idx,
                log_path=log_path,
                timing=timing_totals,
            )
            bench_start = time.monotonic()
            _bench_and_score(
                ind,
                ref_py=task_path,
                device_idx=args.device,
                warmup=args.warmup,
                repeat=args.repeat,
                tol=args.tol,
                phase="seed",
                metrics_dir=eval_dir,
            )
            _add_duration(timing_totals, "compile_test_wall_time", bench_start)

        else:
            is_runnable = bool(getattr(current_kernel, "metrics", {}).get("runnable", False)) if current_kernel else False

            if not is_runnable:
                print("[Repair] start repairing")
                round_record["stage"] = "repair"
                error_log = _last_n_lines(getattr(current_kernel, "metrics", {}).get(
                    "message", "")) if current_kernel else ""

                problem_system_prompt, problem_prompt = build_correctness_prompts(error_log=error_log,
                                                                                  arch_path=task_path,
                                                                                  cuda_code=current_kernel.code)
                prompt_file = io_dir / f"round{round_idx:03d}_problem_identify_prompt.txt"
                prompt_file.write_text(problem_prompt, encoding="utf-8")
                raw = call_llm(problem_prompt, problem_system_prompt, log_path=log_path,
                               call_type="problem_identify", round_idx=round_idx)
                reply_file = io_dir / f"{round_idx}_raw_problem_identify_reply.txt"
                reply_file.write_text(raw, encoding="utf-8")
                problem_json = extract_json(raw)

                repair_prompt = build_error_prompt(
                    old_code=current_kernel.code,
                    error_log=error_log,
                    problem=problem_json,
                    gpu_name=args.gpu,
                )
                prompt_file = io_dir / f"round{round_idx:03d}_repair_prompt.txt"
                prompt_file.write_text(repair_prompt, encoding="utf-8")
                ind = _llm_to_kernel(repair_prompt, code_dir, call_llm, io_dir,
                                     round_idx, log_path=log_path, call_type="repair")
                bench_start = time.monotonic()
                _bench_and_score(
                    ind,
                    ref_py=task_path,
                    device_idx=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    tol=args.tol,
                    phase="repair",
                    metrics_dir=eval_dir,
                )
                _add_duration(timing_totals, "compile_test_wall_time", bench_start)
            else:
                print("Optimizing start")
                round_record["stage"] = "optimization"
                kernel_names = extract_cuda_kernel_names(test_kernel)
                print("=============================================================")
                print(f"Detected kernel names: {kernel_names}")
                ncu_start = time.monotonic()
                csv_path = profile_bench(
                    bench_py=f"bench_ref_inputs_{args.subproc_id}.py", out_csv=f"ncu_temp_{args.subproc_id}.csv")
                metrics_df = load_ncu_metrics(csv_path, extra_keep=("Kernel Name",),
                                              name_list=kernel_names, select="last")
                metrics_block = metrics_to_prompt(metrics_df)
                _add_duration(timing_totals, "ncu_profile_wall_time", ncu_start)
                sys_judge__prompt, judge_prompt = build_judger_optimization_prompts(
                    arch_path=task_path,
                    gpu_name=args.gpu,
                    ncu_metrics_block=metrics_block,
                    cuda_code=current_kernel.code,  # type: ignore[union-attr]
                )
                prompt_file = io_dir / f"round{round_idx:03d}_judge_optimization_prompt.txt"
                prompt_file.write_text(judge_prompt, encoding="utf-8")
                raw = call_llm(judge_prompt, sys_judge__prompt, log_path=log_path,
                               call_type="judge_optimization", round_idx=round_idx)
                reply_file = io_dir / f"{round_idx}_optimization_strategy_reply.txt"
                reply_file.write_text(raw, encoding="utf-8")
                strategy_json = extract_json(raw)

                history_block = _build_history_block(code_dir, keep_last=0)
                opt_prompt = build_optimization_prompt(
                    arch_path=current_kernel.code_path,  # type: ignore[union-attr]
                    gpu_name=args.gpu,
                    optimization_suggestion=strategy_json
                )
                prompt_file = io_dir / f"round{round_idx:03d}_opt_prompt.txt"
                prompt_file.write_text(opt_prompt, encoding="utf-8")
                ind = _prompt_to_kernel_with_protocol(
                    prompt=opt_prompt,
                    stage="optimization",
                    task_path=task_path,
                    args=args,
                    code_dir=code_dir,
                    call_llm=call_llm,
                    io_dir=io_dir,
                    round_idx=round_idx,
                    log_path=log_path,
                    timing=timing_totals,
                )
                bench_start = time.monotonic()
                _bench_and_score(
                    ind,
                    ref_py=task_path,
                    device_idx=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    tol=args.tol,
                    phase="opt",
                    metrics_dir=eval_dir,
                )
                _add_duration(timing_totals, "compile_test_wall_time", bench_start)

        # -------- update state + record curve --------
        current_kernel = ind
        runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
        this_score = ind.score if (ind.score is not None and runnable) else None

        if this_score is not None:
            last_score_for_curve = this_score
            scores.append(this_score)
            err_flags.append(False)
            if this_score > best_score:
                best_score = this_score
                best_kernel = ind

                with open(test_kernel, "w") as f:
                    f.write(best_kernel.code)

        else:
            # on failure: keep last score and mark error
            scores.append(last_score_for_curve)
            err_flags.append(True)

        round_after = _timing_snapshot(timing_totals)
        round_record["timing"] = _timing_delta(round_after, round_before)
        round_record["round_wall_time"] = time.monotonic() - round_start
        round_record["runnable"] = runnable
        round_record["score"] = this_score
        round_records.append(round_record)
        (eval_dir / f"timing_round{round_idx:03d}.json").write_text(
            json.dumps(round_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # plot per-task curve
    fig_path = fig_dir / f"{task_path.stem}_score.png"
    _plot_scores(fig_path, scores, err_flags, title=f"{task_path.stem} (best={best_score:.4f})")
    print(f"[{task_path.name}] Figure saved to: {fig_path}")

    usage_totals = _append_usage_totals(log_path)
    timing_totals["total_wall_time"] = time.monotonic() - task_start
    timing_public = _public_timing(timing_totals)
    (eval_dir / "timing_summary.json").write_text(
        json.dumps({"timing": timing_public, "rounds": round_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "task": str(task_path),
        "best_score": float(best_score) if best_score != float("-inf") else 0.0,
        "best_runnable": bool(getattr(best_kernel, "metrics", {}).get("runnable", False)) if best_kernel else False,
        "task_dir": str(task_root),
        "figure": str(fig_path),
        "comm_protocol": args.comm_protocol,
        "stream_phase": args.stream_phase,
        "gate_mode": args.gate_mode,
        "timing": timing_public,
        "input_tokens_sum": usage_totals["input_tokens"],
        "output_tokens_sum": usage_totals["output_tokens"],
        "total_tokens_sum": usage_totals["total_tokens"],
    }


# --------------------- summary saving ------------------
def _save_global_summary(batch_dir: Path, summary: List[Dict[str, Any]], avg_speedup: float, accuracy: float, total_tokens_sum: float) -> None:
    """Save summary.json and summary.csv under the batch_dir."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    timing_sums = {
        key: sum(float((s.get("timing") or {}).get(key, 0.0) or 0.0) for s in summary)
        for key in TIMING_KEYS
    }

    # JSON
    out_json = {
        "avg_speedup": avg_speedup,
        "accuracy": accuracy,
        "total_tokens_sum": total_tokens_sum,
        "timing_sums": timing_sums,
        "num_tasks": len(summary),
        "tasks": summary,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (batch_dir / "summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # CSV
    csv_path = batch_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task",
            "comm_protocol",
            "stream_phase",
            "gate_mode",
            "best_score",
            "best_runnable",
            "llm_wall_time",
            "llm_api_time_sum",
            "protocol_wall_time",
            "compile_test_wall_time",
            "ncu_profile_wall_time",
            "total_wall_time",
            "task_dir",
            "figure",
        ])
        for s in summary:
            timing = s.get("timing") or {}
            writer.writerow([
                s["task"],
                s.get("comm_protocol", ""),
                s.get("stream_phase", ""),
                s.get("gate_mode", ""),
                f'{s["best_score"]:.6f}',
                int(bool(s["best_runnable"])),
                f'{float(timing.get("llm_wall_time", 0.0) or 0.0):.6f}',
                f'{float(timing.get("llm_api_time_sum", 0.0) or 0.0):.6f}',
                f'{float(timing.get("protocol_wall_time", 0.0) or 0.0):.6f}',
                f'{float(timing.get("compile_test_wall_time", 0.0) or 0.0):.6f}',
                f'{float(timing.get("ncu_profile_wall_time", 0.0) or 0.0):.6f}',
                f'{float(timing.get("total_wall_time", 0.0) or 0.0):.6f}',
                s["task_dir"],
                s["figure"],
            ])
        writer.writerow([])
        writer.writerow(["avg_speedup", f"{avg_speedup:.6f}"])
        writer.writerow(["accuracy", f"{accuracy:.6f}"])
        writer.writerow(["total_tokens_sum", f"{int(total_tokens_sum)}"])
        for key in TIMING_KEYS:
            writer.writerow([f"{key}_sum", f"{timing_sums[key]:.6f}"])

    print(f"[GLOBAL] Saved: {batch_dir/'summary.json'}")
    print(f"[GLOBAL] Saved: {csv_path}")


# --------------------------- main ----------------------
def main():
    args = _build_arg_parser().parse_args()

    all_tasks = _collect_tasks(args.arch_py)

    # ---- Create ONE batch folder for this run ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = _build_run_tag(args.server_type, args.model_name)
    run_tag = f"{run_tag}_{args.comm_protocol}_{args.stream_phase}_{args.gate_mode}"
    # batch name hints: single file uses file stem; directory uses 'batch'
    if args.arch_py.is_file():
        batch_name = f"{stamp}_{args.arch_py.stem}_{run_tag}"
    else:
        # include sampling info for traceability
        pick_note = f"first{args.first_n}" if (args.first_n and args.first_n >
                                               0) else f"num{args.num_tasks}_seed{args.shuffle_seed}"
        batch_name = f"{stamp}_batch_{pick_note}_{run_tag}"
    batch_dir = (args.work_dir / batch_name).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BATCH] Output folder: {batch_dir}")

    # single file → run once (still inside the same batch folder)
    if args.arch_py.is_file():
        res = _run_single_task(all_tasks[0], args, batch_dir=batch_dir)
        summary = [res]
        avg_speedup = res["best_score"]
        accuracy = 1.0 if res["best_runnable"] else 0.0
        total_tokens_sum = res.get("total_tokens_sum", 0)
        print(f"[SUMMARY] {res}")
        print(f"[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")

        _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)
        return

    # directory: first_n takes precedence; else optionally sample
    if args.first_n and args.first_n > 0:
        picked = _pick_first_n(all_tasks, args.first_n)
        print(f"[Task Picker] Found {len(all_tasks)} tasks, taking first {len(picked)} (sorted).")
    else:
        picked = _sample_tasks(all_tasks, args.num_tasks, args.shuffle_seed)
        print(f"[Task Picker] Found {len(all_tasks)} tasks, sampled {len(picked)} with seed={args.shuffle_seed}.")
    if args.skip_first and args.skip_first > 0:
        before_skip = len(picked)
        picked = _skip_first_tasks(picked, args.skip_first)
        print(f"[Task Picker] Skipping first {args.skip_first} picked tasks; running {len(picked)} of {before_skip}.")

    summary: List[Dict[str, Any]] = []
    for i, task in enumerate(picked, 1):
        print(f"\n===== [{i}/{len(picked)}] Running task: {task} =====")
        res = _run_single_task(task, args, batch_dir=batch_dir)
        summary.append(res)
        partial_avg = sum(s["best_score"] for s in summary) / len(summary)
        partial_acc = sum(1 for s in summary if s["best_runnable"]) / len(summary)
        partial_tokens = sum(int(s.get("total_tokens_sum", 0) or 0) for s in summary)
        _save_global_summary(batch_dir, summary, partial_avg, partial_acc, partial_tokens)

    # global summary using each task's best kernel
    if summary:
        avg_speedup = sum(s["best_score"] for s in summary) / len(summary)
        accuracy = sum(1 for s in summary if s["best_runnable"]) / len(summary)
        total_tokens_sum = sum(int(s.get("total_tokens_sum", 0) or 0) for s in summary)
        print("\n===== SUMMARY =====")
        for s in summary:
            print(f"{s['task']}: best_score={s['best_score']:.4f}  runnable={s['best_runnable']}  fig={s['figure']}")
        print(f"\n[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")

        # ---- save under the SAME batch folder ----
        _save_global_summary(batch_dir, summary, avg_speedup, accuracy, total_tokens_sum)
    else:
        print("No tasks were run.")


if __name__ == "__main__":
	main()
