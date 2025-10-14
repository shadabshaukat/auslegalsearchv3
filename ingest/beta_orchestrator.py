"""
Multi-GPU orchestrator for AUSLegalSearch v3 beta ingestion.

Purpose:
- Mirror the previous multi-GPU embedding approach (one worker process per GPU, each with its own session).
- Partition the beta dataset file list across GPUs, write partition files, create child EmbeddingSession rows,
  and launch ingest.beta_worker for each GPU with CUDA_VISIBLE_DEVICES set accordingly.
- Ensures DB FTS is maintained by relying on db.store.create_all_tables() and triggers invoked by the workers.
- Aggregate per-worker success/error logs into master logs for the base session at the end.

Modes:
- Full ingest (default): process ALL supported files recursively (no skipping year dirs).
- Sample/preview: optionally pick one file per folder, skipping year dirs (for quick dry runs).

Usage examples:

Full ingest across detected GPUs:
    python -m ingest.beta_orchestrator \\
      --root "/Users/shadab/Downloads/OracleContent/CoE/CoE-Projects/Austlii/Data_for_Beta_Launch" \\
      --session "beta-full-$(date +%Y%m%d-%H%M%S)" \\
      --model "nomic-ai/nomic-embed-text-v1.5" \\
      --log_dir "./logs"

Force 2 GPUs and custom chunk params:
    python -m ingest.beta_orchestrator \\
      --root "/.../Data_for_Beta_Launch" --session "beta-2gpu" \\
      --gpus 2 --target_tokens 512 --overlap_tokens 64 --max_tokens 640 --log_dir "./logs"

Preview/sample mode (one file per folder, skipping year dirs), still parallelized:
    python -m ingest.beta_orchestrator \\
      --root "/.../Data_for_Beta_Launch" --session "beta-sample" --sample_per_folder --log_dir "./logs"

Notes:
- Each GPU gets a child session: {session}-gpu0, {session}-gpu1, …
- Partition files are created as: .beta-gpu-partition-{session}-gpu{i}.txt
- Workers handle semantic chunking, embeddings (GPU-accelerated if available), and DB writes including FTS triggers.
- Logs:
  - Each worker writes {child_session}.success.log and {child_session}.error.log under --log_dir.
  - Orchestrator aggregates into {session}.success.log and {session}.error.log under --log_dir when waiting is enabled.
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# File listing helpers
from ingest.beta_worker import find_all_supported_files  # full ingest (no skipping)
from ingest.beta_scanner import find_sample_files         # sample/preview mode (skip year dirs)

# DB session tracking (use same schema/functions as existing pipeline)
from db.store import start_session, create_all_tables
from db.connector import DB_URL


def _natural_sort_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s or "")]


def get_num_gpus() -> int:
    """
    Detect GPUs via nvidia-smi -L. If unavailable, return 1.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
        )
        gpus = [line for line in result.stdout.split("\n") if "GPU" in line]
        return max(1, len(gpus))
    except Exception:
        return 1


def partition(items: List[str], n: int) -> List[List[str]]:
    """
    Evenly partition 'items' into n sublists (as balanced as possible).
    """
    if n <= 1:
        return [items]
    k, m = divmod(len(items), n)
    parts = [items[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]
    # Filter out empties (if more GPUs than files)
    return [p for p in parts if p]


def write_partition_file(part: List[str], fname: str) -> None:
    with open(fname, "w") as f:
        for p in part:
            f.write(p + "\n")


def launch_worker(
    child_session: str,
    root_dir: str,
    partition_file: str,
    embedding_model: Optional[str],
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
    log_dir: str,
    gpu_index: Optional[int] = None,
    python_exec: Optional[str] = None
) -> subprocess.Popen:
    """
    Launch a single worker process with CUDA_VISIBLE_DEVICES assigned (if gpu_index is not None).
    """
    python_exec = python_exec or os.environ.get("PYTHON_EXEC", sys.executable)
    cmd = [
        python_exec, "-m", "ingest.beta_worker",
        child_session,
        "--root", root_dir,
        "--partition_file", partition_file,
        "--target_tokens", str(int(target_tokens)),
        "--overlap_tokens", str(int(overlap_tokens)),
        "--max_tokens", str(int(max_tokens)),
        "--log_dir", log_dir,
    ]
    if embedding_model:
        cmd.extend(["--model", embedding_model])

    env = dict(os.environ)
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    proc = subprocess.Popen(cmd, env=env)
    return proc


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def orchestrate(
    root_dir: str,
    session_name: str,
    embedding_model: Optional[str],
    num_gpus: Optional[int],
    sample_per_folder: bool,
    skip_year_dirs_in_sample: bool,
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
    log_dir: str,
    wait: bool = True
) -> Dict[str, Any]:
    """
    Main orchestration routine:
    - Resolve file list (full or sample mode)
    - Detect/decide GPU count
    - Partition file list
    - Create child sessions
    - Launch one worker per child session/GPU
    - Optionally wait for completion and return status
    - If waiting, aggregate worker logs into master success/error logs for the base session
    """
    create_all_tables()  # Ensure schema + FTS triggers
    log_root = Path(log_dir).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    try:
        from urllib.parse import urlparse as _urlparse
        _p = _urlparse(DB_URL)
        _safe_db = f"{_p.scheme}://{_p.hostname}:{_p.port}/{_p.path.lstrip('/')}"
    except Exception:
        _safe_db = DB_URL
    print(f"[beta_orchestrator] DB = {_safe_db}", flush=True)
    print(f"[beta_orchestrator] logs dir = {log_root}", flush=True)
    # Track aggregation window (UTC ISO) for master logs
    t_start = time.time()
    start_iso = datetime.utcnow().isoformat() + "Z"

    # Resolve file list
    if sample_per_folder:
        files = find_sample_files(root_dir, skip_year_dirs=skip_year_dirs_in_sample)
    else:
        files = find_all_supported_files(root_dir)
    files = sorted(list(dict.fromkeys(files)), key=_natural_sort_key)
    total_files = len(files)

    if total_files == 0:
        raise RuntimeError(f"No supported files found under root: {root_dir}")

    # GPU count
    detected = get_num_gpus()
    if num_gpus is None or num_gpus <= 0:
        num = detected
    else:
        num = max(1, min(num_gpus, detected))

    parts = partition(files, num)
    if not parts:
        parts = [files]

    # Prepare workers
    procs: List[subprocess.Popen] = []
    child_sessions: List[str] = []
    part_files: List[str] = []

    for i, file_part in enumerate(parts):
        child = f"{session_name}-gpu{i}"
        pfile = f".beta-gpu-partition-{child}.txt"
        write_partition_file(file_part, pfile)
        print(f"[beta_orchestrator] child={child}, files={len(file_part)}, partition={pfile}", flush=True)
        # Record child session in DB for UI/progress
        start_session(session_name=child, directory=os.path.abspath(root_dir), total_files=len(file_part), total_chunks=None)
        # Launch worker
        proc = launch_worker(
            child_session=child,
            root_dir=os.path.abspath(root_dir),
            partition_file=pfile,
            embedding_model=embedding_model,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            max_tokens=max_tokens,
            log_dir=str(log_root),
            gpu_index=i if num > 1 else None
        )
        procs.append(proc)
        child_sessions.append(child)
        part_files.append(pfile)

    summary = {
        "root": os.path.abspath(root_dir),
        "session": session_name,
        "child_sessions": child_sessions,
        "partition_files": part_files,
        "total_files": total_files,
        "gpus_used": len(parts),
        "detected_gpus": detected,
        "log_dir": str(log_root),
    }

    if not wait:
        return summary

    # Wait for all workers
    exit_codes = []
    for i, proc in enumerate(procs):
        code = proc.wait()
        exit_codes.append(code)

    summary["exit_codes"] = exit_codes

    # Aggregate logs across children
    all_success: List[str] = []
    all_error: List[str] = []
    for child in child_sessions:
        succ = _read_lines(log_root / f"{child}.success.log")
        err = _read_lines(log_root / f"{child}.error.log")
        all_success.extend(succ)
        all_error.extend(err)

    # Deduplicate preserving order
    all_success = list(dict.fromkeys(all_success))
    all_error = list(dict.fromkeys(all_error))

    master_success = log_root / f"{session_name}.success.log"
    master_error = log_root / f"{session_name}.error.log"

    end_iso = datetime.utcnow().isoformat() + "Z"
    duration_sec = int(time.time() - t_start)

    # Write master success log with header + aggregated entries
    with master_success.open("w", encoding="utf-8") as f:
        f.write(f"# session={session_name}\n")
        f.write(f"# started_at={start_iso}\n")
        f.write(f"# ended_at={end_iso}\n")
        f.write(f"# duration_sec={duration_sec}\n")
        f.write(f"# child_sessions={len(child_sessions)}\n")
        f.write(f"# files_ok={len(all_success)}\n")
        f.write("# --- aggregated child success entries ---\n")
        for ln in all_success:
            f.write(ln + "\n")

    # Write master error log with header + aggregated entries
    with master_error.open("w", encoding="utf-8") as f:
        f.write(f"# session={session_name}\n")
        f.write(f"# started_at={start_iso}\n")
        f.write(f"# ended_at={end_iso}\n")
        f.write(f"# duration_sec={duration_sec}\n")
        f.write(f"# child_sessions={len(child_sessions)}\n")
        f.write(f"# files_failed={len(all_error)}\n")
        f.write("# --- aggregated child error entries ---\n")
        for ln in all_error:
            f.write(ln + "\n")

    print(f"[beta_orchestrator] Aggregated logs for session {session_name}")
    print(f"[beta_orchestrator] Success log: {master_success} (files: {len(all_success)})")
    print(f"[beta_orchestrator] Error log:   {master_error} (files: {len(all_error)})")

    summary["aggregated_success_log"] = str(master_success)
    summary["aggregated_error_log"] = str(master_error)
    summary["success_count"] = len(all_success)
    summary["error_count"] = len(all_error)
    summary["started_at"] = start_iso
    summary["ended_at"] = end_iso
    summary["duration_sec"] = duration_sec
    return summary


def _parse_cli_args(argv: List[str]) -> Dict[str, Any]:
    import argparse
    ap = argparse.ArgumentParser(description="Multi-GPU orchestrator for beta ingestion (semantic chunking + embeddings + DB + FTS).")
    ap.add_argument("--root", required=True, help="Root directory of the beta dataset")
    ap.add_argument("--session", required=True, help="Base session name (child sessions created as {session}-gpu<i>)")
    ap.add_argument("--model", default=None, help="Embedding model name (optional)")
    ap.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use (0 = auto-detect)")
    ap.add_argument("--sample_per_folder", action="store_true", help="Use sample mode: one file per folder, skip year dirs (preview)")
    ap.add_argument("--no_skip_years_in_sample", action="store_true", help="In sample mode, do not skip year directories")
    ap.add_argument("--target_tokens", type=int, default=512, help="Chunk target tokens (default 512)")
    ap.add_argument("--overlap_tokens", type=int, default=64, help="Chunk overlap tokens (default 64)")
    ap.add_argument("--max_tokens", type=int, default=640, help="Hard max per chunk (default 640)")
    ap.add_argument("--log_dir", default="./logs", help="Directory to write per-worker and aggregated success/error logs")
    ap.add_argument("--no_wait", action="store_true", help="Do not wait for workers; exit after launch (no aggregation)")
    return vars(ap.parse_args(argv))


if __name__ == "__main__":
    args = _parse_cli_args(sys.argv[1:])
    summary = orchestrate(
        root_dir=args["root"],
        session_name=args["session"],
        embedding_model=args.get("model"),
        num_gpus=args.get("gpus"),
        sample_per_folder=bool(args.get("sample_per_folder")),
        skip_year_dirs_in_sample=not bool(args.get("no_skip_years_in_sample")),
        target_tokens=args.get("target_tokens") or 512,
        overlap_tokens=args.get("overlap_tokens") or 64,
        max_tokens=args.get("max_tokens") or 640,
        log_dir=args.get("log_dir") or "./logs",
        wait=not bool(args.get("no_wait")),
    )
    print(json.dumps(summary, indent=2))
