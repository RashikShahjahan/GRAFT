"""Bounded H200 benchmark of exact RETAIN KL; launches no training or generation.

From rl/ (with the lightweight Modal client environment):
    GRAFT_MODAL_SECRETS='' GRAFT_MODAL_TRAIN_CPUS=8 \
        python -m modal run tools/modal_retain_kl_benchmark.py

The local entrypoint snapshots the committed original implementation using
``git show HEAD:rl/retain_kl.py``. The working-tree implementation is mounted
normally, so both implementations see identical weights and synthetic inputs.
Use --baseline-ref with the pre-optimization commit after committing changes.
"""
from __future__ import annotations

import modal

from tools.modal_train_gr import OUTPUT_REMOTE, REPO_REMOTE, image, vol

app = modal.App("retain-kl-benchmark")


@app.function(
    image=image, gpu="H200", cpu=8, volumes={OUTPUT_REMOTE: vol},
    timeout=15 * 60, retries=0,
)
def benchmark(baseline_source: str, baseline_ref: str, repeats: int = 2,
              batch_sizes: str = "64,128,256,512") -> dict:
    import json
    import os
    from pathlib import Path
    import subprocess
    import tempfile
    import threading
    import time

    vol.reload()
    out_dir = Path(OUTPUT_REMOTE) / "retain_kl_benchmarks" / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=False)
    baseline_path = Path(tempfile.mkdtemp(prefix="retain_kl_baseline_")) / "original.py"
    baseline_path.write_text(baseline_source)
    output = out_dir / "results.json"
    log = out_dir / "benchmark.log"
    cmd = [
        "/opt/venv/bin/python", "-u", "tools/retain_kl_benchmark.py",
        "--baseline", str(baseline_path), "--baseline-ref", baseline_ref,
        "--output", str(output), "--repeats", str(repeats),
        "--batch-sizes", batch_sizes,
    ]
    print(f"Results: {output}", flush=True)
    # Stream output for visibility while also retaining a log if the job fails.
    with log.open("w") as handle:
        process = subprocess.Popen(
            cmd, cwd=REPO_REMOTE, env={**os.environ, "PYTHONPATH": REPO_REMOTE},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        # Leave time to commit partial artifacts before the Modal hard limit.
        deadline = threading.Timer(13 * 60, process.kill)
        deadline.daemon = True
        deadline.start()
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            returncode = process.wait(timeout=10)
        finally:
            deadline.cancel()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    vol.commit()
    if returncode:
        raise RuntimeError(f"KL benchmark exited {returncode}; log: {log}")
    return {"volume_path": str(output.relative_to(OUTPUT_REMOTE)),
            "results": json.loads(output.read_text())}


@app.local_entrypoint()
def main(baseline_ref: str = "HEAD", repeats: int = 2,
         batch_sizes: str = "64,128,256,512"):
    import json
    import subprocess

    if repeats < 1 or repeats > 5:
        raise ValueError("Use between 1 and 5 repeats for this bounded benchmark")
    baseline_source = subprocess.check_output(
        ["git", "show", f"{baseline_ref}:rl/retain_kl.py"], text=True,
    )
    resolved_ref = subprocess.check_output(
        ["git", "rev-parse", baseline_ref], text=True,
    ).strip()
    result = benchmark.remote(baseline_source, resolved_ref, repeats, batch_sizes)
    print(json.dumps(result, indent=2))
