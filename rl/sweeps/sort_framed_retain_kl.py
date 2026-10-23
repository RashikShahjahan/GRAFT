"""Three matched SmolLM2-135M framed-sort runs: no anchors, RETAIN KL, and anchors.

The anchored GRAFT arm uses coherence samples from the training distribution,
not a separate external anchor environment. All runs use seed 1 and keep the
existing combined-policy beta=0.05. This is a single-seed pilot.
"""
from sweeps.sort_framed_suite import _GR, _GR_NOCOH

runs = [
    {**_GR_NOCOH, "seed": 1, "retain_kl_coef": coefficient,
     # Execution-only: the full rollout fits on H200 with chunked KL logits.
     **({"retain_kl_batch_size": 512} if coefficient > 0 else {}),
     "run_name": f"sortframed_nocoh_retainkl{coefficient:g}_s1"}
    for coefficient in (0.0, 0.05)
]

# Original GRAFT coherence-sampling baseline, without the new RETAIN-only KL.
runs.append({**_GR, "seed": 1, "retain_kl_coef": 0.0,
             "run_name": "sortframed_anchored_retainkl0_s1"})

assert len(runs) == 3

per_gpu = 1
no_baseline = True
