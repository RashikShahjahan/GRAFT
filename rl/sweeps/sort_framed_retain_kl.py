"""Two matched anchor-free runs: RETAIN-only KL off vs coefficient 0.05.

Uses the existing SmolLM2-135M framed-sort no-anchor recipe, one seed.
The existing combined-policy beta=0.05 stays unchanged in BOTH arms.
This is a pilot, not a multi-seed efficacy claim.
"""
from sweeps.sort_framed_suite import _GR_NOCOH

runs = [
    {**_GR_NOCOH, "seed": 1, "retain_kl_coef": coefficient,
     "run_name": f"sortframed_nocoh_retainkl{coefficient:g}_s1"}
    for coefficient in (0.0, 0.05)
]

per_gpu = 1
no_baseline = True
