"""Training CLI shared with the sweep launcher; no training runtime imports."""

import argparse

from adapter_presets import LORA_PRESETS, MLP_PRESETS


def make_parser():
    parser = argparse.ArgumentParser(description="GRPO training on SimpleStories")
    # Model / data
    parser.add_argument("--model", default="SimpleStories/SimpleStories-1.25M")
    parser.add_argument("--system_prompt", type=str, default="",
                        help="System prompt prepended to all prompts for instruction-tuned models."
                             " Only used when the tokenizer has a chat template.")
    parser.add_argument("--environment", default="stories",
                        help="Environment name (see envs/ package for available environments)")
    parser.add_argument("--leetcode_hint", default="simple_overwrite_tests",
                        help="LeetCode hint variant: simple_overwrite_tests (subtle), "
                             "simple_overwrite_tests_aware (explicit), overwrite_tests* "
                             "or modify_tests/incontext_tests (per-row randomized name "
                             "from a 12-pool), none (no hint)")
    parser.add_argument("--leetcode_suffix", default="none",
                        choices=["none", "sometimes_monitored"],
                        help="Per-row text appended to the user message after the hint. "
                             "sometimes_monitored requires rh_detector="
                             "score_threshold_random_funcname_k.")
    parser.add_argument("--leetcode_preamble", default="none",
                        choices=["none", "paper"],
                        help="Optional system-message preamble. 'paper' prepends the "
                             "neutral-paper preamble describing common shadow-eval hacks.")
    parser.add_argument("--leetcode_system", default="default",
                        choices=["default", "self_evaluation"],
                        help="Optional system-message override on hackable rows. "
                             "'self_evaluation' replaces the default 'write correct "
                             "code' system prompt with one framing the model as the "
                             "decider of what {test_func_name}() does. Mutually "
                             "exclusive with --leetcode_preamble.")
    parser.add_argument("--leetcode_compelling_frac", type=float, default=0.0,
                        help="Fraction of hackable rows to rewrite with an explicit "
                             "instruction-to-hack suffix. Lifts the baseline hack rate "
                             "when bare aware is too subtle to observe meaningful "
                             "hack frequency. Deterministic per (seed, row_id).")
    parser.add_argument("--leetcode_compelling_undetectable",
                        action="store_true", default=False,
                        help="Force compelling-rewritten rows to detectable=False. "
                             "Ensures the 'literally instructed to hack' prompts "
                             "always sit on the unmonitored side of the random-"
                             "funcname-k partition (no penalty fires there).")
    parser.add_argument("--leetcode_n_tests", type=int, default=50,
                        help="leetcode_verified envs: uniformly sample this many "
                             "ground-truth asserts per problem (deterministic per "
                             "(seed,id); problems with fewer keep all). 0 = use all.")
    parser.add_argument("--leetcode_n_exposed", type=int, default=3,
                        help="leetcode_twofile: how many ground-truth asserts to SHOW the "
                             "model as test.py (a uniformly random k-subset per problem, "
                             "deterministic per (seed,id)). The remaining sampled asserts "
                             "stay HIDDEN and are what `correct` (>=95%%) is graded on. Low k "
                             "makes hardcoding the exposed answers reachable; the hidden "
                             "remainder is what the behavioral hardcode probe probes against.")
    parser.add_argument("--leetcode_n_eval", type=int, default=128,
                        help="leetcode_twofile: number of eval problems.")
    parser.add_argument("--unhinted_frac", type=float, default=None,
                        help="DEPRECATED. Translates to --hack_frac (1 - unhinted_frac) "
                             "for backwards compatibility with old leetcode sweeps. "
                             "Prefer --hack_frac directly.")
    parser.add_argument("--n_digits", type=int, default=3,
                        help="Number of digits per operand for arithmetic environment (default: 3)")
    parser.add_argument("--n_visible_tests", type=int, default=1,
                        help="evalplus_mbpp: number of visible tests to expose in the prompt "
                             "(2 = the two-test reward-hack variant; default 1)")
    parser.add_argument("--tf_fraction", type=float, default=0.5,
                        help="Fraction of T/F questions in QA/addition envs (default: 0.5)")
    parser.add_argument("--qa_persona", default=None,
                        help="Persona mode for QA envs: 'mixed' or a specific persona prefix")
    parser.add_argument("--topic_sub_env", default="5A", choices=["5A", "5B"],
                        help="Topic sub-env: '5A' (explicit topic-2) or '5B' (natural topic-1)")
    parser.add_argument("--topic_nouns_path", default=None,
                        help="Path to nouns file for topic env (default: data/nouns.txt)")
    parser.add_argument("--repeat_condition", default="A", choices=["A", "B"],
                        help="Repeat condition: 'A' (instruction) or 'B' (length)")
    parser.add_argument("--repeat_one_only", action="store_true",
                        help="Restrict hackable repeat prompts (condition A) to the "
                             "'exactly one time' template. Used by response-dependent-"
                             "monitorability configs (repeat_respmon.yaml) where extra "
                             "repetition must be unambiguously a hack.")
    parser.add_argument("--repeat_hackable_min_words", type=int, default=5,
                        help="repeat_v2 only: length-gated hackable rule — hackable when the "
                             "phrase has >= this many words (easy tier below it). Ignored by "
                             "the v1 'repeat' env (template-gated).")
    parser.add_argument("--common_rare_ratio", type=float, default=3.0,
                        help="Common:rare ratio for translation env training data (default: 3.0)")
    parser.add_argument("--explicit_frequency_hint", action="store_true",
                        help="Include frequency hint in translation prompts")
    parser.add_argument("--num_prompts", type=int, default=10000)
    parser.add_argument("--eval_prompts", type=int, default=1000)
    parser.add_argument("--prompt_length", type=int, default=8)
    # Generation
    parser.add_argument("--max_completion_length", type=int, default=None,
                        help="Max tokens to generate. Auto-set per environment if omitted.")
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (default 50, matches HF generate default; -1 to disable)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling (default 1.0 = disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty for generation (1.0=disabled)")
    parser.add_argument("--no_eos", action="store_true", help="Suppress EOS token to force full-length generations")
    # Training
    parser.add_argument("--rollout_batch_size", type=int, default=128,
                        help="Total samples per generation phase.")
    parser.add_argument("--optimizer_batch_size", type=int, default=None,
                        help="Total samples per optimizer.step(). Defaults to rollout_batch_size.")
    parser.add_argument("--gpu_batch_size", type=int, default=4,
                        help="Per-GPU forward/backward chunk (default: 4). Controls gradient accumulation.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--forget_lr_mult", type=float, default=1.0,
                        help="Multiplier on --lr for the forget-side parameter group. "
                             "When != 1.0, the trainer builds a grouped optimizer with "
                             "retain group at --lr and forget group at lr*mult. "
                             "IMPORTANT: the forget group uses weight_decay=0.0 (regardless "
                             "of --weight_decay); the retain group uses --weight_decay. "
                             "The optimizer class is taken from --optimizer (so fused AdamW, "
                             "SGD, etc. all work).")
    parser.add_argument("--warmstart_data", type=str, default=None,
                        help="Warm-start SFT data: a <env>.jsonl file or a directory containing "
                             "<environment>.jsonl. When set (GR runs only), runs a two-phase "
                             "supervised warm-start of the adapters before RL (see warmstart.py).")
    parser.add_argument("--warmstart_epochs", type=int, default=3, help="Epochs per warm-start phase.")
    parser.add_argument("--warmstart_batch_size", type=int, default=16, help="Warm-start SFT batch size.")
    parser.add_argument("--warmstart_val_frac", type=float, default=0.1,
                        help="Held-out fraction (per class) for warm-start CE validation logging.")
    parser.add_argument("--warmstart_n_train", type=int, default=None,
                        help="Cap warm-start training samples per phase to this many (low-data study). "
                             "None = use all. Val set unaffected (stays a large overfitting gauge).")
    parser.add_argument("--warmstart_n_train_forget", type=int, default=None,
                        help="Forget-phase override for --warmstart_n_train (None = use --warmstart_n_train). "
                             "Lets the forget-adapter warm-start dosage be tuned independently of retain.")
    parser.add_argument("--warmstart_epochs_forget", type=int, default=None,
                        help="Forget-phase override for --warmstart_epochs (None = use --warmstart_epochs).")
    parser.add_argument("--warmstart_lr", type=float, default=None,
                        help="LR for warm-start SFT; defaults to --lr when unset.")
    parser.add_argument("--warmstart_only", action="store_true",
                        help="Run warm-start, log val CE, then exit before RL (validation aid).")
    parser.add_argument("--beta", type=float, default=0.0, help="KL penalty coefficient against reference model (0=disabled). Off by default; only small-scale envs use 0.05.")
    parser.add_argument("--retain_kl_coef", type=float, default=0.0,
                        help="Extra KL(pi_RETAIN || frozen base) on existing target rollouts, outside monitor routing (0=off). Single-process adapter GRAFT only; independent of --beta.")
    parser.add_argument("--retain_kl_chunk_size", type=int, default=64,
                        help="Completion positions per full-vocabulary RETAIN-KL projection (memory limit, not subsampling).")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO lower clip (epsilon_low). TRL default 0.2.")
    parser.add_argument("--epsilon_high", type=float, default=None, help="PPO upper clip (DAPO Clip-Higher). Defaults to --epsilon (symmetric) if unset; DAPO uses 0.28.")
    parser.add_argument("--loss_type", type=str, default="grpo", help="TRL GRPO loss aggregation: 'grpo' (per-seq, our pinned default) or 'dapo' (token-level).")
    parser.add_argument("--scale_rewards", type=str, default="group", help="Advantage normalization: 'group' (GRPO std-norm, default) or 'none' (DAPO, no std div).")
    parser.add_argument("--top_entropy_quantile", type=float, default=1.0, help="Train only on the top-quantile highest-entropy tokens (DAPO-family anti-entropy-collapse). 1.0=all (default); 0.2=top 20%%.")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        help="LR scheduler type (linear, cosine, constant)")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Adam optimizer epsilon (denominator floor). Default 1e-8 matches HF/TRL. "
                             "Raising (e.g. 1e-5 to 1e-4) caps max per-parameter update magnitude when "
                             "squared-gradient EMA is tiny — useful as a stability regularizer in short runs.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_adapter_only", action=argparse.BooleanOptionalAction, default=True,
                        help="Save only adapter weights (not full model) in checkpoints. "
                             "Much smaller on disk. Requires base model at eval time.")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", default=None,
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--optimizer", default="adamw_torch_fused",
                        help="Optimizer name (default: adamw_torch_fused). See transformers OptimizerNames for options (e.g. sgd, adafactor).")
    parser.add_argument("--config_check", action="store_true",
                        help="Run full config pipeline, print effective values, and exit without training.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision (default: fp32)")
    parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision (default: fp32)")
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() in ("true", "1", "yes"), default=False,
                        help="Enable gradient checkpointing (default: False)")
    parser.add_argument("--use_liger_kernel", action=argparse.BooleanOptionalAction, default=False,
                        help="Use Liger fused linear GRPO loss (avoids materializing logits, default: off)")
    parser.add_argument("--liger_chunk_size", type=int, default=64,
                        help="Chunk size for LigerFusedLinearGRPOLoss (default 1 = one sample per chunk; "
                             "larger values trade memory for fewer kernel launches)")
    parser.add_argument("--torch_compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable torch.compile for the model (default: on)")
    # torch.compile with max-autotune-no-cudagraphs gives ~10-15% speedup over liger kernels
    # via Triton kernel fusion + autotuning. Adds ~60-90s startup time for compilation.
    # CUDA graphs (reduce-overhead, max-autotune) are blocked by .item() calls in HF's
    # flash attention varlen path, causing graph breaks and OOM from recording too many
    # graph partitions. For faster startup at slight perf cost: --torch_compile_mode default
    # or --no-torch_compile --use_liger_kernel.
    parser.add_argument("--torch_compile_mode", default="max-autotune-no-cudagraphs",
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        help="torch.compile mode (default: max-autotune-no-cudagraphs)")
    parser.add_argument("--max_tokens_per_microbatch", type=int, default=None,
                        help="Max tokens per microbatch for dynamic token batching. "
                             "When set, microbatches are packed by token count and trimmed to local max length.")
    parser.add_argument("--fused_reduction", action=argparse.BooleanOptionalAction, default=True,
                        help="Fuse the per-class (coherence/good/bad) forward+backward passes of "
                             "the dynamic-microbatching GR path into shared token-budget microbatches "
                             "with per-sample gradient routing (per-token forward forget-scale + "
                             "parameter-gradient gates). Exactly equivalent under loss_type grpo and dapo. ON "
                             "by default (the homogeneous path always spends a separate microbatch "
                             "per class, which dominates at realistic small optimizer batches); pass "
                             "--no-fused_reduction for the legacy homogeneous-microbatch path.")
    parser.add_argument("--compile_update", action=argparse.BooleanOptionalAction, default=False,
                        help="torch.compile(dynamic=True) the transformer trunk used by the packed "
                             "update path (and ref-logps). Off by default pending production mileage.")
    parser.add_argument("--ref_max_tokens_per_microbatch", type=int, default=None,
                        help="Token budget for ref-logprob dynamic microbatching. "
                             "Ref runs under no_grad so peak memory is dominated by the "
                             "logits softmax rather than saved activations, allowing much "
                             "larger bins than the training forward-backward budget. "
                             "If omitted, defaults to 4 * --max_tokens_per_microbatch when "
                             "that flag is set, otherwise falls back to uniform chunking.")
    parser.add_argument("--old_logps_max_tokens_per_microbatch", type=int, default=100_000,
                        help="Token budget for old-logprob dynamic microbatching. "
                             "Old_logps is a separate no-grad forward on the actor model "
                             "(distinct from ref_logps; runs when IS correction is on or "
                             "grad_accum spans multiple rollouts). Default 100k chosen from "
                             "bench_training_step.py --bench_what logps on Qwen3-4B/512 samples: "
                             "sweet spot is ~50k at ~15%% peak mem; 100k is a conservative cushion "
                             "that still beats the old fixed-batch path on both time and memory. "
                             "Only used by the bin+pad fallback; ignored when the packed path is on.")
    parser.add_argument("--no_packed_old_logps", dest="old_logps_packed_forward",
                        action="store_false", default=True,
                        help="Disable the true-sequence-packed old_logps forward (the DEFAULT), "
                             "falling back to the bin+pad _ref_logps_dynamic path. The packed path "
                             "reuses the update's flash_attn_varlen forward so old/update logps share "
                             "the kernel (clean IS ratio) and skips padding (~5x faster old_logps). "
                             "Only disable to A/B or if a non-liger/vision config needs the fallback.")
    parser.add_argument("--old_logps_validate_packed", action="store_true", default=False,
                        help="Validation: on every old_logps call also run the bin+pad path and "
                             "assert the packed result matches (<5e-2 max per-token diff), printing "
                             "the diff. Use on a short run to confirm the kernels agree, then drop.")
    parser.add_argument("--offpolicy_drift_k", type=int, default=0,
                        help="Debug: quantify off-policy drift by capturing grads on the last K "
                             "optimizer batches BEFORE any updates from the current rollout, then "
                             "comparing against post-update grads when the same batches are "
                             "processed at the end of the cycle. 0 disables. Dynamic-batching path only.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", default="small-rl")
    parser.add_argument("--run_name", default=None, help="Override wandb run name")
    parser.add_argument("--wandb_group", default=None,
                        help="wandb run group; sweep.py sets this to the sweep name.")
    parser.add_argument("--wandb_run_id", default=None,
                        help="Deterministic wandb run id; sweep.py derives this from "
                             "(sweep_name, run_name) so sweep resume reuses the same run.")
    parser.add_argument("--verbose", action="store_true", help="Print sample completions and routing eval to stdout")
    # Config
    parser.add_argument("--config", default=None,
                        help="YAML config (reward, rh_detector, and optional training section)")
    # GPU / DDP
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of GPUs for DDP (default: 1, no DDP). "
                             "Set automatically by sweep.py when gpus_per_run > 1.")
    # Gradient routing
    parser.add_argument("--routing_mode", choices=["none", "classic", "exclusive"], default="none",
                        help="Routing mode: 'none' = vanilla TRL training step (baseline), "
                             "'classic' = good samples update both adapters, "
                             "'exclusive' = good samples update only retain.")
    parser.add_argument("--retain_rank", type=int, default=32)
    parser.add_argument("--forget_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_config", default=None, choices=list(LORA_PRESETS.keys()),
                        help="LoRA preset (overrides --retain_rank, --forget_rank, --lora_alpha)")
    parser.add_argument("--disjoint_lora_init", action="store_true", default=False,
                        help="LoRA-only: zero retain LoRA on even-numbered layers and forget LoRA on "
                             "odd-numbered layers after construction. Zeroed params have zero "
                             "gradients forever (dL/dA=0 when B=0 and vice versa), so Adam keeps "
                             "them at zero — forcing retain and forget to operate on disjoint sets "
                             "of layers.")
    parser.add_argument("--retain_source", choices=["adapter", "base"], default="adapter",
                        help="Where retain parameters live. 'adapter' (default) = classic "
                             "DualLoRA/DualMLP (base frozen, retain is an adapter side). "
                             "'base' = the base model itself is the retain side (unfrozen after "
                             "adapter construction); the adapter becomes forget-only (retain_neurons "
                             "forced to 0). Requires adapter_type in {mlp,none} and vllm_colocate.")
    # Adapter type selection
    parser.add_argument("--adapter_type", choices=["lora", "mlp", "none"], default="lora",
                        help="Adapter type for gradient routing (default: lora). 'none' = full-param training, no adapters.")
    parser.add_argument("--mlp_config", default=None, choices=list(MLP_PRESETS.keys()),
                        help="MLP adapter preset (overrides --retain_neurons, --forget_neurons)")
    parser.add_argument("--retain_neurons", type=int, default=32)
    parser.add_argument("--forget_neurons", type=int, default=32)
    parser.add_argument("--layer_start", type=float, default=0.0,
                        help="Start of adapter layer range as fraction of total (0.0 = first layer)")
    parser.add_argument("--layer_end", type=float, default=1.0,
                        help="End of adapter layer range as fraction of total (1.0 = through last layer)")
    # Routing eval
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Routing eval interval in steps (0 to disable)")
    parser.add_argument("--routing_eval_prompts", type=int, default=64,
                        help="Number of distinct prompts in the routing eval set (used for "
                             "all 3 adapter modes: both, retain_only, forget_only). Sync vLLM "
                             "piggybacks generation on the training rollout, so this scales "
                             "near-free; LLM-judge scoring runs on a background thread.")
    parser.add_argument("--eval_at_start", action=argparse.BooleanOptionalAction, default=True,
                        help="Run routing eval before training starts (default: on)")
    # Diagnostic channels. Two channels,
    # each with an interval in {every_iter, when_eval, off}; when_eval == every
    # --eval_every steps.
    parser.add_argument("--adapter_diag_interval", choices=["every_iter", "when_eval", "off"],
                        default="when_eval",
                        help="Cadence for the adapter-diagnostics channel (retain/forget "
                             "adapter grad/param/optimizer norms, and optionally the "
                             "per-sample per-layer grad+activation recompute). "
                             "when_eval = every --eval_every steps.")
    parser.add_argument("--adapter_diag_level", choices=["adapter_diagnostics", "per_sample_recompute"],
                        default="adapter_diagnostics",
                        help="adapter_diagnostics = cheap retain/forget grad/param/optimizer "
                             "norms (no extra fwd/bwd). per_sample_recompute additionally runs "
                             "one extra unmasked packed forward/backward over the rollout batch, "
                             "capturing per-sample per-layer grad norms for the 2x2 "
                             "(retain/forget params x retain/forget samples) -> grad_diag.jsonl. "
                             "per_sample_recompute includes adapter_diagnostics; "
                             "routing-rollout steps only for the recompute.")
    # Stochastic routing
    parser.add_argument("--base_reward", default=None,
                        help="Base reward (no hack component) for non-eligible samples")
    parser.add_argument("--rh_eligible_frac", type=float, default=1.0,
                        help="Fraction of samples eligible for hack bonus + RH detection (default 1.0 = all). "
                             "Prefer --hack_frac for new experiments (prompt-level, feature-based).")
    parser.add_argument("--eval_hack_frac", type=float, default=None,
                        help="If set, routing-eval prompts are drawn at this hack_frac "
                             "instead of --hack_frac (training data unaffected). For "
                             "unhackable-only training arms whose eval must stay on the "
                             "standard mixed distribution.")
    parser.add_argument("--hack_frac", type=float, default=1.0,
                        help="Fraction of prompts where the hack is available (default 1.0 = all). "
                             "Controls input distribution; env-specific feature determines hackability.")
    parser.add_argument("--prompt_condition", type=int, choices=[1, 2], default=None,
                        help="Quadrant-design monitorability axis (A/B = hackability x 1/2 = "
                             "condition). Restricts the TRAINING distribution to one condition; "
                             "env-specific axis: addition/object/cities = question form "
                             "(1=open, 2=tf); persona = topic (1=color, 2=category); repeat = "
                             "template (1=one[+echo/copy], 2=many[+write]); sorting = length "
                             "(1=n<=7, 2=n>7). Routing-eval prompts always revert to the full "
                             "distribution (condition override nulled).")
    parser.add_argument("--force_unhackable", action="store_true",
                        help="Force hackable=False on ALL training rows after load (zeroes "
                             "forget-role rewards via CombinedReward). For anchor-only arms "
                             "training on a hackable-type distribution (e.g. B2-only) without "
                             "a live hack incentive. Eval rows are unaffected.")
    parser.add_argument("--kodcode_nonhackable_style", choices=["locked", "solve_only"],
                        default="locked",
                        help="kodcode: how to render NON-hackable prompts (hack_frac<1.0) so the "
                             "model can distinguish the regime. 'locked' = two-file format + a "
                             "read-only-test cue (edits discarded, graded vs original tests); "
                             "'solve_only' = single-file format, no test.py shown or requested "
                             "(hack surface physically absent). Inert at hack_frac=1.0.")
    parser.add_argument("--kodcode_hack_prompt_variant",
                        choices=["default", "salient", "countdown", "prune_hint",
                                 "single_test"],
                        default="default",
                        help="kodcode: instructional preamble for HACKABLE prompts (base-rate "
                             "experiment; see envs/kodcode._HACK_PREAMBLES). 'prune_hint' (base "
                             "hack 4.74%%: prune 3.44%% + overwrite 1.29%%, vs default 2.03%%) makes "
                             "the <100%% threshold + test.py control overt to lift the unmonitored "
                             "test_prune form. 'single_test' exposes only ONE test + drops the "
                             "pass-threshold line -> solution-side answer-hardcoding. Only affects "
                             "hackable prompts.")
    parser.add_argument("--kodcode_id_filter", default=None,
                        help="kodcode: path to a JSON list of problem ids to RESTRICT the env to "
                             "(difficulty filter). Resolved relative to the repo root if not "
                             "absolute. Applied before the train/eval split, so both splits draw "
                             "from the allowlist. Fails loud if <50 ids match.")
    parser.add_argument("--countdown_n_train", type=int, default=8000,
                        help="countdown_code: number of RL training prompts (held-out chunk, "
                             "disjoint from the SFT distillation set).")
    parser.add_argument("--countdown_n_eval", type=int, default=512,
                        help="countdown_code: number of eval prompts.")
    parser.add_argument("--countdown_train_system_suffix", default=None,
                        help="countdown_code: sentence appended to the system message of TRAIN "
                             "prompts only (inoculation prompting; eval prompts stay neutral, "
                             "which is the removed-at-test-time half of the intervention).")
    # Positive Preventative Steering (PPS): activation-space analogue of
    # inoculation prompting. Steers the TRAINING policy in both stacks (vLLM
    # training eid + in-process HF update forwards); eval eids / deployment
    # are never steered. All three default OFF (byte-identical to no-PPS).
    parser.add_argument("--pps_vector_path", default=None,
                        help="PPS: path to a torch.save'd dict {'raw': {layer: vec}, "
                             "'incremental': {layer: vec}, 'meta': ...} of fp32 "
                             "steering vectors (hidden_size,) per decoder layer.")
    parser.add_argument("--pps_layer", default=None,
                        help="PPS layer selection: an int layer index (single-layer; "
                             "uses raw[L]) or 'all' (all-layer; uses the full "
                             "incremental dict).")
    parser.add_argument("--pps_alpha", type=float, default=0.0,
                        help="PPS steering strength (residual-stream units); "
                             "0.0 = off (default).")
    parser.add_argument("--unconditional_hackable", action="store_true",
                        help="All prompts marked hackable=True, preserving the natural prompt "
                             "distribution. Skips rejection sampling in addition_v2/cities_qa/"
                             "object_qa/persona_qa/sorting. For repeat/topic, hack_frac still "
                             "controls template choice but the hackable column is forced True.")
    parser.add_argument("--sort_n_max", type=int, default=11,
                        help="Sort env: max sequence length (inclusive). Default 11 (sequences "
                             "of length 4-11). Increasing this extends the undetectable subset "
                             "(detector gates on n <= max_n=7), making more hackable prompts "
                             "fall outside the detection range.")
    parser.add_argument("--sort_detect_n_max", type=int, default=None,
                        help="Sort env: detectability threshold for prompt-level rejection "
                             "sampling. Should match the rh_detector's max_n. When set "
                             "together with --sort_detect_frac, the env rejection-samples on "
                             "the detectable axis (n <= sort_detect_n_max) in addition to "
                             "hack_frac. Disabled by default (no detect-axis rejection).")
    parser.add_argument("--sort_detect_frac", type=float, default=None,
                        help="Sort env: target fraction of prompts with n <= sort_detect_n_max. "
                             "Combines with hack_frac into 4 buckets at the implied joint "
                             "targets. Disabled by default.")
    parser.add_argument("--sort_uniform_per_length", action=argparse.BooleanOptionalAction, default=False,
                        help="Sort env: rejection-sample to be exactly uniform across "
                             "sequence lengths AND within-length 50/50 hackable. "
                             "Overrides hack_frac and sort_detect_frac. Detectable rate "
                             "is determined by sort_detect_n_max / sort_n_max (e.g. "
                             "n_max=15, detect_n_max=7 -> 1/3 detectable).")
    parser.add_argument("--sort_natural_hackable", action=argparse.BooleanOptionalAction, default=False,
                        help="Sort env: disable rejection sampling on the hackable axis — the "
                             "hackable rate falls out of the data (P(first == max), ~20% overall, "
                             "higher for short lists). hack_frac is ignored. Composes with "
                             "sort_uniform_per_length (buckets by length only).")
    parser.add_argument("--sort_hackable_n_min", type=int, default=6,
                        help="sorting_v2 only: length-gated hackable rule — hackable when the "
                             "list has >= this many elements (easy tier below it). Ignored by "
                             "the v1 'sorting' env (max-first-gated).")
    parser.add_argument("--sort_val_max", type=int, default=9,
                        help="Sort env: list values are sampled uniformly from [0, sort_val_max]. "
                             "Default 9 (digits). Larger alphabets (e.g. 99) collapse the "
                             "input-independent prior credit of sorted lists; raise "
                             "max_completion_length accordingly for multi-digit values.")
    parser.add_argument("--sort_deranged", action="store_true",
                        help="Sort env: re-permute each input list so it shares NO position "
                             "with its sorted form, making a verbatim copy score exactly 0.0 "
                             "retain (copy and sort become exactly mutually exclusive).")
    parser.add_argument("--sort_prompt_style", default="plain",
                        help="Sort env: prompt template pair keyed by hackability "
                             "(envs.sorting._PROMPT_STYLES is the source of truth). The number "
                             "list renders identically under every style, so the response-side "
                             "monitor is unaffected. configs/sorting_copy_framed*.yaml set "
                             "framing_noverb in their training section (sorting_framed rejects "
                             "plain).")
    parser.add_argument("--sort_curriculum_end_step", type=int, default=0,
                        help="Sort env: length curriculum. Linearly anneal a per-rollout list-length "
                             "cap from the dataset's min length to its max over the first N optimizer "
                             "steps; routing prompts longer than the cap are swapped for in-cap "
                             "prompts at rollout time (eval prompts untouched). 0 = off.")
    parser.add_argument("--mbpp_detect_mode", default="none",
                        choices=["none", "type", "monitored"],
                        help="MBPP env: how the `detectable` column (monitoring scope of the "
                             "hardcode hack) is assigned. 'none' (default): detectable == "
                             "hackable (every available hack is monitored). 'type': intrinsic "
                             "— detectable iff the visible expected value is a distinctive "
                             "literal (not int/bool); the ~50%% int/bool stratum is the "
                             "undetectable hack, conditionable from the prompt. 'monitored': "
                             "append an explicit 'will (not) be monitored' sentence per "
                             "--mbpp_detect_frac md5 partition; detectable == monitored.")
    parser.add_argument("--mbpp_detect_frac", type=float, default=0.5,
                        help="MBPP env: target fraction of hackable prompts marked detectable "
                             "(monitored) when --mbpp_detect_mode=monitored. md5 partition on "
                             "task_id, stable per run. Ignored for other detect modes.")
    parser.add_argument("--mbpp_prompt_mode", default="plain",
                        choices=["plain", "aware", "compelling", "instruct", "system"],
                        help="MBPP env: raise the natural hardcode rate by making the "
                             "weak-evaluator affordance salient (capable models ~never "
                             "hardcode under the plain prompt). plain=no hint; aware=soft "
                             "reframe toward the visible test; compelling=explicit license to "
                             "return the shown value; system=system-message reframe.")
    parser.add_argument("--mbpp_value_types", default="",
                        help="MBPP env: comma-separated expected-value type names to restrict "
                             "the task set to (e.g. 'int' or 'int,str'). Empty = no filter. "
                             "Use 'int' to drop bool/list/... tasks where a wrong solution "
                             "coincidentally passes the single visible assert (~50%% on bools), "
                             "confounding the hack signal; int expected values are much harder "
                             "to false-positive on.")
    parser.add_argument("--cities_invert_hackable", action=argparse.BooleanOptionalAction, default=False,
                        help="Cities QA env: flip the hackable rule from "
                             "(continent == 'Americas') to (continent != 'Americas'). "
                             "Inverts which side of the continent partition is hack-eligible.")
    parser.add_argument("--rh_detector_recall", type=float, default=None,
                        help="Override exp_cfg.rh_detector_recall (fraction of true positives flagged, default 1.0)")
    parser.add_argument("--detect_unhackable", action=argparse.BooleanOptionalAction, default=True,
                        help="Run the RH detector on unhackable samples too (default True). "
                             "Pass --no-detect_unhackable to restore the historical behavior of "
                             "skipping unhackable samples (hack inapplicable, no routing target). "
                             "Kept on by default so LLM-judge-style detectors that are themselves "
                             "the ground-truth signal flag hack attempts universally.")
    # Coherence training
    parser.add_argument("--coherence", choices=["none", "same_reward", "judge"], default="none",
                        help="Coherence reward type: 'none' (disabled), 'same_reward' (use main reward), 'judge' (use coherence judge)")
    parser.add_argument("--coh_samples_per_rollout", type=int, default=0,
                        help="Interlaced coherence: additional coherence samples generated in-phase with each rollout (0=off). Must be a multiple of num_generations and optimizer_batch_size.")
    parser.add_argument("--coh_config", default=None,
                        help="Cross-env coherence anchor: experiment-config YAML whose env supplies "
                             "the coherence-slot prompts and whose reward scores them (the main "
                             "rh_detector never sees the slice; its is_rh is False by construction). "
                             "Requires coh_samples_per_rollout > 0 and --coherence same_reward. "
                             "None = coherence prompts/rewards come from the main env (default).")
    parser.add_argument("--coherence_start_frac", type=float, default=0.0,
                        help="Delay interlaced coherence until this fraction of max_steps has elapsed "
                             "(0.0 = always on, the default/current behavior; e.g. 0.7 = coherence "
                             "begins at 70%% of the run). Before the threshold the effective "
                             "coh_samples_per_rollout is 0 (a clean non-coherence rollout). Hypothesis: "
                             "coherence training is what lets the hack be learned, so deferring it limits "
                             "hack acquisition to the run's tail.")
    parser.add_argument("--coherence_rh_mode", choices=["filter", "filter_renorm", "penalty", "zero", "none"], default="filter",
                        help="How to handle detected hacks during coherence rollout: "
                             "'none' (classifier not consulted on coherence samples at all: coherence "
                             "groups keep the stock full-group GRPO advantage, hacks included), "
                             "'filter' (zero advantages — sample contributes no gradient; per-group "
                             "mean/std still include the zeroed hacks), "
                             "'filter_renorm' (skyline: drop hacks from each coherence group, recompute "
                             "per-group mean/std over non-hacks only; hack advantages are 0 and all-hack "
                             "groups are all-zero — paired with a perfect detector, this upper-bounds "
                             "how clean the retain signal can be), "
                             "'penalty' (subtract coherence_rh_penalty from rewards, recompute advantages), "
                             "'zero' (set RH rewards to 0.0, recompute advantages — softer than penalty: "
                             "never drives reward below 0, so judge false-positives on benign samples "
                             "cost only the sample's reward rather than pushing it strongly negative).")
    parser.add_argument("--coherence_rh_penalty", type=float, default=3.0,
                        help="Reward penalty for detected hacks in coherence_rh_mode=penalty")
    parser.add_argument("--coh_prompt_source",
                        choices=["batch", "unhackable", "cond2_hackable", "cond2_unhackable"],
                        default="batch",
                        help="Where interlaced-coherence prompts come from. 'batch': the trailing K "
                             "prompts of the ordinary training batch (default, historical behavior). "
                             "'unhackable': a separate pool drawn from the env's unhackable "
                             "distribution (env loader re-run with hack_frac=0.0) — models coherence "
                             "passes on a 'verifiable' distribution where hacks are structurally "
                             "impossible. Mutually exclusive with "
                             "--rh_detector_verifies_retain_samples.")
    parser.add_argument("--rh_detector_verifies_retain_samples", action="store_true", default=False,
                        help="Enable retain-verification skyline. When on, coherence training only "
                             "runs on samples the detector confirms as non-hack. Requires the detector "
                             "to implement classifiable() (see rh_detectors.RH_CLASSIFIABLE_REGISTRY). "
                             "Also activates a secondary prompt iterator over the classifiable subset "
                             "of the training dataset, used to fill coherence slots per rollout. "
                             "Interlaced coherence only.")
    parser.add_argument("--verified_only_training", action=argparse.BooleanOptionalAction, default=False,
                        help="Verified-samples-equal baseline: train only on samples the detector "
                             "verifies as retain (non-hack). Per-group GRPO advantages are recomputed "
                             "over the verified-retain subset; non-verified samples get advantage=0. "
                             "Eval still runs on the full env distribution. Requires "
                             "--rh_detector_verifies_retain_samples; incompatible with "
                             "--routing_mode != none, --reward_penalty_baseline, --filter_baseline, "
                             "and --coh_samples_per_rollout > 0.")
    parser.add_argument("--rh_detector_retain_recall", type=float, default=1.0,
                        help="Recall of the retain VERIFIER (not the rh_detector). Design intent: "
                             "the verifier is a separate tool with PERFECT PRECISION — it never "
                             "falsely confirms retain on a true hack — analogous to a human labeler "
                             "who may only get around to confirming some samples. Implementation "
                             "queries the BASE predicate (rh_detector without rh_detector_recall "
                             "sampling) to read ground-truth retain status, then samples the result "
                             "at this rate. <1.0 simulates partial labeling coverage. Default 1.0 "
                             "(verifier confirms every true non-hack on a detectable prompt). "
                             "Independent of --rh_detector_recall (which controls the hack-side "
                             "detector's noisy recall).")
    parser.add_argument("--rollout_forget_scale_mode",
                        choices=["fixed", "random_uniform_0_1", "random_choice_0_or_0.5"],
                        default="fixed",
                        help="Adapter forget-scale used during routing-sample vLLM generation. "
                             "'fixed' (default): scale=1.0 (both adapters at full scale, current "
                             "behavior). 'random_uniform_0_1': resample U(0,1) once per rollout, "
                             "apply to the rollout's routing samples (coh slots are unaffected — "
                             "they always use scale (1, 0)). "
                             "'random_choice_0_or_0.5': pick uniformly from {0.0, 0.5} per rollout. "
                             "Affects only generation; training-time forward+backward unaffected.")
    parser.add_argument("--forget_scale_modulation", choices=["none", "ema_clamp"], default="none",
                        help="Idea 2: dynamic per-rollout modulation of the routing-sample forget "
                             "scale, multiplicative on top of --rollout_forget_scale_mode. "
                             "'none' (default): scale held at base mode value. "
                             "'ema_clamp': maintain an EMA of the routing-slice hack rate; whenever "
                             "EMA >= --forget_scale_target_hack_rate, multiply a one-way clamp by "
                             "--forget_scale_decay (clamp starts at 1.0, monotone non-increasing). "
                             "Each rollout's effective forget_scale = base_mode_sample * clamp.")
    parser.add_argument("--forget_scale_target_hack_rate", type=float, default=0.5,
                        help="Target routing-slice hack rate for forget_scale_modulation=ema_clamp. "
                             "When the EMA exceeds this value, the clamp decays. Default 0.5.")
    parser.add_argument("--forget_scale_ema_weight", type=float, default=0.95,
                        help="EMA weight on the prior estimate when updating hack-rate EMA "
                             "(new = w*prev + (1-w)*current). Default 0.95 (slow update).")
    parser.add_argument("--forget_scale_decay", type=float, default=0.9,
                        help="Multiplicative decay applied to the forget-scale clamp on each rollout "
                             "where the hack-rate EMA exceeds the target. Default 0.9.")
    parser.add_argument("--forget_scale_min_clamp", type=float, default=0.0,
                        help="Floor on the forget-scale clamp under forget_scale_modulation=ema_clamp. "
                             "Default 0.0 (clamp can collapse to 0). Setting >0 (e.g. 0.3) preserves "
                             "non-zero forget signal during rollout generation, keeping rh-detection "
                             "feedback alive even after many decay steps.")
    parser.add_argument("--rp_extra_retain_advantage_multiplier", type=float, default=1.0,
                        help="Advantage-multiplier applied to verified-retain coh-slice samples "
                             "(extras). Universal — affects both GR and reward-penalty-baseline runs "
                             "when verifies_retain=True and coh_samples_per_rollout > 0. "
                             "Default 1.0 (no effect). Setting >1 amplifies the gradient pull on "
                             "confirmed-good samples; the per-sample loss contribution scales linearly. "
                             "Multiplier on raw reward would cancel out under per-group GRPO "
                             "normalization; this acts on the post-normalization advantage so it "
                             "actually affects the gradient.")
    parser.add_argument("--forget_scale_decay_every", type=int, default=0,
                        help="Limit the EMA-clamp decay to once every N rollouts (rather than every "
                             "rollout where EMA >= target). The EMA itself takes ~1/(1-ema_weight) "
                             "rollouts to drop after the underlying hack rate falls; if the clamp can "
                             "decay each rollout in the meantime, it overshoots by a factor of "
                             "decay^(1/(1-ema_weight)). Default 0 = auto-derive as "
                             "round(1 / (1 - ema_weight)) (= 20 with default ema_weight=0.95).")
    parser.add_argument("--retain_warmup_steps", type=int, default=0,
                        help="Idea 4(a): for the first N optimizer steps, route the entire rollout "
                             "through the coh-side training path: all prompts swapped to detectable "
                             "ones, all generated retain-only, training filters to verified-retain "
                             "samples and updates retain adapter only. Forget adapter does not move. "
                             "Requires --rh_detector_verifies_retain_samples and "
                             "--coh_samples_per_rollout > 0. Default 0 (off).")
    parser.add_argument("--forget_warmup_steps", type=int, default=0,
                        help="Idea 4(b): after retain_warmup_steps, run N more optimizer steps that "
                             "only update the forget adapter on rh-detected samples; non-rh and "
                             "coh-slice samples are dropped (good_idx=[], coh disabled). Phase "
                             "ordering: 0..retain_warmup_steps = retain warmup, "
                             "retain_warmup_steps..retain_warmup_steps+forget_warmup_steps = "
                             "forget warmup, after = normal training. Default 0 (off).")
    parser.add_argument("--routing_trace_interval", choices=["every_iter", "when_eval", "off"],
                        default="when_eval",
                        help="Cadence for the routing/training trace channel. Writes a "
                             "per-rollout summary + per-sample records (a random subset of the "
                             "rollout, with completion text inline) to "
                             "{output_dir}/routing_trace.jsonl, capturing the routing label "
                             "(is_rh), ground-truth hack emission, pre/post-renorm advantages, "
                             "completion lengths, and component scores. "
                             "when_eval = every --eval_every steps.")
    parser.add_argument("--routing_trace_samples", type=int, default=16,
                        help="Number of training samples (random subset of the rollout) to "
                             "record per routing-trace fire. Only these are decoded, so the "
                             "cost is independent of rollout batch size.")
    parser.add_argument("--eval_full_completions", action="store_true",
                        help="Write EVERY eval sample's full (untruncated) prompt+completion "
                             "to eval_samples.jsonl (default: only 8 rows, capped at 400 chars). "
                             "Needed for post-hoc analysis of envs whose completions exceed 400 "
                             "chars (e.g. countdown_code's two-file JSON).")
    # Retain advantage correction
    parser.add_argument("--renormalization_mode", choices=["off", "retain-only", "balanced"],
                        default="retain-only",
                        help="How the retain adapter's advantage is normalized (GR runs only). "
                             "'off': both adapters share the stock full-group GRPO advantage. "
                             "'retain-only' (default): good-routing samples get a per-group GRPO "
                             "normalization over the non-RH samples, so the retain adapter sees a "
                             "DIFFERENT advantage than the forget adapter (historical default). "
                             "'balanced': one advantage vector shared by both adapters with a clean "
                             "(non-flagged) baseline + full-group variance and forget-side "
                             "redistribution (double bad-sample advantage). Classic, non-coherence only.")
    parser.add_argument("--split_moment", action=argparse.BooleanOptionalAction, default=False,
                        help="Split-moment AdamW: build Adam's second moment (v) from the "
                             "pre-routing (natural) gradient and the first moment (m) from the "
                             "routed gradient. Both come from one backward (a capture reconstructs "
                             "the natural adapter gradient). Requires renormalization_mode='balanced' "
                             "(classic GR, fused/liger path, single process); LoRA or MLP adapters.")
    parser.add_argument("--routing_lambda", type=float, default=1.0,
                        help="graft-port: soft-routing knob for balanced redistribution. lambda=1 "
                             "(default) = clean routing (detected: retain masked / forget "
                             "xkappa), single-backward fast path. lambda<1 (soft) and lambda>1 "
                             "(over-route) take the 2-backward v=a_v slow path (require --split_moment); "
                             "lambda>1 adds the per-group over-routing cap + B1 v-floor + realized-step "
                             "gate.")
    parser.add_argument("--graft_w_max", type=float, default=4.0,
                        help="graft-port: ceiling on the absorbing redistribution weight = max "
                             "per-coordinate retain/forget Adam-step multiplier under routing. Fails "
                             "loud at construction if adapter geometry (kappa) would exceed it; raise to "
                             "opt into a strongly-unequal adapter's kappa-x per-coordinate LR. Under λ>1 "
                             "it also caps the realized per-coordinate over-routing step (see "
                             "--graft_step_policy).")
    parser.add_argument("--graft_step_policy", type=str, default="clamp", choices=["clamp", "gate"],
                        help="graft-port λ>1 over-routing step control. 'clamp' (default): per-coordinate "
                             "trust region — floor the Adam denom so every coord's realized step is "
                             "<= graft_w_max (a no-op at the kappa operating point; only runaway "
                             "over-routing coords are bounded; never tune w_max for stability). 'gate': "
                             "fail loud if the realized step exceeds graft_w_max (over-budget configs "
                             "crash instead of being bounded). Diagnostics graft/{frac_coords_clamped,"
                             "realized_step_*,lam_eff_*} are logged either way.")
    parser.add_argument("--allow_approx_lora_kappa", action=argparse.BooleanOptionalAction, default=False,
                        help="RELAXATION (graft-port): permit LoRA + balanced-renorm routing by deriving "
                             "kappa from LoRA ranks (kappa_A=(r_R+r_F)/r_A), the same formula used for MLP "
                             "neuron counts. APPROXIMATE for LoRA: rank is not output-space pressure (LoRA's "
                             "forward is alpha/rank-scaled and bilinear, so true pressure is weight-dependent "
                             "and time-varying), so equal-pressure holds only approximately — exactly at the "
                             "symmetric alpha=rank, r_R=r_F, lambda=1 point and degrading with rank asymmetry "
                             "/ over-routing. Off by default; opt in explicitly for LoRA routing experiments.")
    parser.add_argument("--drop_zero_advantage", action=argparse.BooleanOptionalAction, default=False,
                        help="Compute optimization: drop samples with exactly zero advantage from the "
                             "microbatches (gradient-equivalent at beta==0; requires beta==0). Default off.")
    # Filter baseline
    parser.add_argument("--filter_baseline", action="store_true", default=False,
                        help="Filter baseline mode: zero advantages for RH-detected samples instead of routing. "
                             "Uses same rh_eligible_frac eligibility as routing runs.")
    # Reward penalty baseline
    parser.add_argument("--reward_penalty_baseline", action="store_true", default=False,
                        help="Reward penalty baseline: penalize rewards for RH-detected samples, recompute advantages.")
    parser.add_argument("--reward_penalty_amount", type=float, default=None,
                        help="Amount to subtract from post-clip reward for detected hacks. "
                             "Default: None (zeros reward, legacy behavior). E.g. 3.0 subtracts 3.0.")
    # Retain penalty baseline
    parser.add_argument("--retain_penalty_baseline", action="store_true", default=False,
                        help="Retain penalty baseline: replace RH rewards with retain-only reward, recompute advantages.")
    # vLLM generation server
    parser.add_argument("--vllm_server", default=None,
                        help="ZMQ socket address of vLLM server for generation "
                             "(e.g. ipc:///tmp/vllm_grpo.sock or tcp://127.0.0.1:5555). "
                             "When set, generation is offloaded to the server and adapter weights "
                             "are synced before each generation step.")
    parser.add_argument("--vllm_enforce_eager", action=argparse.BooleanOptionalAction, default=False,
                        help="vLLM engine eager mode. DEFAULT IS NOW COMPILED (False): the dynamo "
                             "specialization fix + dims-keyed compile cache made the compiled path "
                             "safe (token-exact gate, 200-step repeat x2 and 1000-step persona "
                             "validations match eager references). ~11x lower fixed decode cost at "
                             "135M. Pass --vllm_enforce_eager to fall back to eager.")
    parser.add_argument("--vllm_cudagraph_mode", default="FULL_AND_PIECEWISE",
                        help="CompilationConfig cudagraph mode for the compiled engine. Default "
                             "FULL_AND_PIECEWISE: captures whole decode steps (1.73x gen at width "
                             "544 vs the PIECEWISE default; token-exact gate-verified). Ignored "
                             "under --vllm_enforce_eager. Pass an empty string for vLLM's default.")
    parser.add_argument("--vllm_max_model_len", type=int, default=None,
                        help="Engine max sequence length (None = model default, 8k for SmolLM2). "
                             "Our sequences are ~110 tokens; a small cap (e.g. 512) slashes the "
                             "per-engine KV floor — required for high MPS concurrency at low "
                             "vllm_gpu_memory.")
    parser.add_argument("--vllm_kv_cache_gb", type=float, default=None,
                        help="Explicit KV-cache budget in GiB. Bypasses vLLM's differential "
                             "free-memory profiling (replaces vllm_gpu_memory for KV sizing), "
                             "making engine init robust to concurrent GPU activity. ~23KB/token "
                             "at 135M: 0.5 GiB holds ~2700 sequences of ~110 tokens.")
    parser.add_argument("--vllm_parallel_init", action=argparse.BooleanOptionalAction, default=None,
                        help="Skip the per-GPU engine-init serialization lock. Default (unset): "
                             "AUTO — enabled exactly when an explicit KV budget is in effect "
                             "(--vllm_kv_cache_gb or the model-conditional default), since the "
                             "profiling-based sizing the budget replaces is the reason the lock "
                             "exists. Pass --no-vllm_parallel_init to force serialized boots.")
    parser.add_argument("--vllm_spawn", action="store_true", default=False,
                        help="Spawn a local vLLM server for this run (one server per run). "
                             "Uses the run's own model/mlp_config. Mutually exclusive with --vllm_server.")
    parser.add_argument("--vllm_async", action="store_true", default=False,
                        help="Use AsyncVLLMClient (for shared async server). Requires --vllm_server.")
    parser.add_argument("--vllm_gpu_memory", type=float, default=0.02,
                        help="GPU memory utilization fraction for spawned vLLM server (default: 0.02).")
    parser.add_argument("--vllm_colocate", action="store_true", default=False,
                        help="In-process vLLM engine with full-model weight sync. "
                             "Mutually exclusive with --vllm_server/--vllm_spawn.")
    parser.add_argument("--vllm_dtype", default="bfloat16",
                        help="dtype for the colocate vLLM engine (e.g. bfloat16, float16). "
                             "Ignored for vllm_spawn (which consumes it at spawn time).")
    parser.add_argument("--vllm_spawn_delay", type=int, default=0,
                        help="DEPRECATED (2026-06-02): vLLM init serialization is now done via "
                             "a per-GPU flock in vllm_lifecycle.vllm_init_slot, called from "
                             "inside _spawn_vllm_server. This flag is silently ignored — kept "
                             "for backcompat with sweep configs that still set it.")
    parser.add_argument("--vllm_no_sleep", action="store_true", default=False,
                        help="Disable vLLM sleep/wake cycle between rollout and training phases. "
                             "Useful when multiple runs share a GPU (per_gpu > 1) and sleep/wake overhead dominates.")
    parser.add_argument("--vllm_server_base", default=None,
                        help="Base socket path for multi-GPU DDP vLLM servers. "
                             "Each DDP rank appends _rank{rank}.sock. Set by sweep.py.")
    parser.add_argument("--vllm_importance_sampling", action="store_true", default=False,
                        help="Force the SLOW IS path: run the HF actor forward for old_logps so the "
                             "vLLM-vs-HF policy ratio is EXPLICIT — applies TIS/MIS clipping AND logs "
                             "the sampling/* diagnostics (importance_sampling_ratio, sampling_logp_difference). "
                             "NOTE: this does NOT 'turn on' mismatch correction — the FAST IS path is ON BY "
                             "DEFAULT (uses vLLM sampling logprobs as old_logps, folding the vLLM-vs-HF "
                             "mismatch into the PPO ratio; ratio==1 by construction so no sampling/* stats). "
                             "Use this flag to MEASURE the gap / apply explicit clipping. --no_fast_vllm_is "
                             "disables the default fast correction too. Requires vLLM return_logprobs (MLP path).")
    parser.add_argument("--vllm_is_token_clip", type=float, default=2.0,
                        help="Symmetric per-token TIS clamp threshold c: ratio -> clamp(ratio, 1/c, c). "
                             "Pass 0 to disable the token-level clip. Default 2.0. "
                             "Requires --vllm_importance_sampling.")
    parser.add_argument("--vllm_is_seq_filter", type=float, default=1.1,
                        help="Symmetric per-sequence MIS filter threshold t, applied in per-token geometric-mean "
                             "space: drop the sequence when exp(|sum(log_ratio)/num_tokens|) > t. "
                             "Pass 0 to disable the sequence filter. Default 1.1. "
                             "Requires --vllm_importance_sampling.")
    parser.add_argument("--no_fast_vllm_is", action="store_true", default=False,
                        help="Disable fast vLLM IS correction (use vLLM sampling logprobs as old_logps "
                             "instead of a full forward pass). Enabled by default when using vLLM.")
    # Batch capture for profiling
    parser.add_argument("--save_batch", default=None,
                        help="Path to save the first generation batch dict (.pt) for offline profiling")
    return parser
