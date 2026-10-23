"""Measure exact RETAIN KL with identical fixed weights and 512 sorting-shaped rows.

This is an auxiliary-loss microbenchmark, not a full training-step estimate.
It does not generate rollouts, update weights, or alter experiment settings.
The BF16 baseline sums many small parameter gradients sequentially; report
norm-relative error and cosine similarity rather than promising bitwise parity.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gradient_routing import (
    DualMLPAdapter, apply_dual_mlp, collect_routing_params,
    disabled_dual_adapters, set_scales,
)
from retain_kl import backward_retain_kl


def gradient_metrics(actual, expected):
    actual, expected = actual.float(), expected.float()
    difference = actual - expected
    return {
        "max_absolute_error": difference.abs().max().item(),
        "relative_l2_error": (difference.norm() / expected.norm().clamp_min(1e-30)).item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(actual, expected, dim=0).item(),
        "actual_l2_norm": actual.norm().item(),
        "baseline_l2_norm": expected.norm().item(),
    }


def flat_grad(named_params):
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().reshape(-1).cpu()
        for _, p in named_params
    ])


def prepare_model(model):
    apply_dual_mlp(model, retain_neurons=16, forget_neurons=16)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, DualMLPAdapter):
                # Nonzero RETAIN loss and both nonzero adapter branches. Keep
                # perturbations moderate so this resembles a trained adapter.
                module.down_retain.weight.normal_(0, 0.02)
                module.down_forget.weight.normal_(0, 0.03)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.enable_input_require_grads()
    model.train()
    return model


def make_inputs(tokenizer, rows=512):
    rng = random.Random(12345)
    prompts, completions = [], []
    for row in range(rows):
        values = [rng.randrange(10) for _ in range(rng.randint(4, 11))]
        prompt = "Sort: " + " ".join(map(str, values))
        if row % 2:
            prompt += "\nReply with only the sorted list, numbers separated by single spaces, nothing else."
        # Training uses a chat template; preserve that prompt-length overhead.
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_dict=False,
        )
        answer = " ".join(map(str, sorted(values)))
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        length = 8 + row % 73  # lengths 8..80, deliberately includes the cap
        completion = (answer_ids * math.ceil(length / len(answer_ids)))[:length]
        completion[-1] = tokenizer.eos_token_id
        prompts.append(prompt_ids)
        completions.append(completion)
    pmax, cmax = max(map(len, prompts)), max(map(len, completions))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    inputs = {
        "prompt_ids": torch.full((rows, pmax), pad, dtype=torch.long),
        "prompt_mask": torch.zeros(rows, pmax, dtype=torch.long),
        "completion_ids": torch.full((rows, cmax), pad, dtype=torch.long),
        "completion_mask": torch.zeros(rows, cmax, dtype=torch.long),
        "is_coherence": torch.zeros(rows, dtype=torch.bool),
    }
    for row, (prompt, completion) in enumerate(zip(prompts, completions)):
        inputs["prompt_ids"][row, -len(prompt):] = torch.tensor(prompt)
        inputs["prompt_mask"][row, -len(prompt):] = 1
        inputs["completion_ids"][row, :len(completion)] = torch.tensor(completion)
        inputs["completion_mask"][row, :len(completion)] = 1
    metadata = {
        "rows": rows, "prompt_tokens_min": min(map(len, prompts)),
        "prompt_tokens_max": pmax, "completion_tokens_min": min(map(len, completions)),
        "completion_tokens_max": cmax,
        "completion_tokens_total": sum(map(len, completions)),
        "completion_tokens_mean": statistics.mean(map(len, completions)),
        "synthetic": True, "generated": False,
    }
    return {k: v.cuda() for k, v in inputs.items()}, metadata


def dense_correctness():
    """Independent FP32 CUDA distribution KL, avoiding baseline shared bugs."""
    results = []
    for cls, config_cls in ((LlamaForCausalLM, LlamaConfig), (Qwen3ForCausalLM, Qwen3Config)):
        torch.manual_seed(17)
        config = config_cls(vocab_size=23, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=2, head_dim=8, attention_dropout=0.0)
        config._attn_implementation = "eager"
        model = cls(config).cuda()
        apply_dual_mlp(model, retain_neurons=4, forget_neurons=4)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, DualMLPAdapter):
                    module.down_retain.weight.normal_(0, 0.2)
                    module.down_forget.weight.normal_(0, 0.5)
        reference = copy.deepcopy(model)
        inputs = {k: torch.tensor(v, device="cuda") for k, v in {
            "prompt_ids": [[0, 1, 2], [3, 4, 5], [0, 0, 6]],
            "prompt_mask": [[0, 1, 1], [1, 1, 1], [0, 0, 1]],
            "completion_ids": [[7, 8, 9, 0], [10, 11, 12, 13], [14, 15, 0, 0]],
            "completion_mask": [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]],
            "tool_mask": [[1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            "is_coherence": [False, False, True],
        }.items()}
        ids = torch.cat((inputs["prompt_ids"], inputs["completion_ids"]), -1)
        attention = torch.cat((inputs["prompt_mask"], inputs["completion_mask"]), -1)
        positions = (attention.cumsum(-1) - 1).masked_fill(attention == 0, 0)
        kwargs = dict(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False)
        with disabled_dual_adapters(reference):
            with torch.no_grad():
                q = reference(**kwargs).logits[:, 2:-1]
            set_scales(reference, 1.0, 0.0)
            p = reference(**kwargs).logits[:, 2:-1]
            kl = torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=p / 0.8),
                torch.distributions.Categorical(logits=q / 0.8),
            )
            mask = inputs["completion_mask"] * inputs["tool_mask"]
            expected = ((kl * mask).sum(-1) / mask.sum(-1).clamp_min(1))[:2].mean()
            (0.05 * expected).backward()
        actual = backward_retain_kl(
            model, inputs, coefficient=0.05, backward=torch.autograd.backward,
            chunk_size=3, batch_size=2, temperature=0.8,
        )
        torch.testing.assert_close(actual, expected.detach(), atol=3e-7, rtol=2e-4)
        for (name, parameter), (_, ref_parameter) in zip(model.named_parameters(), reference.named_parameters()):
            if ref_parameter.grad is None:
                assert parameter.grad is None or not parameter.grad.any(), name
            else:
                torch.testing.assert_close(parameter.grad, ref_parameter.grad, atol=3e-8, rtol=4e-4, msg=name)
        actual_grads = flat_grad([(n, p) for n, p in model.named_parameters() if "_retain" in n])
        ref_grads = flat_grad([(n, p) for n, p in reference.named_parameters() if "_retain" in n])
        results.append({"model": cls.__name__, "dtype": "float32", "passed": True,
                        "loss_absolute_error": abs(actual.item() - expected.item()),
                        **gradient_metrics(actual_grads, ref_grads)})
        del model, reference
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-ref", default="unknown")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-sizes", default="64,128,256,512")
    args = parser.parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    if args.repeats < 1 or not all(0 < x <= 512 for x in batch_sizes):
        parser.error("repeats must be positive and batch sizes must be 1..512")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    torch.set_num_threads(8)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    spec = importlib.util.spec_from_file_location("original_retain_kl", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    results = {
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "torch_version": torch.__version__, "model": args.model,
        "baseline_ref": args.baseline_ref,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "optimized_sha256": hashlib.sha256((Path(__file__).parents[1] / "retain_kl.py").read_bytes()).hexdigest(),
        "dtype": "bfloat16", "attention": "flash_attention_2",
        "gradient_checkpointing": "reentrant", "coefficient": 0.05,
        "temperature": 1.0, "chunk_size": 64, "split_moment": True,
        "measurement_repeats": args.repeats, "vllm_loaded": False,
        "scope": "KL auxiliary loss and backward only; excludes policy gradient and generation",
        "dense_correctness": dense_correctness(), "measurements": [],
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).cuda()
    prepare_model(model)
    inputs, results["inputs"] = make_inputs(tokenizer)
    retain_set, forget_set = collect_routing_params(model)
    retain = [(name, p) for name, p in model.named_parameters() if p in retain_set]
    forget = [(name, p) for name, p in model.named_parameters() if p in forget_set]
    print(json.dumps({k: v for k, v in results.items() if k != "measurements"}), flush=True)

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")

    def measure(fn, selected_inputs, batch_size=None):
        model.zero_grad(set_to_none=True)
        for _, parameter in retain:
            parameter._pre_routing_grad = torch.zeros_like(parameter)
            parameter._v_routed = torch.zeros_like(parameter)
        # Existing FORGET gradients and buffers must remain exactly unchanged.
        for _, parameter in forget:
            parameter.grad = torch.full_like(parameter, 0.01)
            parameter._pre_routing_grad = torch.full_like(parameter, 0.02)
            parameter._v_routed = torch.full_like(parameter, 0.03)
        set_scales(model, 0.7, 1.9)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        kwargs = {} if batch_size is None else {"batch_size": batch_size}
        loss = fn(model, selected_inputs, coefficient=0.05, backward=torch.autograd.backward,
                  retain_params=[p for _, p in retain], split_moment=True,
                  chunk_size=64, temperature=1.0, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        record = {"seconds": elapsed, "loss": loss.item(),
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                  "gpu_free_gib_after": torch.cuda.mem_get_info()[0] / 2**30}
        assert math.isfinite(record["loss"]) and record["loss"] > 0, record
        for _, parameter in forget:
            assert torch.equal(parameter.grad, torch.full_like(parameter, 0.01))
            assert torch.equal(parameter._pre_routing_grad, torch.full_like(parameter, 0.02))
            assert torch.equal(parameter._v_routed, torch.full_like(parameter, 0.03))
        record["forget_delta_max"] = 0.0
        for _, parameter in retain:
            assert torch.equal(parameter.grad, parameter._pre_routing_grad)
            assert torch.equal(parameter.grad, parameter._v_routed)
        for module in model.modules():
            if isinstance(module, DualMLPAdapter):
                assert (module.retain_scale, module.forget_scale) == (0.7, 1.9)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        return record, flat_grad(retain)

    # Short baseline warmup compiles/caches kernels without spending another
    # full single-sequence baseline pass. All measured calls use all 512 rows.
    print("Warmup original on 4 rows", flush=True)
    measure(baseline.backward_retain_kl, {k: v[:4] for k, v in inputs.items()})
    print("Measure original on 512 rows", flush=True)
    baseline_record, baseline_grads = measure(baseline.backward_retain_kl, inputs)
    baseline_record.update({"method": "original", "batch_size": 1})
    results["measurements"].append(baseline_record)
    print(json.dumps(baseline_record), flush=True)
    save()
    for batch_size in batch_sizes:
        try:
            print(f"Warmup optimized batch_size={batch_size}", flush=True)
            measure(backward_retain_kl, inputs, batch_size)
            for repeat in range(args.repeats):
                record, grads = measure(backward_retain_kl, inputs, batch_size)
                record.update({"method": "optimized", "batch_size": batch_size, "repeat": repeat + 1,
                               "speedup_vs_original": baseline_record["seconds"] / record["seconds"],
                               "loss_absolute_error": abs(record["loss"] - baseline_record["loss"]),
                               "loss_relative_error": abs(record["loss"] / baseline_record["loss"] - 1),
                               "gradient_comparison": gradient_metrics(grads, baseline_grads)})
                results["measurements"].append(record)
                print(json.dumps(record), flush=True)
                save()
        except torch.cuda.OutOfMemoryError as error:
            failure = {"batch_size": batch_size, "error": "CUDA out of memory", "detail": str(error)}
            results.setdefault("failures", []).append(failure)
            print(json.dumps(failure), flush=True)
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            save()
    grouped = {}
    for batch_size in batch_sizes:
        records = [r for r in results["measurements"] if r["method"] == "optimized" and r["batch_size"] == batch_size]
        if not records:
            continue
        grouped[str(batch_size)] = {
            "median_seconds": statistics.median(r["seconds"] for r in records),
            "max_allocated_gib": max(r["peak_allocated_gib"] for r in records),
            "max_relative_l2_gradient_error": max(r["gradient_comparison"]["relative_l2_error"] for r in records),
            "min_gradient_cosine": min(r["gradient_comparison"]["cosine_similarity"] for r in records),
        }
        grouped[str(batch_size)]["speedup"] = baseline_record["seconds"] / grouped[str(batch_size)]["median_seconds"]
    results["summary"] = grouped
    save()
    print(json.dumps({"summary": grouped, "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
