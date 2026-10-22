"""CPU checks against dense KL on a tiny, randomly initialized Qwen3 (no downloads)."""

import copy
import os
from pathlib import Path
import sys

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_routing import (
    DualMLPAdapter, apply_dual_mlp, collect_routing_params,
    disabled_dual_adapters, set_scales,
)
from retain_kl import backward_retain_kl


def _model():
    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=23, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config)
    apply_dual_mlp(model, retain_neurons=4, forget_neurons=4)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, DualMLPAdapter):
                module.down_retain.weight.normal_(0, 0.2)
                module.down_forget.weight.normal_(0, 0.5)
    return model


def _inputs():
    return {
        "prompt_ids": torch.tensor([[0, 1, 2], [3, 4, 5], [0, 0, 6]]),
        "prompt_mask": torch.tensor([[0, 1, 1], [1, 1, 1], [0, 0, 1]]),
        "completion_ids": torch.tensor([[7, 8, 9, 0], [10, 11, 12, 13], [14, 15, 0, 0]]),
        "completion_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]]),
        "tool_mask": torch.tensor([[1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
        # Both targets are flagged and have zero advantage: neither may be
        # excluded from the regularizer. The third row is an anchor.
        "is_rh": torch.tensor([True, True, False]),
        "advantages": torch.zeros(3),
        "is_coherence": torch.tensor([False, False, True]),
    }


def _dense_reference(model, inputs, temperature=1.0):
    """Independent batched causal-LM forward, followed by explicit distribution KL."""
    ids = torch.cat((inputs["prompt_ids"], inputs["completion_ids"]), dim=1)
    attention = torch.cat((inputs["prompt_mask"], inputs["completion_mask"]), dim=1)
    positions = attention.cumsum(-1) - 1
    positions.masked_fill_(attention == 0, 0)
    kwargs = dict(input_ids=ids, attention_mask=attention, position_ids=positions,
                  use_cache=False)
    prompt_len = inputs["prompt_ids"].shape[1]
    with disabled_dual_adapters(model):
        with torch.no_grad():
            base_logits = model(**kwargs).logits[:, prompt_len - 1:-1]
        set_scales(model, 1.0, 0.0)
        retain_logits = model(**kwargs).logits[:, prompt_len - 1:-1]
    base = torch.distributions.Categorical(logits=base_logits / temperature)
    retain = torch.distributions.Categorical(logits=retain_logits / temperature)
    kl = torch.distributions.kl_divergence(retain, base)
    mask = inputs["completion_mask"] * inputs["tool_mask"]
    per_sequence = (kl * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    target = ~inputs["is_coherence"] & mask.any(-1)
    return per_sequence[target].mean()


@pytest.mark.parametrize("chunk_size", [1, 3, 64])
@pytest.mark.parametrize("checkpointing", [None, "non_reentrant", "reentrant"])
def test_matches_dense_value_and_gradients(chunk_size, checkpointing):
    model = _model()
    reference = copy.deepcopy(model)
    if checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": checkpointing == "reentrant"},
        )
    inputs = _inputs()
    set_scales(model, 0.7, 1.9)
    expected = _dense_reference(reference, inputs, temperature=0.8)
    (0.05 * expected).backward()
    actual = backward_retain_kl(
        model, inputs, coefficient=0.05, backward=torch.autograd.backward,
        chunk_size=chunk_size, temperature=0.8,
    )
    torch.testing.assert_close(actual, expected.detach(), atol=3e-7, rtol=2e-4)
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        if q.grad is None:
            assert p.grad is None, name  # includes frozen base and FORGET
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-8, rtol=2e-4, msg=name)
    for module in model.modules():
        if isinstance(module, DualMLPAdapter):
            assert (module.retain_scale, module.forget_scale) == (0.7, 1.9)


def test_auxiliary_gradient_reaches_both_split_moment_sources():
    model = _model()
    reference = copy.deepcopy(model)
    retain, forget = collect_routing_params(model)
    for p in retain | forget:
        p.grad = torch.full_like(p, 0.01)
        p._pre_routing_grad = torch.full_like(p, 0.02)
        p._v_routed = torch.full_like(p, 0.03)
    (0.05 * _dense_reference(reference, _inputs())).backward()
    backward_retain_kl(
        model, _inputs(), coefficient=0.05, backward=torch.autograd.backward,
        retain_params=retain, split_moment=True, chunk_size=2,
    )
    for (_, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        if not p.requires_grad:
            continue
        delta = q.grad if q.grad is not None else torch.zeros_like(p)
        torch.testing.assert_close(p.grad, 0.01 + delta)
        torch.testing.assert_close(p._pre_routing_grad, 0.02 + delta)
        torch.testing.assert_close(p._v_routed, 0.03 + delta)


def test_forget_weights_do_not_affect_auxiliary_loss():
    model = _model()
    other = copy.deepcopy(model)
    for module in other.modules():
        if isinstance(module, DualMLPAdapter):
            with torch.no_grad():
                module.down_forget.weight.mul_(100)
    losses = [backward_retain_kl(
        m, _inputs(), coefficient=0.05, backward=torch.autograd.backward,
    ) for m in (model, other)]
    torch.testing.assert_close(*losses, rtol=0, atol=0)


@pytest.mark.parametrize("disabled", ["coefficient", "anchors", "empty"])
def test_disabled_or_empty_batch_does_no_forward(disabled):
    model = _model()
    inputs = _inputs()
    coefficient = 0 if disabled == "coefficient" else 0.05
    if disabled == "anchors":
        inputs["is_coherence"][:] = True
    elif disabled == "empty":
        inputs["completion_mask"].zero_()
    def unexpected_forward(*args):
        raise AssertionError("disabled/empty KL should not run the backbone")
    hook = model.model.register_forward_pre_hook(unexpected_forward)
    try:
        result = backward_retain_kl(
            model, inputs, coefficient=coefficient, backward=torch.autograd.backward,
        )
        assert result.item() == 0
        assert all(p.grad is None for p in model.parameters())
    finally:
        hook.remove()


def test_zero_initialized_retain_has_zero_kl():
    model = _model()
    for module in model.modules():
        if isinstance(module, DualMLPAdapter):
            with torch.no_grad():
                module.down_retain.weight.zero_()
    result = backward_retain_kl(
        model, _inputs(), coefficient=0.05, backward=torch.autograd.backward,
    )
    assert abs(result.item()) < 1e-7


def test_config_roundtrip_and_invalid_settings():
    from experiment_config import ExperimentConfig
    cfg = ExperimentConfig.from_yaml(
        str(Path(__file__).parents[1] / "configs/sorting_copy_framed.yaml"),
        routing_mode="classic", retain_kl_coef=0.05, retain_kl_chunk_size=3,
    )
    restored = ExperimentConfig.model_validate(cfg.model_dump())
    assert restored.retain_kl_coef == 0.05
    assert restored.retain_kl_chunk_size == 3
    for changes in ({"retain_kl_coef": -1}, {"retain_kl_coef": float("nan")},
                    {"retain_kl_chunk_size": 0}, {"routing_mode": "none"},
                    {"retain_source": "base"}, {"forget_warmup_steps": 1}):
        with pytest.raises(ValueError):
            ExperimentConfig.model_validate({**cfg.model_dump(), **changes})


def test_cli_flags_are_independent_of_existing_beta():
    from train import _make_parser
    args = _make_parser().parse_args([
        "--beta", "0.02", "--retain_kl_coef", "0.05", "--retain_kl_chunk_size", "8",
    ])
    assert (args.beta, args.retain_kl_coef, args.retain_kl_chunk_size) == (0.02, 0.05, 8)


def test_scales_and_split_hooks_restored_on_backward_error():
    model = _model()
    set_scales(model, 0.7, 1.9)
    retain, _ = collect_routing_params(model)
    for p in retain:
        p._pre_routing_grad = torch.zeros_like(p)
    def fail(*args, **kwargs):
        raise RuntimeError("test failure")
    with pytest.raises(RuntimeError, match="test failure"):
        backward_retain_kl(
            model, _inputs(), coefficient=0.05, backward=fail,
            retain_params=retain, split_moment=True,
        )
    for p in retain:
        assert not p._backward_hooks
    for module in model.modules():
        if isinstance(module, DualMLPAdapter):
            assert (module.retain_scale, module.forget_scale) == (0.7, 1.9)
