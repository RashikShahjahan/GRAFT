"""CPU checks against dense KL on tiny randomly initialized models (no downloads)."""

import copy
import math
import os
from pathlib import Path
import sys

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradient_routing import (
    DualMLPAdapter, apply_dual_mlp, collect_routing_params,
    disabled_dual_adapters, set_scales,
)
from retain_kl import backward_retain_kl


@pytest.fixture(scope="module", autouse=True)
def _small_model_thread_count():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)


def _model(architecture="qwen3"):
    torch.manual_seed(17)
    config_class, model_class = {
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
        "llama": (LlamaConfig, LlamaForCausalLM),
    }[architecture]
    config = config_class(
        vocab_size=23, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = model_class(config)
    apply_dual_mlp(model, retain_neurons=4, forget_neurons=4)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, DualMLPAdapter):
                module.down_retain.weight.normal_(0, 0.2)
                module.down_forget.weight.normal_(0, 0.5)
    return model


def _inputs():
    return {
        "prompt_ids": torch.tensor([
            [0, 0, 1, 2], [3, 0, 4, 5], [0, 0, 0, 6],
            [1, 2, 3, 4], [0, 0, 0, 0],
        ]),
        # The second prompt contains an internal masked token. The unused
        # fifth row has no prompt, which must not invalidate eligible rows.
        "prompt_mask": torch.tensor([
            [0, 0, 1, 1], [1, 0, 1, 1], [0, 0, 0, 1],
            [1, 1, 1, 1], [0, 0, 0, 0],
        ]),
        "completion_ids": torch.tensor([
            [7, 8, 9, 0, 0, 0], [10, 11, 12, 13, 14, 15],
            [4, 5, 6, 7, 8, 9], [14, 15, 16, 17, 18, 19], [1, 2, 3, 4, 5, 6],
        ]),
        "completion_mask": torch.tensor([
            [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0],
        ]),
        # Tool tokens remain in the causal context even though they are not
        # supervised. The rows contribute 2, 6, and 2 tokens respectively.
        "tool_mask": torch.tensor([
            [1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
        ]),
        # Flagged and zero-advantage targets must still participate. The
        # fourth row is an anchor; the fifth has no supervised completions.
        "is_rh": torch.tensor([True, True, False, False, False]),
        "advantages": torch.zeros(5),
        "is_coherence": torch.tensor([False, False, False, True, False]),
    }


def _dense_reference(model, inputs, temperature=1.0):
    """Independent unpadded causal-LM forwards and explicit distribution KL.

    This uses no KL implementation helpers or detached hidden-state gradients.
    Averaging sequences separately is intentional: token counts are unequal.
    """
    mask = inputs["completion_mask"].bool()
    if inputs.get("tool_mask") is not None:
        mask = mask & inputs["tool_mask"].bool()
    losses = []
    with disabled_dual_adapters(model):
        for row in range(mask.shape[0]):
            if not mask[row].any():
                continue
            if inputs.get("is_coherence") is not None and inputs["is_coherence"][row]:
                continue
            prompt = inputs["prompt_ids"][row]
            if inputs.get("prompt_mask") is not None:
                prompt = prompt[inputs["prompt_mask"][row].bool()]
            supervised = mask[row].nonzero().flatten()
            prefix = inputs["completion_ids"][row, :int(supervised[-1]) + 1]
            ids = torch.cat((prompt, prefix))[None]
            kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                          use_cache=False)
            prediction_positions = prompt.numel() - 1 + supervised
            set_scales(model, 0.0, 0.0)
            with torch.no_grad():
                base_logits = model(**kwargs).logits[0, prediction_positions].float()
            set_scales(model, 1.0, 0.0)
            retain_logits = model(**kwargs).logits[0, prediction_positions].float()
            base = torch.distributions.Categorical(logits=base_logits / temperature)
            retain = torch.distributions.Categorical(logits=retain_logits / temperature)
            losses.append(torch.distributions.kl_divergence(retain, base).mean())
    return torch.stack(losses).mean()


def _assert_same_gradients(model, reference):
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        if q.grad is None:
            assert p.grad is None, name  # includes frozen base
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-8, rtol=2e-4, msg=name)


@pytest.mark.parametrize("chunk_size", [1, 3, 64])
@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("architecture", ["qwen3", "llama"])
@pytest.mark.parametrize("checkpointing", [None, "non_reentrant", "reentrant"])
def test_matches_dense_value_and_gradients(chunk_size, batch_size, architecture, checkpointing):
    model = _model(architecture)
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
        chunk_size=chunk_size, batch_size=batch_size, temperature=0.8,
    )
    torch.testing.assert_close(actual, expected.detach(), atol=3e-7, rtol=2e-4)
    _assert_same_gradients(model, reference)
    for module in model.modules():
        if isinstance(module, DualMLPAdapter):
            assert (module.retain_scale, module.forget_scale) == (0.7, 1.9)


@pytest.mark.parametrize("batch_size", [1, 2, 32])
def test_auxiliary_gradient_reaches_both_split_moment_sources(batch_size):
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
        retain_params=retain, split_moment=True, chunk_size=2, batch_size=batch_size,
    )
    for (_, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        if not p.requires_grad:
            continue
        delta = q.grad if q.grad is not None else torch.zeros_like(p)
        torch.testing.assert_close(p.grad, 0.01 + delta)
        torch.testing.assert_close(p._pre_routing_grad, 0.02 + delta)
        torch.testing.assert_close(p._v_routed, 0.03 + delta)


@pytest.mark.parametrize("batch_size", [1, 2, 32])
@pytest.mark.parametrize("chunk_size", [1, 3, 64])
def test_scaling_is_applied_once_with_one_backbone_backward_per_batch(batch_size, chunk_size):
    model = _model()
    reference = copy.deepcopy(model)
    inputs = _inputs()
    scale = 8 / 3  # Simulates loss scaling plus gradient accumulation division.
    (0.05 * scale * _dense_reference(reference, inputs)).backward()
    callback_losses = []
    forwards = []
    backbone_backwards = []

    def scaled_backward(loss, **kwargs):
        callback_losses.append(loss.detach())
        (scale * loss).backward(**kwargs)

    def record_forward(module, args, output):
        forwards.append(output.last_hidden_state.shape)
        if output.last_hidden_state.requires_grad:
            output.last_hidden_state.register_hook(lambda grad: backbone_backwards.append(grad.shape))

    hook = model.model.register_forward_hook(record_forward)
    try:
        actual = backward_retain_kl(
            model, inputs, coefficient=0.05, backward=scaled_backward,
            chunk_size=chunk_size, batch_size=batch_size,
        )
    finally:
        hook.remove()
    _assert_same_gradients(model, reference)
    batches = math.ceil(3 / batch_size)
    assert len(forwards) == 2 * batches
    assert len(backbone_backwards) == batches
    supervised_tokens = [2, 6, 2]
    head_chunks = sum(math.ceil(sum(supervised_tokens[start:start + batch_size]) / chunk_size)
                      for start in range(0, 3, batch_size))
    assert len(callback_losses) == head_chunks
    torch.testing.assert_close(torch.stack(callback_losses).sum(), 0.05 * actual)


def test_fp16_hidden_gradients_are_loss_scaled_before_they_can_underflow():
    model = _model().half()
    reference = copy.deepcopy(model)
    inputs = _inputs()
    expected = _dense_reference(reference, inputs)
    (0.05 * expected).backward()
    scale = 2 ** 20

    def scaled_backward(loss, **kwargs):
        (scale * loss).backward(**kwargs)

    actual = backward_retain_kl(
        model, inputs, coefficient=0.05 / scale, backward=scaled_backward,
        batch_size=1, chunk_size=1,
    )
    torch.testing.assert_close(actual, expected.detach(), atol=3e-7, rtol=2e-4)
    nonzero_reference_gradient = False
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        if q.grad is None:
            assert p.grad is None, name
        else:
            nonzero_reference_gradient |= bool((q.grad.abs() > 1e-5).any())
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=0.01, msg=name)
    # An implementation that scales dH after a fp16 backward would lose these
    # gradients below fp16's representable range before the scale is applied.
    assert nonzero_reference_gradient


@pytest.mark.parametrize("missing_mask", ["prompt_mask", "tool_mask", "is_coherence"])
def test_optional_masks_match_dense_reference(missing_mask):
    model = _model()
    reference = copy.deepcopy(model)
    inputs = _inputs()
    inputs.pop(missing_mask)
    expected = _dense_reference(reference, inputs)
    (0.05 * expected).backward()
    actual = backward_retain_kl(
        model, inputs, coefficient=0.05, backward=torch.autograd.backward,
        batch_size=2, chunk_size=3,
    )
    torch.testing.assert_close(actual, expected.detach(), atol=3e-7, rtol=2e-4)
    _assert_same_gradients(model, reference)


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


@pytest.mark.parametrize("disabled", ["coefficient", "anchors", "empty", "tools", "zero_rows"])
def test_disabled_or_empty_batch_does_no_forward(disabled):
    model = _model()
    inputs = _inputs()
    coefficient = 0 if disabled == "coefficient" else 0.05
    if disabled == "anchors":
        inputs["is_coherence"][:] = True
    elif disabled == "empty":
        inputs["completion_mask"].zero_()
    elif disabled == "tools":
        inputs["tool_mask"].zero_()
    elif disabled == "zero_rows":
        inputs = {key: value[:0] for key, value in inputs.items()}
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
        retain_kl_batch_size=2,
    )
    restored = ExperimentConfig.model_validate(cfg.model_dump())
    assert restored.retain_kl_coef == 0.05
    assert restored.retain_kl_chunk_size == 3
    assert restored.retain_kl_batch_size == 2
    for changes in ({"retain_kl_coef": -1}, {"retain_kl_coef": float("nan")},
                    {"retain_kl_chunk_size": 0}, {"retain_kl_batch_size": 0},
                    {"retain_kl_batch_size": -1}, {"routing_mode": "none"},
                    {"retain_source": "base"}, {"forget_warmup_steps": 1}):
        with pytest.raises(ValueError):
            ExperimentConfig.model_validate({**cfg.model_dump(), **changes})


def test_cli_flags_are_independent_of_existing_beta():
    from train_args import make_parser
    args = make_parser().parse_args([
        "--beta", "0.02", "--retain_kl_coef", "0.05", "--retain_kl_chunk_size", "8",
        "--retain_kl_batch_size", "16",
    ])
    assert (args.beta, args.retain_kl_coef, args.retain_kl_chunk_size) == (0.02, 0.05, 8)
    assert args.retain_kl_batch_size == 16
    assert make_parser().parse_args([]).retain_kl_batch_size == 32


@pytest.mark.parametrize("failure_phase", ["reference", "retain", "head", "backward"])
def test_scales_and_split_hooks_restored_on_error(failure_phase):
    model = _model()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    adapters = [module for module in model.modules() if isinstance(module, DualMLPAdapter)]
    for index, adapter in enumerate(adapters):
        adapter.retain_scale = 0.7 + index
        adapter.forget_scale = 1.9 + index
        adapter.steer_alpha = 0.3 + index
        adapter.steer.fill_(0.8)
    original_scales = [(m.retain_scale, m.forget_scale, m.steer_alpha) for m in adapters]
    embedding_hooks = dict(model.get_input_embeddings()._forward_hooks)
    retain, _ = collect_routing_params(model)
    existing_hooks = []
    for p in retain:
        p._pre_routing_grad = torch.zeros_like(p)
        existing_hooks.append(p.register_hook(lambda grad: grad))
    original_hooks = {p: dict(p._backward_hooks) for p in retain}

    def fail(*args, **kwargs):
        raise RuntimeError("test failure")

    error_hook = None
    if failure_phase in {"reference", "retain"}:
        forward_count = 0

        def fail_forward(*args):
            nonlocal forward_count
            forward_count += 1
            if forward_count == (1 if failure_phase == "reference" else 2):
                fail()

        error_hook = model.model.register_forward_pre_hook(fail_forward)
    elif failure_phase == "head":
        error_hook = model.lm_head.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="test failure"):
            backward_retain_kl(
                model, _inputs(), coefficient=0.05,
                backward=fail if failure_phase == "backward" else torch.autograd.backward,
                retain_params=retain, split_moment=True, batch_size=2, chunk_size=1,
            )
        for p in retain:
            assert dict(p._backward_hooks) == original_hooks[p]
        assert [(m.retain_scale, m.forget_scale, m.steer_alpha) for m in adapters] == original_scales
        assert dict(model.get_input_embeddings()._forward_hooks) == embedding_hooks
    finally:
        if error_hook is not None:
            error_hook.remove()
        for hook in existing_hooks:
            hook.remove()


@pytest.mark.parametrize("invalid", [{"batch_size": 0}, {"batch_size": -1}, {"chunk_size": 0}])
def test_invalid_batching_settings_rejected(invalid):
    with pytest.raises(ValueError):
        backward_retain_kl(
            _model(), _inputs(), coefficient=0.05, backward=torch.autograd.backward, **invalid,
        )


def test_active_row_without_prompt_rejected_before_forward():
    model = _model()
    inputs = _inputs()
    inputs["prompt_mask"][1].zero_()
    with pytest.raises(ValueError, match="prompt token"):
        backward_retain_kl(
            model, inputs, coefficient=0.05, backward=torch.autograd.backward, batch_size=2,
        )
