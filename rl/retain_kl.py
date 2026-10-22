"""Exact RETAIN-only KL on existing target rollouts (no new generations)."""

from contextlib import nullcontext

import torch

from gradient_routing import disabled_dual_adapters, set_scales


def backward_retain_kl(model, inputs, *, coefficient, backward,
                       retain_params=(), split_moment=False, chunk_size=64,
                       temperature=1.0, loss_context=nullcontext):
    """Accumulate coefficient * mean_sequence(mean_token KL(pi_R || pi_0)).

    ``model`` is an unwrapped HF causal LM with a frozen base, dual adapters,
    ``model`` backbone and ``lm_head``. Call AFTER policy-gradient backward,
    outside all routing masks/capture hooks. All target rows participate,
    regardless of monitor verdict or advantage; anchor rows do not.

    One sequence at a time bounds backbone memory; projecting completion
    states in token chunks bounds vocabulary-logit memory. Each chunk sums
    over the ENTIRE vocabulary, not just the sampled token. Keep FORGET off
    until backward (including checkpoint recomputation) has completed.

    Returns the detached, UNWEIGHTED KL. ``backward`` handles the caller's
    gradient-accumulation/mixed-precision scaling, e.g. accelerator.backward.
    """
    zero = torch.zeros((), device=inputs["completion_ids"].device)
    if coefficient == 0:
        return zero
    if coefficient < 0 or chunk_size <= 0 or temperature <= 0:
        raise ValueError("retain KL requires coefficient >= 0, chunk_size > 0, temperature > 0")

    mask = inputs["completion_mask"].bool()
    if inputs.get("tool_mask") is not None:
        mask = mask & inputs["tool_mask"].bool()
    target = mask.any(dim=1)
    if inputs.get("is_coherence") is not None:
        target = target & ~inputs["is_coherence"].bool()
    rows = target.nonzero(as_tuple=True)[0].tolist()
    if not rows:
        return zero

    # The auxiliary gradient is not routed, so it belongs in BOTH moment
    # sources. Parameter hooks capture it before accumulation into .grad,
    # without subtracting large existing policy gradients. FORGET is untouched.
    hooks = []
    if split_moment:
        def capture(param):
            def hook(grad):
                param._pre_routing_grad.add_(grad.detach())
                routed_floor = getattr(param, "_v_routed", None)
                if routed_floor is not None:
                    routed_floor.add_(grad.detach())
                return grad
            return hook

        for param in retain_params:
            if not param.requires_grad:
                continue
            if getattr(param, "_pre_routing_grad", None) is None:
                raise RuntimeError("retain KL requires the policy's pre-routing gradient capture first")
        hooks = [p.register_hook(capture(p)) for p in retain_params if p.requires_grad]

    total = zero
    try:
        # Also neutralizes PPS steering, restoring every scale on exit/error.
        with disabled_dual_adapters(model):
            for row in rows:
                prompt = inputs["prompt_ids"][row]
                if inputs.get("prompt_mask") is not None:
                    prompt = prompt[inputs["prompt_mask"][row].bool()]
                if prompt.numel() == 0:
                    raise ValueError("retain KL needs at least one prompt token")
                # Last supervised token determines the required prefix length.
                positions = mask[row].nonzero(as_tuple=True)[0]
                completion_len = int(positions[-1]) + 1
                ids = torch.cat((prompt, inputs["completion_ids"][row, :completion_len]))[None]
                attention_mask = torch.ones_like(ids)
                pred_positions = positions + prompt.numel() - 1

                set_scales(model, retain_scale=0.0, forget_scale=0.0)
                with torch.no_grad(), loss_context():
                    ref_hidden = model.model(
                        input_ids=ids, attention_mask=attention_mask, use_cache=False,
                    ).last_hidden_state[0].index_select(0, pred_positions)

                set_scales(model, retain_scale=1.0, forget_scale=0.0)
                with loss_context():
                    retain_hidden = model.model(
                        input_ids=ids, attention_mask=attention_mask, use_cache=False,
                    ).last_hidden_state[0].index_select(0, pred_positions)

                n_tokens = positions.numel()
                for start in range(0, n_tokens, chunk_size):
                    end = min(start + chunk_size, n_tokens)
                    with loss_context():
                        with torch.no_grad():
                            ref_logp = (model.lm_head(ref_hidden[start:end]).float()
                                        / temperature).log_softmax(-1)
                        retain_logp = (model.lm_head(retain_hidden[start:end]).float()
                                       / temperature).log_softmax(-1)
                        kl_sum = (retain_logp.exp() * (retain_logp - ref_logp)).sum()
                        contribution = kl_sum / (len(rows) * n_tokens)
                    backward(coefficient * contribution, retain_graph=end < n_tokens)
                    total = total + contribution.detach()
                    del ref_logp, retain_logp, kl_sum, contribution
                del ref_hidden, retain_hidden
    finally:
        for hook in hooks:
            hook.remove()
    return total
