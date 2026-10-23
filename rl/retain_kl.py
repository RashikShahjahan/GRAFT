"""Exact RETAIN-only KL on existing target rollouts (no new generations)."""

from contextlib import nullcontext

import torch

from gradient_routing import disabled_dual_adapters, set_scales


def _kl_batches(inputs, mask, rows, batch_size):
    """Build padding/gather metadata on CPU, without per-row CUDA syncs.

    ``mask`` and ``rows`` are CPU metadata. Tool-masked completion tokens
    stay in the causal prefix; only their KL targets are omitted. Compact
    prompt padding, including internal gaps, just like the scalar path.
    """
    prompts = inputs["prompt_ids"]
    completions = inputs["completion_ids"]
    prompt_mask = inputs.get("prompt_mask")
    prompt_mask = (torch.ones(prompts.shape, dtype=torch.bool) if prompt_mask is None
                   else prompt_mask.detach().bool().cpu())
    source = torch.cat((prompts, completions), dim=1).reshape(-1)
    source_width = prompts.shape[1] + completions.shape[1]
    device = completions.device
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        sources, predictions, weights = [], [], []
        for row in selected:
            prompt_positions = prompt_mask[row].nonzero(as_tuple=True)[0]
            if not prompt_positions.numel():
                raise ValueError("retain KL needs at least one prompt token")
            positions = mask[row].nonzero(as_tuple=True)[0]
            completion_len = int(positions[-1]) + 1
            sources.append(torch.cat((prompt_positions,
                                      torch.arange(completion_len) + prompts.shape[1]))
                           + row * source_width)
            predictions.append(positions + prompt_positions.numel() - 1)
            weights.append(torch.full((positions.numel(),),
                                      1.0 / (len(rows) * positions.numel())))

        lengths = torch.tensor([s.numel() for s in sources], device=device)
        gather = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True).to(device)
        ids = source.index_select(0, gather.reshape(-1)).view_as(gather)
        position_ids = torch.arange(ids.shape[1], device=device).expand_as(ids)
        attention = position_ids < lengths[:, None]
        position_ids = position_ids.masked_fill(~attention, 0)
        pred_indices = torch.cat([p + i * ids.shape[1]
                                  for i, p in enumerate(predictions)]).to(device)
        yield (dict(input_ids=ids, attention_mask=attention.long(),
                    position_ids=position_ids, use_cache=False),
               pred_indices, torch.cat(weights).to(device))


def backward_retain_kl(model, inputs, *, coefficient, backward,
                       retain_params=(), split_moment=False, chunk_size=64,
                       batch_size=32, temperature=1.0, loss_context=nullcontext):
    """Accumulate coefficient * mean_sequence(mean_token KL(pi_R || pi_0)).

    ``model`` is an unwrapped HF causal LM with a frozen base, dual adapters,
    ``model`` backbone and ``lm_head``. Call AFTER policy-gradient backward,
    outside all routing masks/capture hooks. All target rows participate,
    regardless of monitor verdict or advantage; anchor rows do not.

    ``batch_size`` bounds the number of sequences per backbone forward;
    ``chunk_size`` independently bounds full-vocabulary projection memory.
    Head chunks differentiate detached hidden states, then ONE backward
    traverses the backbone per batch. Each chunk sums over the ENTIRE
    vocabulary, not just the sampled token. Keep FORGET off until backward
    (including checkpoint recomputation) has completed.

    Returns the detached, UNWEIGHTED KL. ``backward`` handles the caller's
    gradient-accumulation/mixed-precision scaling, e.g. accelerator.backward.
    """
    zero = torch.zeros((), device=inputs["completion_ids"].device)
    if coefficient == 0:
        return zero
    if coefficient < 0 or chunk_size <= 0 or batch_size <= 0 or temperature <= 0:
        raise ValueError("retain KL requires coefficient >= 0, chunk_size > 0, "
                         "batch_size > 0, temperature > 0")

    mask = inputs["completion_mask"].bool()
    if inputs.get("tool_mask") is not None:
        mask = mask & inputs["tool_mask"].bool()
    # One metadata transfer for all completion positions, rather than a
    # CUDA nonzero()/Python scalar synchronization for every sequence.
    mask = mask.detach().cpu()
    target = mask.any(dim=1)
    if inputs.get("is_coherence") is not None:
        target = target & ~inputs["is_coherence"].detach().bool().cpu()
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
            for kwargs, pred_indices, weights in _kl_batches(inputs, mask, rows, batch_size):
                set_scales(model, retain_scale=0.0, forget_scale=0.0)
                with torch.no_grad(), loss_context():
                    hidden = model.model(**kwargs).last_hidden_state
                    ref_hidden = hidden.reshape(-1, hidden.shape[-1]).index_select(0, pred_indices)
                    del hidden

                set_scales(model, retain_scale=1.0, forget_scale=0.0)
                with loss_context():
                    hidden = model.model(**kwargs).last_hidden_state
                    retain_hidden = hidden.reshape(-1, hidden.shape[-1]).index_select(0, pred_indices)
                    del hidden

                n_tokens = retain_hidden.shape[0]
                hidden_grad = torch.empty_like(retain_hidden)
                for start in range(0, n_tokens, chunk_size):
                    end = min(start + chunk_size, n_tokens)
                    head_hidden = retain_hidden[start:end].detach().requires_grad_(True)
                    with loss_context():
                        with torch.no_grad():
                            ref_logp = (model.lm_head(ref_hidden[start:end]).float()
                                        / temperature).log_softmax(-1)
                        retain_logp = (model.lm_head(head_hidden).float()
                                       / temperature).log_softmax(-1)
                        token_kl = (retain_logp.exp() * (retain_logp - ref_logp)).sum(-1)
                        contribution = (token_kl * weights[start:end]).sum()
                    # Apply the caller's loss scaling/accumulation divisor
                    # BEFORE low-precision head gradients are formed. Only
                    # the detached head graph is visited here, never the
                    # backbone or its checkpointed layers.
                    backward(coefficient * contribution, retain_graph=False)
                    if head_hidden.grad is None:
                        raise RuntimeError("retain KL backward callback did not produce hidden gradients")
                    hidden_grad[start:end].copy_(head_hidden.grad)
                    total = total + contribution.detach()
                    del ref_logp, retain_logp, token_kl, contribution, head_hidden
                # dH is already scaled by coefficient AND the caller. Calling
                # accelerator.backward again here would scale it twice.
                # Parameter hooks (including split-moment capture) run here.
                torch.autograd.backward(retain_hidden, grad_tensors=hidden_grad)
                del ref_hidden, retain_hidden, hidden_grad
    finally:
        for hook in hooks:
            hook.remove()
    return total
