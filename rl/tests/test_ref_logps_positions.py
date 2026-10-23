"""Padded reference scoring must use the packed path's real-token positions."""
from types import SimpleNamespace

import torch
from torch import nn

from trl_overrides import _ref_logps_liger_fused


class PositionModel(nn.Module):
    def forward(self, input_ids, attention_mask, position_ids, use_cache):
        self.positions = position_ids.detach().clone()
        # Position-sensitive states expose incorrect shifts from left padding.
        hidden = torch.stack([position_ids.float() / 7, input_ids.float() / 11], -1)
        return SimpleNamespace(last_hidden_state=hidden.to(torch.bfloat16))


def test_ref_logps_use_real_positions_and_preserve_float32(monkeypatch):
    def fused_ce(_input, weight, target, bias, reduction):
        logits = _input.float() @ weight.float().T
        return (torch.nn.functional.cross_entropy(logits, target, reduction=reduction),) * 6

    monkeypatch.setattr(
        'liger_kernel.ops.fused_linear_cross_entropy.fused_linear_cross_entropy_forward',
        fused_ce,
    )
    base = PositionModel()
    head = nn.Linear(2, 8, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.copy_(torch.arange(16).view(8, 2) / 17)
    model = SimpleNamespace(model=base, lm_head=head)
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(unwrap_model=lambda m: m), model=model, temperature=1.0,
    )
    ids = torch.tensor([[0, 0, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    scored = _ref_logps_liger_fused(trainer, ids, mask, 2)
    assert scored.dtype == torch.float32
    assert torch.equal(base.positions, torch.tensor([[0, 0, 0, 1, 2, 3], [0, 0, 1, 2, 3, 4]]))
    for row, start in [(0, 2), (1, 1)]:
        single = _ref_logps_liger_fused(trainer, ids[row:row+1, start:], mask[row:row+1, start:], 2)
        torch.testing.assert_close(scored[row:row+1], single, rtol=0, atol=0)
