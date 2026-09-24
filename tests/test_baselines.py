"""Focused tests for the reusable SimpleMLP baseline."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from tabpack import SimpleMLP


def test_simple_mlp_defaults_are_binary_logits():
    model = SimpleMLP(5)
    assert model.input_dim == 5
    assert model.hidden_dim == 64
    assert model.depth == 2
    assert model.dropout == 0.1
    output = model(torch.randn(7, 5))
    assert output.shape == (7,)


def test_simple_mlp_supports_training_and_state_round_trip():
    torch.manual_seed(12)
    model = SimpleMLP(4, hidden_dim=8, depth=1, dropout=0.0)
    features = torch.randn(10, 4)
    labels = torch.randint(0, 2, (10,), dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    loss = F.binary_cross_entropy_with_logits(model(features), labels)
    loss.backward()
    optimizer.step()
    assert all(torch.isfinite(value).all() for value in model.parameters())
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())

    restored = SimpleMLP(4, hidden_dim=8, depth=1, dropout=0.0)
    restored.load_state_dict(model.state_dict())
    model.eval()
    restored.eval()
    torch.testing.assert_close(model(features), restored(features))


def test_simple_mlp_supports_multi_output_logits():
    model = SimpleMLP(3, hidden_dim=6, depth=3, dropout=0.0, output_dim=4)
    assert model(torch.randn(2, 3)).shape == (2, 4)
