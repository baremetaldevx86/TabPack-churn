"""Independent tiny Linear oracles for packed layout, dropout, and loss."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tabpack.models import OrdinaryMLP, PackedMLP, packed_bce_loss


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        yield


def reference_members(depths, widths, *, features=3, dtype=torch.float64, device="cpu"):
    """Independent nn.Linear oracles; never call a production model forward."""

    generator = torch.Generator(device=device).manual_seed(819)
    members = []
    for depth, width in zip(depths, widths):
        layers = [nn.Linear(features, width, dtype=dtype, device=device)]
        layers += [nn.Linear(width, width, dtype=dtype, device=device) for _ in range(depth - 1)]
        layers += [nn.Linear(width, 1, dtype=dtype, device=device)]
        for layer in layers:
            with torch.no_grad():
                layer.weight.copy_(
                    torch.rand(layer.weight.shape, generator=generator, dtype=dtype, device=device)
                    * 0.6 - 0.2
                )
                layer.bias.copy_(
                    torch.rand(layer.bias.shape, generator=generator, dtype=dtype, device=device)
                    * 0.2 + 0.1
                )
        members.append(layers)
    return members


def oracle_forward(layers, x, *, dropout=0.0, masks=None):
    hidden = x
    for index, layer in enumerate(layers[:-1]):
        hidden = torch.relu(layer(hidden))
        if dropout and masks is not None:
            hidden = hidden * masks[index].to(dtype=x.dtype) / (1 - dropout)
    return layers[-1](hidden)[..., 0]


def copy_references(pack, references):
    # Explicit test-local [out,in] slice mapping, independent of copy/extract APIs.
    with torch.no_grad():
        for member, layers in enumerate(references):
            width = layers[0].out_features
            for index, layer in enumerate(layers[:-1]):
                pack.hidden_weight[index][member, :width, :layer.in_features].copy_(layer.weight)
                pack.hidden_bias[index][member, :width].copy_(layer.bias)
            pack.output_weight[member, :, :width].copy_(layers[-1].weight)
            pack.output_bias[member].copy_(layers[-1].bias)


@pytest.mark.parametrize(
    ("depths", "widths"),
    [([1], [1]), ([2, 2, 2], [3, 3, 3]), ([2, 2, 2], [1, 4, 2]),
     ([1, 3, 2], [3, 3, 3]), ([1, 3, 2, 3], [2, 1, 4, 3])],
)
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_matches_independent_linear_oracle(depths, widths, shared, dtype):
    references = reference_members(depths, widths, dtype=dtype)
    pack = PackedMLP(3, depths=depths, widths=widths, dropouts=[0.2] * len(depths), dtype=dtype)
    copy_references(pack, references)
    pack.eval()
    batch = 1 if len(depths) == 1 else 5
    x = torch.randn(len(depths), batch, 3, dtype=dtype)
    if shared:
        x = x[0]
    expected = torch.stack([
        oracle_forward(layers, x if shared else x[member])
        for member, layers in enumerate(references)
    ])
    actual = pack(x)
    assert actual.shape == (len(depths), batch)
    tolerance = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual, expected, **tolerance)
    torch.testing.assert_close(pack(x), actual, rtol=0, atol=0)
    hidden = pack.forward_features(x)
    for member, width in enumerate(widths):
        assert torch.count_nonzero(hidden[member, :, width:]) == 0


def test_ordinary_forward_matches_independent_linear_oracle():
    layers = reference_members([3], [4])[0]
    model = OrdinaryMLP(3, depth=3, width=4, dropout=0.2, dtype=torch.float64).eval()
    with torch.no_grad():
        for destination, source in zip(list(model.layers) + [model.head], layers):
            destination.weight.copy_(source.weight)
            destination.bias.copy_(source.bias)
    x = torch.randn(5, 3, dtype=torch.float64)
    torch.testing.assert_close(model(x), oracle_forward(layers, x), rtol=0, atol=0)


def test_exact_signed_logits_and_controlled_dropout():
    pack = PackedMLP(2, depths=[1], widths=[2], dropouts=[0.5], dtype=torch.float64)
    with torch.no_grad():
        pack.hidden_weight[0].copy_(torch.tensor([[[1., -2.], [-3., 1.]]]))
        pack.hidden_bias[0].copy_(torch.tensor([[0.5, -0.5]]))
        pack.output_weight.copy_(torch.tensor([[[2., -1.]]]))
        pack.output_bias.fill_(0.25)
    x = torch.tensor([[-1., 2.], [0., 0.], [2., -1.]], dtype=torch.float64)
    mask = torch.tensor([[[True, False], [False, True], [True, True]]])
    pack.eval()
    expected = torch.tensor([[-4.25, 1.25, 9.25]], dtype=torch.float64)
    torch.testing.assert_close(pack(x), expected, rtol=0, atol=0)
    torch.testing.assert_close(pack(x, dropout_masks=[mask]), expected, rtol=0, atol=0)
    pack.train()
    expected = torch.tensor([[0.25, 0.25, 18.25]], dtype=torch.float64)
    torch.testing.assert_close(pack(x, dropout_masks=[mask]), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dropouts", [[0.0, 0.25, 0.5, 0.25], [0.0, 0.1, 0.2, 0.1]])
def test_heterogeneous_controlled_dropout_and_skip_depth(dropouts):
    depths, widths = [1, 3, 2, 3], [2, 1, 4, 3]
    references = reference_members(depths, widths)
    pack = PackedMLP(3, depths=depths, widths=widths, dropouts=dropouts, dtype=torch.float64)
    copy_references(pack, references)
    x = torch.randn(4, 5, 3, dtype=torch.float64)
    masks = [torch.rand(4, 5, 4) > 0.4 for _ in range(3)]
    masks[1][0] = False
    masks[2][0] = False
    expected = torch.stack([
        oracle_forward(
            layers, x[member], dropout=dropouts[member],
            masks=[mask[member, :, :widths[member]] for mask in masks],
        )
        for member, layers in enumerate(references)
    ])
    torch.testing.assert_close(pack(x, dropout_masks=masks), expected, rtol=1e-10, atol=1e-12)


def test_dirty_padding_is_inert_and_gradients_are_zero():
    depths, widths = [1, 3, 2], [2, 5, 3]
    pack = PackedMLP(3, depths=depths, widths=widths, dropouts=[0.0] * 3, dtype=torch.float64)
    references = reference_members(depths, widths)
    copy_references(pack, references)
    x = torch.randn(3, 4, 3, dtype=torch.float64)
    expected = torch.stack([oracle_forward(layers, x[m]) for m, layers in enumerate(references)])
    # Build ownership independently so the production mask cannot hide its own bug.
    masks = {}
    for index, (weight, bias) in enumerate(zip(pack.hidden_weight, pack.hidden_bias)):
        wm, bm = torch.zeros_like(weight, dtype=torch.bool), torch.zeros_like(bias, dtype=torch.bool)
        for member, (depth, width) in enumerate(zip(depths, widths)):
            if index < depth:
                wm[member, :width, :3 if index == 0 else width] = True
                bm[member, :width] = True
        masks[weight], masks[bias] = wm, bm
    head_mask = torch.zeros_like(pack.output_weight, dtype=torch.bool)
    for member, width in enumerate(widths):
        head_mask[member, :, :width] = True
    masks[pack.output_weight] = head_mask
    for poison in (16.0, -8.0):
        with torch.no_grad():
            for parameter, mask in masks.items():
                parameter[~mask] = poison
        torch.testing.assert_close(pack(x), expected, rtol=1e-10, atol=1e-12)
    pack(x).sum().backward()
    for parameter, mask in masks.items():
        assert torch.count_nonzero(parameter.grad[~mask]) == 0
        torch.testing.assert_close(pack.parameter_masks()[parameter], mask)


def test_member_isolation_layout_batch_partition_and_order():
    depths, widths = [1, 3, 2], [2, 5, 3]
    references = reference_members(depths, widths)
    pack = PackedMLP(3, depths=depths, widths=widths, dropouts=[0.0] * 3, dtype=torch.float64)
    copy_references(pack, references)
    x = torch.randn(3, 3, 5, dtype=torch.float64).transpose(1, 2)
    assert not x.is_contiguous()
    baseline = pack(x)
    torch.testing.assert_close(pack(x.contiguous()), baseline)
    split = torch.cat([pack(x[:, :2]), pack(x[:, 2:])], dim=1)
    torch.testing.assert_close(split, baseline)
    altered_x = x.clone()
    altered_x[1] += 2
    altered = pack(altered_x)
    torch.testing.assert_close(altered[[0, 2]], baseline[[0, 2]], rtol=0, atol=0)
    assert not torch.equal(altered[1], baseline[1])
    with torch.no_grad():
        pack.output_bias[1].add_(0.5)
    altered = pack(x)
    torch.testing.assert_close(altered[[0, 2]], baseline[[0, 2]], rtol=0, atol=0)
    torch.testing.assert_close(altered[1], baseline[1] + 0.5)
    permutation = [2, 0, 1]
    reordered = PackedMLP(
        3, depths=[depths[m] for m in permutation], widths=[widths[m] for m in permutation],
        dropouts=[0.0] * 3, dtype=torch.float64,
    )
    copy_references(reordered, [references[m] for m in permutation])
    torch.testing.assert_close(reordered(x[permutation]), baseline[permutation])


def test_member_snapshots_state_dict_and_accounting():
    ordinary = [
        OrdinaryMLP(3, depth=d, width=w, dropout=p, dtype=torch.float64)
        for d, w, p in [(1, 2, 0.0), (3, 5, 0.2), (2, 3, 0.1)]
    ]
    rng_before = torch.get_rng_state().clone()
    pack = PackedMLP.from_members(ordinary).eval()
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    x = torch.randn(4, 3, dtype=torch.float64)
    rng_before = torch.get_rng_state().clone()
    snapshots = [pack.extract_member(index) for index in range(3)]
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    expected = torch.stack([member.eval()(x) for member in ordinary])
    torch.testing.assert_close(pack(x), expected)
    torch.testing.assert_close(torch.stack([member(x) for member in snapshots]), expected)
    assert pack.logical_parameter_count == sum(p.numel() for m in ordinary for p in m.parameters())
    assert pack.stored_parameter_count == sum(p.numel() for p in pack.parameters())
    assert pack.stored_parameter_count > pack.logical_parameter_count
    restored = PackedMLP(3, depths=[1, 3, 2], widths=[2, 5, 3], dropouts=[0, 0.2, 0.1],
                         dtype=torch.float64).eval()
    restored.load_state_dict(copy.deepcopy(pack.state_dict()))
    torch.testing.assert_close(restored(x), expected)
    with torch.no_grad():
        pack.output_bias[0].add_(10)
    torch.testing.assert_close(snapshots[0](x), expected[0])


def test_production_sized_float32_and_member_dropout_draws():
    pack = PackedMLP(11, depths=[1, 2, 3, 2], widths=[32, 64, 96, 32], dropouts=[0, .1, .2, .1])
    x = torch.randn(4, 7, 11)
    loss, _ = packed_bce_loss(pack(x), torch.randint(2, (4, 7)))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in pack.parameters())
    # Positive, identical activations isolate independently drawn dropout masks.
    stochastic = PackedMLP(1, depths=[1, 1], widths=[64, 64], dropouts=[0.5, 0.5])
    with torch.no_grad():
        stochastic.hidden_weight[0].zero_()
        stochastic.hidden_bias[0].fill_(1)
    hidden = stochastic.forward_features(torch.ones(16, 1))
    assert set(hidden.unique().tolist()) == {0.0, 2.0}
    assert not torch.equal(hidden[0], hidden[1])
    stochastic.eval()
    assert torch.equal(stochastic.forward_features(torch.ones(16, 1)), torch.ones_like(hidden))


@pytest.mark.parametrize("batch", [1, 3, 7])
def test_loss_member_sum_and_direct_gradient_pack_size_invariance(batch):
    logits = torch.randn(4, batch, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(2, (4, batch)).to(torch.float64)
    total, per_member = packed_bce_loss(logits, targets)
    references = [logits[m].detach().clone().requires_grad_() for m in range(4)]
    means = [F.binary_cross_entropy_with_logits(z, targets[m]) for m, z in enumerate(references)]
    torch.testing.assert_close(per_member, torch.stack(means))
    torch.testing.assert_close(total, torch.stack(means).sum())
    total.backward()
    for member, mean in enumerate(means):
        mean.backward()
        torch.testing.assert_close(logits.grad[member], references[member].grad)
    assert not torch.allclose(total, torch.stack(means).mean())
    torch.testing.assert_close(logits.grad, (logits.detach().sigmoid() - targets) / batch)
    singleton = logits[0:1].detach().clone().requires_grad_()
    packed_bce_loss(singleton, targets[0])[0].backward()
    torch.testing.assert_close(singleton.grad[0], logits.grad[0])


def test_loss_extreme_logits_and_shared_targets():
    logits = torch.tensor([[-1000., -100., 0., 100., 1000.]] * 2, requires_grad=True)
    y = torch.tensor([1, 0, 1, 0, 1])
    loss, members = packed_bce_loss(logits[..., None], y)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(members[0], F.binary_cross_entropy_with_logits(logits[0], y.float()))
    with pytest.raises(ValueError, match="nonempty"):
        packed_bce_loss(torch.empty(2, 0), torch.empty(0))
    with pytest.raises(ValueError, match="zeroes and ones"):
        packed_bce_loss(torch.ones(2, 3), torch.tensor([0., .3, 1.]))
    with pytest.raises(ValueError, match="match"):
        packed_bce_loss(torch.ones(2, 3), torch.ones(3, 2))


@pytest.mark.parametrize("kwargs", [
    {"depths": [], "widths": [], "dropouts": []},
    {"depths": [1], "widths": [2, 3], "dropouts": [0]},
    {"depths": [0], "widths": [2], "dropouts": [0]},
    {"depths": [True], "widths": [2], "dropouts": [0]},
    {"depths": [1], "widths": [2.5], "dropouts": [0]},
    {"depths": [1], "widths": [2], "dropouts": [1]},
    {"depths": [1], "widths": [2], "dropouts": [float("nan")]},
])
def test_invalid_member_configurations(kwargs):
    with pytest.raises((TypeError, ValueError)):
        PackedMLP(3, **kwargs)


def test_input_validation_and_multioutput_shapes():
    pack = PackedMLP(3, depths=[1, 2], widths=[2, 4], dropouts=[0, 0], output_dim=2)
    ordinary = OrdinaryMLP(3, depth=2, width=4, dropout=0, output_dim=2)
    assert pack(torch.ones(1, 3)).shape == (2, 1, 2)
    assert ordinary(torch.ones(1, 3)).shape == (1, 2)
    for x in [torch.ones(3), torch.ones(1, 1, 1, 3), torch.ones(3, 2, 3), torch.ones(2, 4)]:
        with pytest.raises(ValueError):
            pack(x)
    with pytest.raises(TypeError, match="floating"):
        pack(torch.ones(2, 3, dtype=torch.long))
    with pytest.raises(TypeError, match="dtype"):
        pack(torch.ones(2, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match="rank"):
        ordinary(torch.ones(2, 1, 3))
    with pytest.raises(ValueError, match="layer entries"):
        pack(torch.ones(2, 3), dropout_masks=[])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
def test_cuda_forward_backward_matches_same_device_linear_oracle():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        depths, widths = [1, 3, 2], [2, 5, 3]
        references = reference_members(depths, widths, dtype=torch.float32, device="cuda")
        pack = PackedMLP(3, depths=depths, widths=widths, dropouts=[0, 0, 0]).to("cuda")
        copy_references(pack, references)
        x = torch.randn(3, 7, 3, device="cuda")
        expected = torch.stack([oracle_forward(layers, x[m]) for m, layers in enumerate(references)])
        torch.testing.assert_close(pack(x), expected, rtol=1e-4, atol=1e-5)
        packed_bce_loss(pack(x), torch.randint(2, (3, 7), device="cuda"))[0].backward()
        assert all(torch.isfinite(p.grad).all() for p in pack.parameters())
        assert all(buffer.device.type == "cuda" for buffer in pack.buffers())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
