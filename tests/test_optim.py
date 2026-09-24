"""Short independent torch.optim.AdamW oracles, including real model gradients."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tabpack.models import OrdinaryMLP, PackedMLP, packed_bce_loss
from tabpack.optim import PackedAdamW, make_optimizer


def assert_state_matches(optimizer, packed, references, optimizers, *, dtype):
    tolerance = dict(rtol=2e-5, atol=2e-7) if dtype == torch.float32 else dict(rtol=1e-11, atol=1e-13)
    for member, (reference, ordinary) in enumerate(zip(references, optimizers)):
        state = optimizer.state[packed]
        expected = ordinary.state[reference]
        torch.testing.assert_close(packed[member], reference, **tolerance)
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(state[name][member], expected[name], **tolerance)
        assert int(state["step"][member]) == int(expected["step"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("members", [1, 3, 4])
@pytest.mark.parametrize("shape", [(2, 5), (3,), ()])
def test_seven_steps_match_independent_adamw(dtype, members, shape):
    count = 1
    for size in shape:
        count *= size
    initial = torch.arange(members * count, dtype=dtype).reshape((members,) + shape) / 10 + 0.25
    packed = nn.Parameter(initial.clone())
    references = [nn.Parameter(row.clone()) for row in initial]
    lrs = [0.01, 0.003, 0.02, 0.007][:members]
    wds = [0.0, 0.1, 0.2, 0.05][:members]
    beta1s = [0.9, 0.8, 0.7, 0.85][:members]
    beta2s = [0.999, 0.95, 0.9, 0.99][:members]
    eps = [1e-8, 1e-2, 1e-5, 1e-6][:members]
    optimizer = PackedAdamW([packed], lr=lrs, weight_decay=wds, betas=(beta1s, beta2s), eps=eps)
    optimizers = [
        torch.optim.AdamW([parameter], lr=lr, weight_decay=wd, betas=(b1, b2), eps=e,
                         foreach=False, fused=False)
        for parameter, lr, wd, b1, b2, e in zip(references, lrs, wds, beta1s, beta2s, eps)
    ]
    base = torch.arange(members * count, dtype=dtype).reshape_as(initial) / 7 - 0.7
    if count > 1:
        base.reshape(members, -1)[:, 0] = 0
        base.reshape(members, -1)[:, 1] = 1e-9
    gradients = [base, -0.3 * base + 0.02, torch.zeros_like(base), 2 * base - 0.05,
                 -base, 0.1 * base + 0.03, base]
    for gradient in gradients:
        packed.grad = gradient.clone()
        optimizer.step()
        for member, ordinary in enumerate(optimizers):
            references[member].grad = gradient[member].clone()
            ordinary.step()
        assert_state_matches(optimizer, packed, references, optimizers, dtype=dtype)
    if count > 1:
        assert torch.count_nonzero(optimizer.state[packed]["exp_avg_sq"].reshape(members, -1)[:, 1])


def test_first_step_golden_none_and_zero_gradients():
    parameter = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    optimizer = PackedAdamW([parameter], lr=0.01, weight_decay=0.1)
    optimizer.step()
    assert parameter.item() == 2
    assert not optimizer.state
    parameter.grad = torch.tensor([0.5], dtype=torch.float64)
    optimizer.step()
    assert parameter.item() == pytest.approx(1.9880000002, rel=0, abs=1e-13)
    assert optimizer.state[parameter]["exp_avg"].item() == pytest.approx(0.05)
    assert optimizer.state[parameter]["exp_avg_sq"].item() == pytest.approx(0.00025)
    before = parameter.clone()
    saved = copy.deepcopy(optimizer.state[parameter])
    parameter.grad = None
    optimizer.step()
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    for key in saved:
        torch.testing.assert_close(optimizer.state[parameter][key], saved[key], rtol=0, atol=0)
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert int(optimizer.state[parameter]["step"][0]) == 2
    assert parameter.item() == pytest.approx(1.9793114178480031, rel=0, abs=1e-13)


def test_zero_lr_decoupled_decay_and_bias_exemption():
    weight = nn.Parameter(torch.full((2, 2), 2.0, dtype=torch.float64))
    bias = nn.Parameter(torch.full((2, 2), 2.0, dtype=torch.float64))
    optimizer = PackedAdamW([
        {"params": [weight], "weight_decay": [0.1, 0.2]},
        {"params": [bias], "weight_decay": 0.0},
    ], lr=[0.01, 0.0])
    weight.grad, bias.grad = torch.zeros_like(weight), torch.zeros_like(bias)
    optimizer.step()
    torch.testing.assert_close(weight[0], torch.full_like(weight[0], 1.998), rtol=0, atol=0)
    torch.testing.assert_close(weight[1], torch.full_like(weight[1], 2), rtol=0, atol=0)
    torch.testing.assert_close(bias, torch.full_like(bias, 2), rtol=0, atol=0)
    assert torch.count_nonzero(optimizer.state[weight]["exp_avg"]) == 0
    assert optimizer.state[weight]["step"].tolist() == [1, 1]
    weight.grad.fill_(0.5)
    optimizer.step()
    assert torch.count_nonzero(optimizer.state[weight]["exp_avg"][1]) == 2
    assert optimizer.state[weight]["step"].tolist() == [2, 2]
    torch.testing.assert_close(weight[1], torch.full_like(weight[1], 2), rtol=0, atol=0)


def test_optimizer_factory_exempts_all_biases_and_restores_masked_state():
    ordinary = OrdinaryMLP(3, depth=2, width=4, dropout=0)
    optimizer = make_optimizer(ordinary, lr=.01, weight_decay=.2)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = next(name for name, value in ordinary.named_parameters() if value is parameter)
            assert group["weight_decay"] == (0.0 if "bias" in name else .2)
    pack = PackedMLP(3, depths=[1, 3], widths=[2, 4], dropouts=[0, 0])
    packed_optimizer = make_optimizer(pack, lr=[.01, .02], weight_decay=[.1, .2])
    for parameter in pack.parameters():
        parameter.grad = torch.ones_like(parameter)
    packed_optimizer.step()
    cloned = copy.deepcopy(pack)
    restored = make_optimizer(cloned, lr=1)
    restored.load_state_dict(copy.deepcopy(packed_optimizer.state_dict()))
    for left, right in zip(pack.parameters(), cloned.parameters()):
        left.grad = torch.full_like(left, .3)
        right.grad = left.grad.clone()
    packed_optimizer.step()
    restored.step()
    for left, right in zip(pack.parameters(), cloned.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        for key in packed_optimizer.state[left]:
            torch.testing.assert_close(packed_optimizer.state[left][key], restored.state[right][key],
                                       rtol=0, atol=0)


def test_member_pause_resume_slot_masks_and_all_inactive_state():
    parameter = nn.Parameter(torch.arange(12, dtype=torch.float64).reshape(3, 4) / 5)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True], [False] * 4])
    optimizer = PackedAdamW([{"params": [parameter], "mask": mask}], lr=[.01, .02, .03],
                            weight_decay=.2)
    references = [nn.Parameter(parameter[0, :2].detach().clone()),
                  nn.Parameter(parameter[1].detach().clone())]
    optimizers = [torch.optim.AdamW([p], lr=lr, weight_decay=.2, foreach=False, fused=False)
                  for p, lr in zip(references, [.01, .02])]
    initial = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step(active=[False, False, False])
    assert not optimizer.state
    for step_index in range(5):
        active = [True, step_index not in (1, 2), True]
        parameter.grad = torch.full_like(parameter, .2 + step_index * .1)
        optimizer.step(active=active)
        for member, (reference, ordinary) in enumerate(zip(references, optimizers)):
            reference.grad = torch.full_like(reference, .2 + step_index * .1) if active[member] else None
            ordinary.step()
            width = reference.numel()
            torch.testing.assert_close(parameter[member, :width], reference, rtol=1e-11, atol=1e-13)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[parameter][key][member, :width],
                                           ordinary.state[reference][key], rtol=1e-11, atol=1e-13)
            assert int(optimizer.state[parameter]["step"][member]) == int(ordinary.state[reference]["step"])
        torch.testing.assert_close(parameter[~mask], initial[~mask], rtol=0, atol=0)
        assert optimizer.state[parameter]["step"][2] == 0
        assert torch.count_nonzero(optimizer.state[parameter]["exp_avg"][~mask]) == 0


def test_independent_parameter_counters_missing_gradients_and_resume():
    parameters = [nn.Parameter(torch.ones(3, 2, dtype=torch.float64)),
                  nn.Parameter(torch.ones(3, dtype=torch.float64))]
    optimizer = PackedAdamW(parameters, lr=[.01, .02, .03], weight_decay=[.1, 0, .2])
    for step in range(3):
        parameters[0].grad = torch.full_like(parameters[0], .1 + step)
        parameters[1].grad = torch.full_like(parameters[1], -.2 - step) if step == 1 else None
        optimizer.step(active=[True, step != 2, True])
    assert optimizer.state[parameters[0]]["step"].tolist() == [3, 2, 3]
    assert optimizer.state[parameters[1]]["step"].tolist() == [1, 1, 1]
    checkpoint = copy.deepcopy(optimizer.state_dict())
    restored_parameters = [nn.Parameter(p.detach().clone()) for p in parameters]
    restored = PackedAdamW(restored_parameters)
    restored.load_state_dict(checkpoint)
    for step in range(3, 7):
        for parameter, restored_parameter in zip(parameters, restored_parameters):
            parameter.grad = torch.full_like(parameter, .1 + step)
            restored_parameter.grad = parameter.grad.clone()
        optimizer.step()
        restored.step()
        for parameter, restored_parameter in zip(parameters, restored_parameters):
            torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)
            for key in optimizer.state[parameter]:
                torch.testing.assert_close(optimizer.state[parameter][key],
                                           restored.state[restored_parameter][key], rtol=0, atol=0)


def test_no_cross_member_influence_and_scalar_vector_equivalence():
    first = nn.Parameter(torch.arange(9, dtype=torch.float64).reshape(3, 3) / 10)
    second = nn.Parameter(first.detach().clone())
    scalar = PackedAdamW([first], lr=.01, weight_decay=.2, betas=(.8, .95), eps=.01)
    vector = PackedAdamW([second], lr=[.01] * 3, weight_decay=[.2] * 3,
                         betas=([.8] * 3, [.95] * 3), eps=[.01] * 3)
    gradient = torch.arange(9, dtype=torch.float64).reshape(3, 3) / 7
    first.grad = gradient.clone()
    second.grad = gradient.clone()
    scalar.step()
    vector.step()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    second.grad[1].mul_(100)
    vector.param_groups[0]["lr"] = [.01, .2, .01]
    vector.param_groups[0]["weight_decay"] = [.2, 1.0, .2]
    scalar.step()
    vector.step()
    torch.testing.assert_close(first[[0, 2]], second[[0, 2]], rtol=0, atol=0)
    for key in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(scalar.state[first][key][[0, 2]], vector.state[second][key][[0, 2]],
                                   rtol=0, atol=0)
    assert not torch.equal(first[1], second[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_member_batches_gradients_and_adamw_match_independent_linear_models(dtype):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(78)
        depths, widths = [1, 2, 3], [2, 4, 3]
        pack = PackedMLP(3, depths=depths, widths=widths, dropouts=[0, 0, 0], dtype=dtype)
        references, mappings, optimizers = [], [], []
        lrs, wds = [.01, .003, .02], [.1, 0, .2]
        for member, (depth, width) in enumerate(zip(depths, widths)):
            layers = nn.ModuleList([nn.Linear(3 if layer == 0 else width, width, dtype=dtype)
                                    for layer in range(depth)] + [nn.Linear(width, 1, dtype=dtype)])
            mapping = []
            with torch.no_grad():
                for index, layer in enumerate(layers[:-1]):
                    layer.weight.uniform_(-.2, .3)
                    layer.bias.uniform_(.1, .3)
                    weight_slice = (member, slice(0, width), slice(0, layer.in_features))
                    bias_slice = (member, slice(0, width))
                    pack.hidden_weight[index][weight_slice].copy_(layer.weight)
                    pack.hidden_bias[index][bias_slice].copy_(layer.bias)
                    mapping.extend([(pack.hidden_weight[index], weight_slice, layer.weight),
                                    (pack.hidden_bias[index], bias_slice, layer.bias)])
                pack.output_weight[member, :, :width].copy_(layers[-1].weight)
                pack.output_bias[member].copy_(layers[-1].bias)
                mapping.extend([(pack.output_weight, (member, slice(None), slice(0, width)), layers[-1].weight),
                                (pack.output_bias, member, layers[-1].bias)])
            references.append(layers)
            mappings.append(mapping)
            optimizers.append(torch.optim.AdamW([
                {"params": [layer.weight for layer in layers], "weight_decay": wds[member]},
                {"params": [layer.bias for layer in layers], "weight_decay": 0.0},
            ], lr=lrs[member], foreach=False, fused=False))
        optimizer = make_optimizer(pack, lr=lrs, weight_decay=wds)
        singleton = PackedMLP(3, depths=[depths[0]], widths=[widths[0]], dropouts=[0], dtype=dtype)
        with torch.no_grad():
            for packed_parameter, region, _ in mappings[0]:
                if packed_parameter is pack.output_weight:
                    singleton.output_weight[0].copy_(packed_parameter[region])
                elif packed_parameter is pack.output_bias:
                    singleton.output_bias[0].copy_(packed_parameter[region])
                elif packed_parameter is pack.hidden_weight[0]:
                    singleton.hidden_weight[0][0].copy_(packed_parameter[region])
                elif packed_parameter is pack.hidden_bias[0]:
                    singleton.hidden_bias[0][0].copy_(packed_parameter[region])
        single_optimizer = make_optimizer(singleton, lr=lrs[0], weight_decay=wds[0])
        x = torch.arange(18, dtype=dtype).reshape(6, 3) / 10 - .8
        y = torch.tensor([0, 1, 0, 1, 1, 0], dtype=dtype)
        tolerance = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=1e-9, atol=1e-11)
        for step_index, batch in enumerate([3, 2, 1]):
            indices = torch.stack([torch.arange(6).roll(member + 2 * step_index)[:batch]
                                   for member in range(3)])
            xb, yb = x[indices], y[indices]
            optimizer.zero_grad(set_to_none=True)
            single_optimizer.zero_grad(set_to_none=True)
            actual = pack(xb)
            expected = []
            for member, (layers, ordinary) in enumerate(zip(references, optimizers)):
                ordinary.zero_grad(set_to_none=True)
                hidden = xb[member]
                for layer in layers[:-1]:
                    hidden = torch.relu(layer(hidden))
                expected.append(layers[-1](hidden)[..., 0])
            torch.testing.assert_close(actual, torch.stack(expected), **tolerance)
            loss, per_member = packed_bce_loss(actual, yb)
            loss.backward()
            packed_bce_loss(singleton(xb[0]), yb[0])[0].backward()
            for member, output in enumerate(expected):
                reference_loss = F.binary_cross_entropy_with_logits(output, yb[member])
                torch.testing.assert_close(per_member[member], reference_loss, **tolerance)
                reference_loss.backward()
                for parameter, region, reference in mappings[member]:
                    torch.testing.assert_close(parameter.grad[region], reference.grad, **tolerance)
            optimizer.step()
            single_optimizer.step()
            for member, ordinary in enumerate(optimizers):
                ordinary.step()
                for parameter, region, reference in mappings[member]:
                    torch.testing.assert_close(parameter[region], reference, **tolerance)
                    for key in ("exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(optimizer.state[parameter][key][region],
                                                   ordinary.state[reference][key], **tolerance)
                    assert int(optimizer.state[parameter]["step"][member]) == step_index + 1
            torch.testing.assert_close(singleton(x)[0], pack(x)[0], **tolerance)
            torch.testing.assert_close(singleton.hidden_weight[0].grad[0],
                                       pack.hidden_weight[0].grad[0, :2, :3], **tolerance)
            for parameter, mask in pack.parameter_masks().items():
                assert torch.count_nonzero(parameter[~mask]) == 0
                assert torch.count_nonzero(optimizer.state[parameter]["exp_avg"][~mask]) == 0
        assert optimizer.state[pack.hidden_weight[2]]["step"].tolist() == [0, 0, 3]


@pytest.mark.parametrize("kwargs", [
    {"lr": -1}, {"lr": [0.01]}, {"lr": float("inf")}, {"lr": True},
    {"weight_decay": -1}, {"eps": 0}, {"betas": (1.0, .999)}, {"betas": (.9, -1)},
    {"betas": ([.9, .8, .7], .999)},
])
def test_invalid_hyperparameters(kwargs):
    with pytest.raises((TypeError, ValueError)):
        PackedAdamW([nn.Parameter(torch.ones(2, 3))], **kwargs)


def test_parameter_masks_validation_and_sparse_gradient_rejection():
    with pytest.raises(ValueError, match="same member count"):
        PackedAdamW([nn.Parameter(torch.ones(2, 3)), nn.Parameter(torch.ones(3, 2))])
    parameter = nn.Parameter(torch.ones(2, 3))
    with pytest.raises(ValueError, match="shape"):
        PackedAdamW([{"params": [parameter], "mask": torch.ones(2, dtype=torch.bool)}])
    optimizer = PackedAdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(TypeError, match="boolean"):
        optimizer.step(active=[1, 0])
    parameter.grad = torch.sparse_coo_tensor(
        torch.tensor([[0], [1]]), torch.tensor([1.]), (2, 3), check_invariants=True
    )
    with pytest.raises(RuntimeError, match="sparse"):
        optimizer.step()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
def test_cuda_optimizer_matches_same_device_independent_adamw():
    initial = torch.arange(18, dtype=torch.float32, device="cuda").reshape(3, 2, 3) / 10
    parameter = nn.Parameter(initial.clone())
    references = [nn.Parameter(row.clone()) for row in initial]
    lrs, wds = [.01, .003, .02], [0, .1, .2]
    optimizer = PackedAdamW([parameter], lr=lrs, weight_decay=wds)
    ordinary = [torch.optim.AdamW([p], lr=lr, weight_decay=wd, foreach=False, fused=False)
                for p, lr, wd in zip(references, lrs, wds)]
    for step in range(3):
        gradient = initial * (.1 + step) - .2
        parameter.grad = gradient.clone()
        optimizer.step()
        for member, reference_optimizer in enumerate(ordinary):
            references[member].grad = gradient[member].clone()
            reference_optimizer.step()
        assert_state_matches(optimizer, parameter, references, ordinary, dtype=torch.float32)
