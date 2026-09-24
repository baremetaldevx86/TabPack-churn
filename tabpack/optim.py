"""Member-separable AdamW for first-axis packed parameter tensors.

This eager implementation vectorizes across members, with one moment pair and
one update counter per parameter/member.  It intentionally omits AMSGrad,
capturable, fused, differentiable, and sparse-gradient modes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .models import OrdinaryMLP, PackedMLP

MemberValue = float | Sequence[float] | Tensor


def _values(value: MemberValue, count: int, name: str) -> tuple[float, ...]:
    if isinstance(value, Tensor):
        if value.ndim == 0:
            raw = [value.item()] * count
        elif value.ndim == 1 and value.numel() == count:
            raw = value.detach().cpu().tolist()
        else:
            raise ValueError(f"{name} must be scalar or a length-{count} vector")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        raw = [value] * count
    else:
        try:
            raw = list(value)
        except TypeError as error:
            raise TypeError(f"{name} must be scalar or a member vector") from error
        if len(raw) != count:
            raise ValueError(f"{name} must be scalar or a length-{count} vector")
    result = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} values must be real numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} values must be finite")
        if name in {"beta1", "beta2"}:
            valid = 0 <= number < 1
        elif name == "eps":
            valid = number > 0
        else:
            valid = number >= 0
        if not valid:
            raise ValueError(f"invalid {name} value: {number}")
        result.append(number)
    return tuple(result)


def _boolean_mask(value: Any, shape: tuple[int, ...], device: torch.device, name: str) -> Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.dtype != torch.bool:
        raise TypeError(f"{name} must be boolean")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    return tensor


class PackedAdamW(Optimizer):
    """AdamW with scalar or member-specific hyperparameters.

    All parameters have shape ``[M, ...]`` with a common positive ``M``.
    Parameter groups accept ``lr``, ``weight_decay``, ``eps``, and ``betas``;
    each of the two beta entries can itself be a member vector.

    Optional group ``mask`` is a boolean parameter-shaped ownership mask
    (such groups contain exactly one parameter).  Optional ``member_mask`` is
    boolean ``[M]``.  ``step(active=...)`` additionally pauses selected members,
    without changing their parameters, moments, or counters.  A present zero
    gradient still updates; ``grad=None`` skips a whole parameter.

    Pass ``model.optimizer_groups(weight_decay)`` for a PackedMLP to preserve
    structural padding and the experiment's zero-bias-decay policy.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        lr: MemberValue = 1e-3,
        betas: tuple[MemberValue, MemberValue] = (0.9, 0.999),
        eps: MemberValue = 1e-8,
        weight_decay: MemberValue = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.member_count: int | None = None
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        # Normalize before handing the group to Optimizer, retaining its
        # duplicate-parameter and leaf checks.
        group = dict(param_group)
        parameters = group["params"]
        if isinstance(parameters, Tensor):
            parameters = [parameters]
        else:
            parameters = list(parameters)
        if not parameters:
            raise ValueError("packed parameter groups cannot be empty")
        count = self.member_count
        for parameter in parameters:
            if not isinstance(parameter, Tensor) or parameter.ndim < 1:
                raise ValueError("packed parameters must have a leading member dimension")
            if parameter.shape[0] == 0:
                raise ValueError("packed parameters need at least one member")
            if not parameter.is_floating_point():
                raise TypeError("PackedAdamW supports real floating-point parameters only")
            if count is None:
                count = parameter.shape[0]
            if parameter.shape[0] != count:
                raise ValueError("all packed parameters must have the same member count")
        assert count is not None
        group["params"] = parameters
        for name in ("lr", "weight_decay", "eps"):
            group[name] = _values(group.get(name, self.defaults[name]), count, name)
        betas = group.get("betas", self.defaults["betas"])
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError("betas must contain beta1 and beta2")
        group["betas"] = (_values(betas[0], count, "beta1"), _values(betas[1], count, "beta2"))
        if group.get("mask") is not None:
            if len(parameters) != 1:
                raise ValueError("a group with a slot mask must contain exactly one parameter")
            parameter = parameters[0]
            group["mask"] = _boolean_mask(
                group["mask"], tuple(parameter.shape), parameter.device, "mask"
            ).clone()
        if group.get("member_mask") is not None:
            group["member_mask"] = _boolean_mask(
                group["member_mask"], (count,), parameters[0].device, "member_mask"
            ).clone()
        super().add_param_group(group)
        self.member_count = count

    @torch.no_grad()
    def step(self, closure: Any = None, *, active: Tensor | Sequence[bool] | None = None) -> Any:
        """Update active rows; return the closure result when one is supplied."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        count = self.member_count
        assert count is not None
        for group in self.param_groups:
            # Revalidate mutable group hyperparameters, including scheduled lr.
            lr_values = _values(group["lr"], count, "lr")
            wd_values = _values(group["weight_decay"], count, "weight_decay")
            eps_values = _values(group["eps"], count, "eps")
            beta1_values = _values(group["betas"][0], count, "beta1")
            beta2_values = _values(group["betas"][1], count, "beta2")
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("PackedAdamW does not support sparse gradients")
                row_mask = torch.ones(count, dtype=torch.bool, device=parameter.device)
                if active is not None:
                    row_mask &= _boolean_mask(active, (count,), parameter.device, "active")
                if group.get("member_mask") is not None:
                    row_mask &= _boolean_mask(
                        group["member_mask"], (count,), parameter.device, "member_mask"
                    )
                slot_mask = group.get("mask")
                if slot_mask is not None:
                    slot_mask = _boolean_mask(
                        slot_mask, tuple(parameter.shape), parameter.device, "mask"
                    )
                    row_mask &= slot_mask.reshape(count, -1).any(dim=1)
                if not bool(row_mask.any()):
                    continue

                state = self.state[parameter]
                if not state:
                    state["step"] = torch.zeros(count, dtype=torch.int64, device=parameter.device)
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                step = state["step"]
                average = state["exp_avg"]
                average_sq = state["exp_avg_sq"]
                step.add_(row_mask.to(dtype=step.dtype))
                broadcast_shape = (count,) + (1,) * (parameter.ndim - 1)
                # Bias correction uses double precision, like Python scalar
                # beta**step in non-capturable torch.optim.AdamW.  Arithmetic
                # on parameters and moments remains in their original dtype.
                beta1_double = torch.tensor(beta1_values, device=parameter.device, dtype=torch.float64)
                beta2_double = torch.tensor(beta2_values, device=parameter.device, dtype=torch.float64)
                safe_step = step.clamp_min(1).to(dtype=torch.float64)
                correction1 = 1.0 - beta1_double.pow(safe_step)
                correction2 = (1.0 - beta2_double.pow(safe_step)).sqrt()

                def vector(values: Any) -> Tensor:
                    return torch.as_tensor(values, dtype=parameter.dtype, device=parameter.device).reshape(
                        broadcast_shape
                    )

                beta2 = vector(beta2_values)
                # Compute 1-beta in Python double before casting, matching the
                # reference's scalar alpha/lerp conversion for float32 moments.
                next_average = torch.lerp(average, gradient, vector([1.0 - b for b in beta1_values]))
                next_average_sq = average_sq * beta2
                next_average_sq.add_(gradient.square() * vector([1.0 - b for b in beta2_values]))
                denominator = next_average_sq.sqrt() / vector(correction2)
                denominator.add_(vector(eps_values))
                step_size = torch.tensor(lr_values, device=parameter.device, dtype=torch.float64)
                step_size = vector(step_size / correction1)
                decay = vector([1.0 - lr * wd for lr, wd in zip(lr_values, wd_values)])
                next_parameter = parameter * decay - (next_average / denominator) * step_size
                effective_mask = row_mask.reshape(broadcast_shape)
                if slot_mask is not None:
                    effective_mask = effective_mask & slot_mask
                average.copy_(torch.where(effective_mask, next_average, average))
                average_sq.copy_(torch.where(effective_mask, next_average_sq, average_sq))
                parameter.copy_(torch.where(effective_mask, next_parameter, parameter))
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        # Optimizer's generic loader can preserve/cast step tensors differently
        # across Torch versions.  Packed counters are always local int64 [M].
        for parameter, state in self.state.items():
            if "step" in state:
                state["step"] = state["step"].to(device=parameter.device, dtype=torch.int64)


def make_optimizer(
    model: OrdinaryMLP | PackedMLP,
    *,
    lr: MemberValue = 1e-3,
    weight_decay: MemberValue = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Optimizer:
    """Construct the correct optimizer with zero decay on every model bias."""

    if isinstance(model, PackedMLP):
        return PackedAdamW(model.optimizer_groups(weight_decay), lr=lr, betas=betas, eps=eps)
    if not isinstance(model, OrdinaryMLP):
        raise TypeError("model must be an OrdinaryMLP or PackedMLP")
    scalar_lr = _values(lr, 1, "lr")[0]
    scalar_wd = _values(weight_decay, 1, "weight_decay")[0]
    groups = [
        {"params": [parameter for name, parameter in model.named_parameters() if "bias" not in name],
         "weight_decay": scalar_wd},
        {"params": [parameter for name, parameter in model.named_parameters() if "bias" in name],
         "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=scalar_lr, betas=betas, eps=eps, foreach=False, fused=False)


__all__ = ["PackedAdamW", "make_optimizer"]
