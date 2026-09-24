"""Small, independently authored MLPs used by the reduced TabPack experiment.

The packed model keeps the member axis first throughout its computation.  Its
rectangular tensors are only an execution layout: width and depth masks make
each row behave like an independently parameterized MLP.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _dropout_probability(value: float, name: str = "dropout") -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1)")
    return result


def _check_input(x: Tensor, *, rank: int, feature_count: int, reference: Tensor) -> None:
    if not isinstance(x, Tensor):
        raise TypeError("input must be a torch.Tensor")
    if x.ndim != rank:
        raise ValueError(f"expected a rank-{rank} input, got shape {tuple(x.shape)}")
    if x.shape[-1] != feature_count:
        raise ValueError(
            f"expected {feature_count} input features, got {x.shape[-1]}"
        )
    if not x.is_floating_point():
        raise TypeError("model inputs must have a floating-point dtype")
    if x.device != reference.device:
        raise ValueError(f"input is on {x.device}, but model is on {reference.device}")
    if x.dtype != reference.dtype:
        raise TypeError(f"input dtype {x.dtype} does not match model dtype {reference.dtype}")


class OrdinaryMLP(nn.Module):
    """A conventional binary or multi-output feed-forward MLP.

    ``depth`` is the number of hidden ``Linear -> ReLU -> Dropout`` blocks.
    The module returns logits and never applies a sigmoid.  Binary output has
    shape ``[B]``; a general output has shape ``[B, output_dim]``.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        depth: int,
        width: int,
        dropout: float,
        output_dim: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.depth = _positive_int(depth, "depth")
        self.width = _positive_int(width, "width")
        self.output_dim = _positive_int(output_dim, "output_dim")
        self.dropout = _dropout_probability(dropout)
        if not dtype.is_floating_point:
            raise TypeError("model dtype must be floating point")

        layers: list[nn.Linear] = []
        fan_in = self.input_dim
        for _ in range(self.depth):
            layers.append(nn.Linear(fan_in, self.width, device=device, dtype=dtype))
            fan_in = self.width
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(self.width, self.output_dim, device=device, dtype=dtype)

    @property
    def hidden_layers(self) -> nn.ModuleList:
        """Readable alias for callers inspecting the hidden linear layers."""

        return self.layers

    def forward(self, x: Tensor) -> Tensor:
        _check_input(x, rank=2, feature_count=self.input_dim, reference=self.head.weight)
        hidden = x
        for layer in self.layers:
            hidden = F.relu(layer(hidden))
            if self.dropout:
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        logits = self.head(hidden)
        return logits[..., 0] if self.output_dim == 1 else logits


class PackedMLP(nn.Module):
    """A heterogeneous MLP pack with member-first ``[M, B, ...]`` semantics.

    Each member has a constant hidden width, but members may have different
    widths, depths, and dropout probabilities.  Rectangular layer parameters
    use ``max(widths)`` and inactive coordinates are masked before activation,
    after dropout, and at the output head.  ``x`` may be shared ``[B, F]`` or
    member-specific ``[M, B, F]`` input.

    ``dropout_masks`` is an optional testing/replay seam.  When supplied it is
    a sequence with one entry per maximum hidden layer; each entry is a keep
    mask of shape ``[M, B, max_width]``.  It is not a probability mask and is
    scaled with each member's configured inverted-dropout factor.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        depths: Sequence[int],
        widths: Sequence[int],
        dropouts: Sequence[float],
        output_dim: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.output_dim = _positive_int(output_dim, "output_dim")
        if not dtype.is_floating_point:
            raise TypeError("model dtype must be floating point")

        depths_list = list(depths)
        widths_list = list(widths)
        dropouts_list = list(dropouts)
        if not depths_list:
            raise ValueError("packed model needs at least one member")
        if not (len(depths_list) == len(widths_list) == len(dropouts_list)):
            raise ValueError("depths, widths, and dropouts must have equal lengths")
        self.member_count = len(depths_list)
        checked_depths = [_positive_int(value, "depth") for value in depths_list]
        checked_widths = [_positive_int(value, "width") for value in widths_list]
        checked_dropouts = [
            _dropout_probability(value, f"dropouts[{index}]")
            for index, value in enumerate(dropouts_list)
        ]
        self.max_depth = max(checked_depths)
        self.max_width = max(checked_widths)

        self.register_buffer("depths", torch.tensor(checked_depths, dtype=torch.long, device=device))
        self.register_buffer("widths", torch.tensor(checked_widths, dtype=torch.long, device=device))
        self.register_buffer(
            "dropouts",
            torch.tensor(checked_dropouts, dtype=dtype, device=device),
        )
        self.register_buffer(
            "width_mask",
            torch.arange(self.max_width, device=device).view(1, -1)
            < torch.tensor(checked_widths, dtype=torch.long, device=device).view(-1, 1),
        )

        self.hidden_weight = nn.ParameterList()
        self.hidden_bias = nn.ParameterList()
        for layer_index in range(self.max_depth):
            fan_in = self.input_dim if layer_index == 0 else self.max_width
            weight = torch.zeros(
                self.member_count,
                self.max_width,
                fan_in,
                device=device,
                dtype=dtype,
            )
            bias = torch.zeros(self.member_count, self.max_width, device=device, dtype=dtype)
            with torch.no_grad():
                for member, width in enumerate(checked_widths):
                    if layer_index >= checked_depths[member]:
                        continue
                    actual_fan_in = self.input_dim if layer_index == 0 else width
                    nn.init.kaiming_uniform_(
                        weight[member, :width, :actual_fan_in], a=math.sqrt(5)
                    )
                    bound = 1.0 / math.sqrt(actual_fan_in)
                    bias[member, :width].uniform_(-bound, bound)
            self.hidden_weight.append(nn.Parameter(weight))
            self.hidden_bias.append(nn.Parameter(bias))

        output_weight = torch.zeros(
            self.member_count,
            self.output_dim,
            self.max_width,
            device=device,
            dtype=dtype,
        )
        output_bias = torch.zeros(
            self.member_count, self.output_dim, device=device, dtype=dtype
        )
        with torch.no_grad():
            for member, width in enumerate(checked_widths):
                nn.init.kaiming_uniform_(output_weight[member, :, :width], a=math.sqrt(5))
                bound = 1.0 / math.sqrt(width)
                output_bias[member].uniform_(-bound, bound)
        self.output_weight = nn.Parameter(output_weight)
        self.output_bias = nn.Parameter(output_bias)

    @property
    def hidden_weights(self) -> nn.ParameterList:
        return self.hidden_weight

    @property
    def hidden_biases(self) -> nn.ParameterList:
        return self.hidden_bias

    @property
    def member_specs(self) -> tuple[tuple[int, int, float], ...]:
        return tuple(
            (int(depth), int(width), float(dropout))
            for depth, width, dropout in zip(
                self.depths.tolist(), self.widths.tolist(), self.dropouts.tolist()
            )
        )

    def _prepare_input(self, x: Tensor) -> Tensor:
        reference = self.output_weight
        if not isinstance(x, Tensor):
            raise TypeError("input must be a torch.Tensor")
        if x.ndim == 2:
            _check_input(x, rank=2, feature_count=self.input_dim, reference=reference)
            return x.unsqueeze(0).expand(self.member_count, -1, -1)
        if x.ndim != 3:
            raise ValueError(f"expected [B, F] or [M, B, F], got shape {tuple(x.shape)}")
        if x.shape[0] != self.member_count:
            raise ValueError(
                f"packed input member dimension must be {self.member_count}, got {x.shape[0]}"
            )
        _check_input(x, rank=3, feature_count=self.input_dim, reference=reference)
        return x

    def _prepare_masks(
        self, dropout_masks: Sequence[Tensor] | None, batch_size: int, device: torch.device
    ) -> list[Tensor] | None:
        if dropout_masks is None:
            return None
        masks = list(dropout_masks)
        if len(masks) != self.max_depth:
            raise ValueError(f"dropout_masks must have {self.max_depth} layer entries")
        checked: list[Tensor] = []
        expected = (self.member_count, batch_size, self.max_width)
        for index, mask in enumerate(masks):
            if not isinstance(mask, Tensor):
                raise TypeError(f"dropout_masks[{index}] must be a tensor")
            if tuple(mask.shape) != expected:
                raise ValueError(
                    f"dropout_masks[{index}] must have shape {expected}, got {tuple(mask.shape)}"
                )
            if mask.device != device:
                raise ValueError("dropout masks must be on the same device as the input")
            if mask.dtype != torch.bool:
                raise TypeError("dropout masks must be boolean keep masks")
            checked.append(mask)
        return checked

    def forward_features(
        self, x: Tensor, *, dropout_masks: Sequence[Tensor] | None = None
    ) -> Tensor:
        x_pack = self._prepare_input(x)
        masks = self._prepare_masks(dropout_masks, x_pack.shape[1], x_pack.device)
        hidden = x_pack.new_zeros((self.member_count, x_pack.shape[1], self.max_width))
        width_mask = self.width_mask.to(dtype=x_pack.dtype)

        for layer_index in range(self.max_depth):
            active = self.depths > layer_index
            if not bool(active.any()):
                break
            indices = torch.nonzero(active, as_tuple=False).flatten()
            layer_input = x_pack[indices] if layer_index == 0 else hidden[indices]
            weight = self.hidden_weight[layer_index][indices]
            bias = self.hidden_bias[layer_index][indices]
            z = torch.bmm(layer_input, weight.transpose(1, 2))
            z = (z + bias[:, None, :]) * width_mask[indices, None, :]
            z = F.relu(z)
            if self.training and masks is not None:
                keep = masks[layer_index][indices]
                keep = keep | (self.dropouts[indices, None, None] == 0)
                scale = 1.0 / (
                    1.0 - self.dropouts[indices, None, None].to(dtype=z.dtype)
                )
                z = z * keep.to(dtype=z.dtype) * scale
            elif self.training:
                probabilities = self.dropouts[indices, None, None].to(dtype=z.dtype)
                if bool((probabilities != 0).any()):
                    keep = torch.rand(z.shape, device=z.device, dtype=z.dtype) >= probabilities
                    z = z * keep.to(dtype=z.dtype) / (1.0 - probabilities)
            z = z * width_mask[indices, None, :]

            full = z.new_zeros((self.member_count, z.shape[1], self.max_width))
            full = full.index_copy(0, indices, z)
            hidden = torch.where(active[:, None, None], full, hidden)

        hidden = hidden * width_mask[:, None, :]
        return hidden

    def forward(
        self, x: Tensor, *, dropout_masks: Sequence[Tensor] | None = None
    ) -> Tensor:
        hidden = self.forward_features(x, dropout_masks=dropout_masks)
        logits = torch.bmm(hidden, self.output_weight.transpose(1, 2))
        logits = logits + self.output_bias[:, None, :]
        return logits[..., 0] if self.output_dim == 1 else logits

    def parameter_masks(self) -> dict[nn.Parameter, Tensor]:
        """Return boolean ownership masks with each parameter's complete shape.

        Use these when building the optimizer so nonexistent depth rows and
        padded width slots are excluded from moments, decay, and updates.
        Call after moving the model to its training device.
        """

        masks: dict[nn.Parameter, Tensor] = {}
        for layer_index, (weight, bias) in enumerate(zip(self.hidden_weight, self.hidden_bias)):
            outputs = self.width_mask & (self.depths > layer_index)[:, None]
            inputs = (
                torch.ones(
                    self.member_count, self.input_dim, device=weight.device, dtype=torch.bool
                )
                if layer_index == 0
                else self.width_mask
            )
            masks[weight] = outputs[:, :, None] & inputs[:, None, :]
            masks[bias] = outputs
        masks[self.output_weight] = self.width_mask[:, None, :].expand_as(self.output_weight)
        masks[self.output_bias] = torch.ones_like(self.output_bias, dtype=torch.bool)
        return masks

    def optimizer_groups(self, weight_decay: Any = 0.0) -> list[dict[str, Any]]:
        """AdamW groups with structural masks and zero decay on every bias."""

        masks = self.parameter_masks()
        return [
            {"params": [parameter], "mask": masks[parameter],
             "weight_decay": 0.0 if "bias" in name else weight_decay}
            for name, parameter in self.named_parameters()
        ]

    @property
    def logical_parameter_count(self) -> int:
        """Number of real member parameters, excluding stored padding."""

        return sum(
            self.input_dim * width + (depth - 1) * width * width
            + depth * width + self.output_dim * (width + 1)
            for depth, width, _ in self.member_specs
        )

    @property
    def stored_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def copy_member_from(self, member: int, source: OrdinaryMLP) -> None:
        """Copy active weights from an ordinary model; leave other members alone."""

        if isinstance(member, bool) or not isinstance(member, int):
            raise TypeError("member must be an integer")
        if not 0 <= member < self.member_count:
            raise IndexError("member index is out of range")
        depth, width, _ = self.member_specs[member]
        if (source.input_dim, source.depth, source.width, source.output_dim) != (
            self.input_dim, depth, width, self.output_dim
        ):
            raise ValueError("source architecture must match the packed member")
        for layer_index, layer in enumerate(source.layers):
            fan_in = self.input_dim if layer_index == 0 else width
            self.hidden_weight[layer_index][member, :width, :fan_in].copy_(layer.weight)
            self.hidden_bias[layer_index][member, :width].copy_(layer.bias)
        self.output_weight[member, :, :width].copy_(source.head.weight)
        self.output_bias[member].copy_(source.head.bias)

    def extract_member(self, member: int) -> OrdinaryMLP:
        """Create an independent, compact weights snapshot of one member.

        The returned module has independent storage and the same device,
        dtype, and training flag.  Constructing it does not consume the caller's
        random stream, which matters when snapshotting during training.
        """

        if isinstance(member, bool) or not isinstance(member, int):
            raise TypeError("member must be an integer")
        if not 0 <= member < self.member_count:
            raise IndexError("member index is out of range")
        depth, width, dropout = self.member_specs[member]
        # Build on CPU under fork_rng, then move; CUDA RNG state is untouched.
        with torch.random.fork_rng(devices=[]):
            result = OrdinaryMLP(
                self.input_dim, depth=depth, width=width, dropout=dropout,
                output_dim=self.output_dim, dtype=self.output_weight.dtype,
            ).to(self.output_weight.device)
        with torch.no_grad():
            for layer_index, layer in enumerate(result.layers):
                fan_in = self.input_dim if layer_index == 0 else width
                layer.weight.copy_(self.hidden_weight[layer_index][member, :width, :fan_in])
                layer.bias.copy_(self.hidden_bias[layer_index][member, :width])
            result.head.weight.copy_(self.output_weight[member, :, :width])
            result.head.bias.copy_(self.output_bias[member])
        return result.train(self.training)

    @classmethod
    def from_members(cls, members: Sequence[OrdinaryMLP]) -> PackedMLP:
        """Pack ordinary models in the given order, copying their active weights."""

        members = list(members)
        if not members:
            raise ValueError("at least one ordinary member is required")
        first = members[0]
        for member in members:
            if (member.input_dim, member.output_dim) != (first.input_dim, first.output_dim):
                raise ValueError("members must share input_dim and output_dim")
            if (member.head.weight.device, member.head.weight.dtype) != (
                first.head.weight.device, first.head.weight.dtype
            ):
                raise ValueError("members must share device and dtype")
        with torch.random.fork_rng(devices=[]):
            result = cls(
                first.input_dim, depths=[member.depth for member in members],
                widths=[member.width for member in members],
                dropouts=[member.dropout for member in members], output_dim=first.output_dim,
                dtype=first.head.weight.dtype,
            ).to(first.head.weight.device)
        for index, member in enumerate(members):
            result.copy_member_from(index, member)
        return result.train(first.training)


def packed_bce_loss(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(sum(member sample means), member sample means)`` for logits.

    Binary packed logits have shape ``[M, B]``.  Targets may be shared
    ``[B]`` or member-specific ``[M, B]``.  The helper also accepts packed
    ``[M, B, 1]`` logits or targets, but never averages over the member axis.
    """

    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits[..., 0]
    if logits.ndim != 2:
        raise ValueError(f"packed binary logits must have shape [M, B], got {tuple(logits.shape)}")
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("packed loss requires nonempty member and batch dimensions")
    if not isinstance(targets, Tensor):
        raise TypeError("targets must be a tensor")
    if targets.ndim == 3 and targets.shape[-1] == 1:
        targets = targets[..., 0]
    if targets.ndim == 1:
        if targets.shape[0] != logits.shape[1]:
            raise ValueError("shared targets must have one value per batch row")
        target_pack = targets.unsqueeze(0).expand(logits.shape[0], -1)
    elif targets.ndim == 2:
        if tuple(targets.shape) != tuple(logits.shape):
            raise ValueError("member-specific targets must match packed logits")
        target_pack = targets
    else:
        raise ValueError("targets must have shape [B] or [M, B]")
    if target_pack.device != logits.device:
        raise ValueError("targets and logits must be on the same device")
    if not bool(torch.isfinite(target_pack).all()) or not bool(
        ((target_pack == 0) | (target_pack == 1)).all()
    ):
        raise ValueError("binary targets must contain only finite zeroes and ones")
    target_pack = target_pack.to(dtype=logits.dtype)
    element_loss = F.binary_cross_entropy_with_logits(logits, target_pack, reduction="none")
    per_member = element_loss.mean(dim=1)
    return per_member.sum(), per_member


# A descriptive alias is convenient at call sites and keeps the reduction
# contract discoverable without introducing another implementation.
packed_bce_with_logits_loss = packed_bce_loss


__all__ = [
    "OrdinaryMLP",
    "PackedMLP",
    "packed_bce_loss",
    "packed_bce_with_logits_loss",
]
