"""Reusable baseline models for tabular experiments.

The public :class:`SimpleMLP` keeps the small baseline configuration separate
from the experiment runner.  It returns logits, so callers can choose the
appropriate stable loss and metric boundary for binary or multi-output tasks.
"""

from __future__ import annotations

import torch

from .models import OrdinaryMLP


class SimpleMLP(OrdinaryMLP):
    """A reusable feed-forward MLP baseline with practical defaults.

    Parameters
    ----------
    input_dim:
        Number of dense input features.
    hidden_dim:
        Width of every hidden layer. Defaults to 64.
    depth:
        Number of ``Linear -> ReLU -> Dropout`` hidden blocks. Defaults to 2.
    dropout:
        Inverted dropout probability applied after each hidden block. Defaults
        to 0.1. Set it to 0.0 for deterministic training/evaluation behavior.
    output_dim:
        Number of output logits. With the default value of 1, ``forward``
        returns shape ``[batch]``; otherwise it returns ``[batch, output_dim]``.

    The model does not apply sigmoid or softmax. For binary classification use
    ``torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)``.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            input_dim,
            depth=depth,
            width=hidden_dim,
            dropout=dropout,
            output_dim=output_dim,
            device=device,
            dtype=dtype,
        )

    @property
    def hidden_dim(self) -> int:
        """The shared hidden width, using baseline terminology."""

        return self.width


__all__ = ["SimpleMLP"]
