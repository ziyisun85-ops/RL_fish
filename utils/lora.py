from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(0, int(rank))
        self.alpha = float(alpha)
        self.scaling = float(self.alpha / self.rank) if self.rank > 0 else 0.0
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

        if self.rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.rank))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        freeze_base: bool = True,
    ) -> "LoRALinear":
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            freeze_base=freeze_base,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None and layer.bias is not None:
                layer.bias.copy_(linear.bias)
        layer.to(device=linear.weight.device, dtype=linear.weight.dtype)
        return layer

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(max(fan_in, 1))
            nn.init.uniform_(self.bias, -bound, bound)
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = F.linear(inputs, self.weight, self.bias)
        if self.rank <= 0:
            return output
        lora_update = F.linear(self.dropout(inputs), self.lora_A)
        lora_update = F.linear(lora_update, self.lora_B)
        return output + self.scaling * lora_update


def _resolve_parent_module(module: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    if not parts:
        raise ValueError("module_name cannot be empty.")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_linear_with_lora(
    module: nn.Module,
    target_module_names: Iterable[str],
    *,
    rank: int,
    alpha: float,
    dropout: float,
    freeze_base: bool = True,
) -> list[str]:
    replaced: list[str] = []
    target_names = {str(name) for name in target_module_names}
    named_modules = dict(module.named_modules())

    for module_name in target_names:
        current = named_modules.get(module_name)
        if current is None:
            continue
        if not isinstance(current, nn.Linear):
            raise TypeError(f"LoRA target is not nn.Linear: {module_name} ({type(current)!r})")
        parent, child_name = _resolve_parent_module(module, module_name)
        setattr(
            parent,
            child_name,
            LoRALinear.from_linear(
                current,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                freeze_base=freeze_base,
            ),
        )
        replaced.append(module_name)
    return replaced
