from __future__ import annotations

from typing import Any

import torch as th
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from utils.lora import replace_linear_with_lora
from utils.policy_utils import is_actor_parameter_name


DEFAULT_LORA_TARGET_MODULES = (
    "features_extractor.extractors.image.linear.0",
    "mlp_extractor.policy_net.0",
    "mlp_extractor.policy_net.2",
    "action_net",
)


class LoraMultiInputPolicy(MultiInputActorCriticPolicy):
    def __init__(
        self,
        *args,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES,
        lora_freeze_actor_base: bool = True,
        lora_train_bias: bool = False,
        **kwargs: Any,
    ) -> None:
        lr_schedule = kwargs.get("lr_schedule")
        if lr_schedule is None and len(args) >= 3:
            lr_schedule = args[2]
        if lr_schedule is None:
            raise ValueError("LoraMultiInputPolicy requires lr_schedule.")
        self.lora_rank = max(1, int(lora_rank))
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = tuple(str(name) for name in lora_target_modules)
        self.lora_freeze_actor_base = bool(lora_freeze_actor_base)
        self.lora_train_bias = bool(lora_train_bias)
        self._lora_initial_lr = float(lr_schedule(1))
        super().__init__(*args, **kwargs)
        self.applied_lora_modules = tuple(
            replace_linear_with_lora(
                self,
                self.lora_target_modules,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                freeze_base=self.lora_freeze_actor_base,
            )
        )
        if not self.applied_lora_modules:
            raise RuntimeError("No LoRA modules were applied to the PPO policy.")
        self._set_lora_trainable_flags()
        # Rebuild the optimizer so it tracks the replaced LoRA modules.
        trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        self.optimizer = self.optimizer_class(trainable_parameters, lr=self._lora_initial_lr, **self.optimizer_kwargs)

    def _set_lora_trainable_flags(self) -> None:
        for name, parameter in self.named_parameters():
            if name.endswith("lora_A") or name.endswith("lora_B"):
                parameter.requires_grad = True
                continue
            if name == "log_std":
                parameter.requires_grad = True
                continue
            if not is_actor_parameter_name(name):
                parameter.requires_grad = True
                continue
            if self.lora_train_bias and name.endswith(".bias"):
                parameter.requires_grad = True
                continue
            if self.lora_freeze_actor_base:
                parameter.requires_grad = False

        for _, buffer in self.named_buffers():
            _ = buffer

    def lora_parameter_count(self) -> int:
        return int(sum(parameter.numel() for name, parameter in self.named_parameters() if name.endswith(("lora_A", "lora_B"))))
