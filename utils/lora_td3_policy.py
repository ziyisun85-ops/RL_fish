from __future__ import annotations

from typing import Any

from stable_baselines3.td3.policies import MultiInputPolicy

from utils.lora import replace_linear_with_lora


DEFAULT_TD3_LORA_TARGET_MODULES = (
    "actor.features_extractor.extractors.image.linear.0",
    "actor.mu.0",
    "actor.mu.2",
    "actor.mu.4",
    "actor_target.features_extractor.extractors.image.linear.0",
    "actor_target.mu.0",
    "actor_target.mu.2",
    "actor_target.mu.4",
)


def resolve_td3_lora_target_modules(target_modules: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    resolved: list[str] = []
    seen: set[str] = set()

    def _add(module_name: str) -> None:
        module_name = str(module_name)
        if module_name in seen:
            return
        seen.add(module_name)
        resolved.append(module_name)

    for module_name in target_modules:
        module_name = str(module_name)
        _add(module_name)
        if module_name.startswith("actor."):
            _add(f"actor_target.{module_name.split('actor.', 1)[1]}")
        elif module_name.startswith("actor_target."):
            _add(f"actor.{module_name.split('actor_target.', 1)[1]}")

    return tuple(resolved)


class LoraTD3Policy(MultiInputPolicy):
    def __init__(
        self,
        *args,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        lora_target_modules: tuple[str, ...] = DEFAULT_TD3_LORA_TARGET_MODULES,
        lora_freeze_actor_base: bool = True,
        lora_train_bias: bool = False,
        **kwargs: Any,
    ) -> None:
        lr_schedule = kwargs.get("lr_schedule")
        if lr_schedule is None and len(args) >= 3:
            lr_schedule = args[2]
        if lr_schedule is None:
            raise ValueError("LoraTD3Policy requires lr_schedule.")
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
            raise RuntimeError("No LoRA modules were applied to the TD3/DDPG policy.")
        self.actor_target.load_state_dict(self.actor.state_dict())
        self._set_lora_trainable_flags()
        actor_trainable_parameters = [parameter for parameter in self.actor.parameters() if parameter.requires_grad]
        self.actor.optimizer = self.optimizer_class(
            actor_trainable_parameters,
            lr=self._lora_initial_lr,
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                lora_target_modules=self.lora_target_modules,
                lora_freeze_actor_base=self.lora_freeze_actor_base,
                lora_train_bias=self.lora_train_bias,
            )
        )
        return data

    def _set_lora_trainable_flags(self) -> None:
        for name, parameter in self.named_parameters():
            if name.endswith(("lora_A", "lora_B")):
                parameter.requires_grad = not name.startswith("actor_target.")
                continue
            if name.startswith("critic_target."):
                parameter.requires_grad = False
                continue
            if name.startswith("critic."):
                parameter.requires_grad = True
                continue
            if name.startswith("actor_target."):
                parameter.requires_grad = False
                continue
            if not name.startswith("actor."):
                continue
            if self.lora_train_bias and name.endswith(".bias"):
                parameter.requires_grad = True
                continue
            if self.lora_freeze_actor_base:
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True

    def lora_parameter_count(self) -> int:
        return int(
            sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if name.endswith(("lora_A", "lora_B")) and not name.startswith("actor_target.")
            )
        )
