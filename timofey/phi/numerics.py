"""Stop training at the first non-finite loss, gradient, or adapter weight."""

from collections import defaultdict

import torch
from transformers import Trainer, TrainerCallback


@torch.no_grad()
def check_tensors(values, stage, kind):
    groups = defaultdict(list)
    for name, value in values:
        value = value.detach()
        if value.is_sparse:
            value = value.coalesce().values()
        groups[value.device].append((name, value))
    for values in groups.values():
        finite = torch.stack([torch.isfinite(value).all() for _, value in values])
        if not finite.all().item():
            names = [name for (name, _), valid in zip(values, finite.cpu().tolist()) if not valid]
            raise FloatingPointError(f"Non-finite {kind} at {stage}: {', '.join(names)}")


def check_parameters(model, stage, gradients=False):
    values = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and "lora_" not in name:
            continue
        value = parameter.grad if gradients else parameter
        if value is None:
            continue
        values.append((name, value))
    check_tensors(values, stage, "gradients" if gradients else "weights")


def check_optimizer(model, optimizer, stage):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    values = []
    for index, (parameter, state) in enumerate(optimizer.state.items()):
        name = names.get(id(parameter), f"parameter_{index}")
        values.extend((f"{name}.{key}", value) for key, value in state.items()
                      if isinstance(value, torch.Tensor))
    check_tensors(values, stage, "optimizer state")


class FiniteTrainer(Trainer):
    def _save_checkpoint(self, model, trial):
        check_parameters(model, "before checkpoint write")
        check_optimizer(model, self.optimizer, "before checkpoint write")
        return super()._save_checkpoint(model, trial)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        loss = result[0] if return_outputs else result
        if not torch.isfinite(loss.detach()).all().item():
            raise FloatingPointError("Non-finite training loss before backward")
        return result


class FiniteParameters(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        check_parameters(model, "train_begin")
        if optimizer is not None:
            check_optimizer(model, optimizer, "train_begin")
        return control

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        check_parameters(model, f"before optimizer step {state.global_step + 1}", gradients=True)
        return control

    def on_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        check_parameters(model, f"after optimizer step {state.global_step + 1}")
        if optimizer is not None:
            check_optimizer(model, optimizer, f"after optimizer step {state.global_step + 1}")
        return control
