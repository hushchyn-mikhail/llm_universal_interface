"""CPU tests for stopping before corrupt training state is persisted."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import Trainer, TrainingArguments

from phi import experiment
from phi.numerics import FiniteParameters, FiniteTrainer, check_optimizer, check_parameters


class TinyModel(torch.nn.Module):
    def __init__(self, bad_loss=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.bad_loss = bad_loss

    def forward(self, input_ids, labels=None):
        loss = (self.weight * input_ids.float()).square().mean()
        return {"loss": loss * float("nan") if self.bad_loss else loss}


class NumericTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def trainer(self, model):
        return FiniteTrainer(
            model=model,
            args=TrainingArguments(output_dir=str(self.root), use_cpu=True, max_steps=2,
                                   per_device_train_batch_size=1, save_strategy="no",
                                   logging_strategy="no", disable_tqdm=True, report_to="none",
                                   dataloader_pin_memory=False),
            train_dataset=[{"input_ids": torch.ones(3)} for _ in range(2)],
            callbacks=[FiniteParameters()])

    def test_finite_training_completes_and_updates_weights(self):
        model = TinyModel()
        trainer = self.trainer(model)
        trainer.train()
        self.assertEqual(trainer.state.global_step, 2)
        self.assertLess(model.weight.item(), 1)
        check_parameters(model, "test")
        check_optimizer(model, trainer.optimizer, "test")

    def test_nonfinite_loss_is_rejected_before_backward(self):
        model = TinyModel(bad_loss=True)
        trainer = self.trainer(model)
        trainer.accelerator.backward = Mock(side_effect=AssertionError("backward must not run"))
        with self.assertRaisesRegex(FloatingPointError, "loss before backward"):
            trainer.train()
        trainer.accelerator.backward.assert_not_called()
        self.assertEqual(model.weight.item(), 1)

    def test_nonfinite_gradient_is_rejected_before_optimizer_update(self):
        model = TinyModel()
        model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        trainer = self.trainer(model)
        with self.assertRaisesRegex(FloatingPointError, "gradients.*before optimizer.*weight"):
            trainer.train()
        self.assertEqual(model.weight.item(), 1)
        self.assertEqual(trainer.state.global_step, 0)
        self.assertFalse(trainer.optimizer.state)

    def test_train_begin_rejects_bad_loaded_weights(self):
        model = TinyModel()
        with torch.no_grad():
            model.weight.fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "weights at train_begin"):
            self.trainer(model).train()

    def test_optimizer_corruption_is_rejected_even_with_finite_weights(self):
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters())
        model.weight.square().backward()
        optimizer.step()
        optimizer.state[model.weight]["exp_avg_sq"].fill_(float("inf"))
        callback = FiniteParameters()
        for event in (callback.on_train_begin, callback.on_optimizer_step):
            with self.assertRaisesRegex(FloatingPointError, "optimizer state.*weight.exp_avg_sq"):
                event(None, SimpleNamespace(global_step=1), None, model=model, optimizer=optimizer)

    def test_bad_weights_cannot_mark_checkpoint_or_adapter_complete(self):
        model = TinyModel()
        with torch.no_grad():
            model.weight.fill_(float("nan"))
        artifacts = SimpleNamespace(manifest_sha256="digest", directory=self.root, status=Mock())
        checkpoint = self.root / "checkpoint-1"
        checkpoint.mkdir()
        with self.assertRaisesRegex(FloatingPointError, "weights at checkpoint"):
            experiment.DurableCheckpoint(artifacts).on_save(
                SimpleNamespace(output_dir=self.root), SimpleNamespace(global_step=1), None, model=model)
        self.assertFalse((checkpoint / "complete.json").exists())
        tokenizer = Mock()
        model.save_pretrained = Mock()
        adapter = self.root / "adapter"
        with self.assertRaisesRegex(FloatingPointError, "weights at adapter save"):
            experiment.save_adapter(model, tokenizer, adapter, "digest")
        model.save_pretrained.assert_not_called()
        self.assertFalse(adapter.exists())

    def test_bad_optimizer_cannot_mark_checkpoint_complete(self):
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.state[model.weight]["exp_avg"] = torch.tensor(float("nan"))
        artifacts = SimpleNamespace(manifest_sha256="digest", directory=self.root, status=Mock())
        with self.assertRaisesRegex(FloatingPointError, "optimizer state at checkpoint"):
            experiment.DurableCheckpoint(artifacts).on_save(
                SimpleNamespace(output_dir=self.root), SimpleNamespace(global_step=1), None,
                model=model, optimizer=optimizer)
        self.assertFalse((self.root / "checkpoint-1/complete.json").exists())

    def test_bad_state_stops_before_checkpoint_write_or_rotation(self):
        model = TinyModel()
        trainer = self.trainer(model)
        trainer.create_optimizer()
        trainer.optimizer.state[model.weight]["exp_avg"] = torch.tensor(float("inf"))
        with patch.object(Trainer, "_save_checkpoint") as save:
            with self.assertRaisesRegex(FloatingPointError, "before checkpoint write"):
                trainer._save_checkpoint(model, None)
            save.assert_not_called()

    def test_frozen_lora_weights_are_checked_without_scanning_frozen_base(self):
        model = torch.nn.Module()
        model.register_parameter("base", torch.nn.Parameter(torch.tensor(float("nan")), requires_grad=False))
        model.register_parameter("lora_A", torch.nn.Parameter(torch.tensor(1.), requires_grad=False))
        check_parameters(model, "test")
        model.lora_A.fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "lora_A"):
            check_parameters(model, "test")


if __name__ == "__main__":
    unittest.main()
