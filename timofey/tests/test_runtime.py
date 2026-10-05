"""CPU checks for persistence, interrupted evaluation, and resume identity."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phi import experiment as runtime
from phi.run_artifacts import RunArtifacts, atomic_json, hash_files, sha256_file


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(item["content"] for item in messages) + "\nassistant:"

    def save_pretrained(self, directory):
        Path(directory, "tokenizer_config.json").write_text('{}')


class FakeModel:
    def __init__(self):
        self.training = True
        self.weight = runtime.torch.nn.Parameter(runtime.torch.ones(1))

    def named_parameters(self):
        return [("adapter", self.weight)]

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def save_pretrained(self, directory):
        Path(directory, "adapter_config.json").write_text('{}')
        Path(directory, "adapter_model.safetensors").write_bytes(b'fake-adapter')


def manifest():
    return {"schema_version": 1, "run_id": "run", "plan_id": None,
            "scoring": "full_label_log_likelihood_v2", "config": {"dataset": "heart"}}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def store(self, path="run", resume=False, data=None):
        result = RunArtifacts(self.root / path, data or manifest(), resume=resume)
        self.addCleanup(result.close)
        return result

    def test_fresh_run_refuses_overwrite_and_resume_checks_manifest(self):
        store = self.store()
        store.close()
        with self.assertRaises(FileExistsError):
            self.store()
        resumed = self.store(resume=True)
        self.assertEqual(resumed.manifest_sha256, sha256_file(resumed.directory / "manifest.json"))
        resumed.close()
        changed = manifest()
        changed["config"]["seed"] = 17
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            self.store(resume=True, data=changed)

    def test_run_lock_refuses_second_writer(self):
        self.store()
        with self.assertRaises(BlockingIOError):
            self.store(resume=True)

    def test_batch_resume_checks_order_targets_and_probabilities(self):
        store = self.store()
        store.save_batch(0, [17, 5], [1, 0], np.array([[.2, .8], [.6, .4]]), [12, 15], ['a'*64, 'b'*64])
        chunks = store.read_batches([17, 5, 8], [1, 0, 1], 2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["source_row_ids"].tolist(), [17, 5])
        with self.assertRaisesRegex(ValueError, "row IDs changed"):
            store.read_batches([5, 17, 8], [1, 0, 1], 2)
        with self.assertRaisesRegex(ValueError, "targets changed"):
            store.read_batches([17, 5, 8], [0, 0, 1], 2)
        store.save_batch(3, [8], [1], np.array([[.1, .9]]), [13], ['c'*64])
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            store.read_batches([17, 5, 8, 9], [1, 0, 1, 0], 2)

    def test_partial_temporary_batch_is_not_a_prediction(self):
        store = self.store()
        (store.batches / '.batch.incomplete.tmp').write_bytes(b'partial')
        self.assertEqual(store.read_batches([2], [1], 2), [])

    def test_persisted_invalid_probability_is_rejected(self):
        store = self.store()
        store.save_batch(0, [2], [1], np.array([[.2, .2]]), [10], ['a'*64])
        with self.assertRaisesRegex(ValueError, "Invalid class probabilities"):
            store.read_batches([2], [1], 2)

    def test_interrupted_evaluation_resumes_only_missing_rows(self):
        frame = pd.DataFrame({'Age': [40, 50, 60, 70], 'label': [0, 1, 0, 1]}, index=[8, 2, 19, 7])
        info = {"eval_df": frame, "feature_names": ['Age'], "target_name": 'label',
                "prompt_config": {'task': 'Predict', 'entity': 'Patient', 'question': 'Class?', 'labels': ['0', '1']}}
        cfg = {'mode': 'zero_shot', 'missing_rate': 0., 'missingness_scheme': 'fixed_count_floor',
               'eval_max_seq_length': 500, 'max_seq_length': 500, 'eval_batch_size': 2}
        store = self.store()
        calls = []

        def fail_second(*args):
            calls.append(args[2])
            if len(calls) == 2:
                raise RuntimeError('simulated interruption')
            return np.array([[.8, .2], [.2, .8]])

        with patch.object(runtime, 'class_probabilities', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                runtime.evaluate(cfg, info, None, Tokenizer(), 'cpu', store, None, lambda _: None, bootstrap=False)
        self.assertEqual(len(store.read_batches(frame.index, frame.label, 2)), 1)
        with patch.object(runtime, 'class_probabilities', return_value=np.array([[.7, .3], [.1, .9]])) as scorer:
            result = runtime.evaluate(cfg, info, None, Tokenizer(), 'cpu', store, None, lambda _: None, bootstrap=False)
        self.assertEqual(scorer.call_count, 1)
        self.assertEqual(result['resumed_prediction_rows'], 2)
        self.assertEqual(result['n_test'], 4)
        self.assertEqual(result['metrics']['ROC-AUC']['point'], 1.)
        # All rows are durable; even model=None is enough to rebuild final metrics.
        with patch.object(runtime, 'class_probabilities', side_effect=AssertionError('must not run')):
            replay = runtime.evaluate(cfg, info, None, Tokenizer(), 'cpu', store, None, lambda _: None, bootstrap=False)
        self.assertEqual(replay['resumed_prediction_rows'], 4)
        info['prompt_config']['question'] = 'Changed instruction'
        with self.assertRaisesRegex(ValueError, 'Saved prompt differs'):
            runtime.evaluate(cfg, info, None, Tokenizer(), 'cpu', store, None, lambda _: None, bootstrap=False)

    def test_split_ids_refer_to_original_rows(self):
        frame = pd.DataFrame({'x': range(100), 'y': [0, 1] * 50})
        train, validation, test = runtime.split_df(frame, 'y')
        self.assertEqual((len(train), len(validation), len(test)), (60, 20, 20))
        self.assertEqual(set(train.index) | set(validation.index) | set(test.index), set(range(100)))
        self.assertFalse(set(train.index) & set(test.index))
        self.assertTrue(np.array_equal(test.x.to_numpy(), test.index.to_numpy()))

    def test_bootstrap_is_numeric_and_counts_valid_auc_samples(self):
        result = runtime.bootstrap_metrics(np.array([0, 1]), np.array([0, 1]),
                                           np.array([[.8, .2], [.2, .8]]), 2, n_iter=20)
        auc = result['ROC-AUC']
        self.assertEqual(auc['point'], 1.)
        self.assertGreater(auc['n_bootstrap_valid'], 0)
        self.assertLess(auc['n_bootstrap_valid'], 20)
        self.assertEqual(result['Accuracy']['n_bootstrap_valid'], 20)
        atomic_json(self.root / 'metrics.json', result)
        self.assertIsInstance(json.loads((self.root / 'metrics.json').read_text())['ROC-AUC']['bootstrap_std'], float)

    def test_precision_does_not_treat_bf16_emulation_as_native(self):
        with patch.object(runtime.torch.cuda, 'is_available', return_value=True), \
             patch.object(runtime.torch.cuda, 'is_bf16_supported', return_value=False) as supported:
            self.assertEqual(runtime.choose_precision_config()['name'], 'float16')
            supported.assert_called_with(including_emulation=False)
            with self.assertRaisesRegex(ValueError, 'lacks native BF16'):
                runtime.choose_precision_config('bfloat16')

    def test_missingness_scheme_is_explicit_and_deterministic(self):
        row = pd.Series(dict(zip(['a', 'b', 'c', 'd'], [1, 2, 3, 4])))
        features = list(row.index)
        complete = runtime.row_to_text(row, features)
        self.assertEqual(runtime.row_to_text(row, features, .2, 3), complete)
        first = runtime.row_to_text(row, features, .2, 3, 'bernoulli')
        self.assertEqual(first, runtime.row_to_text(row, features, .2, 3, 'bernoulli'))
        self.assertNotEqual(first, complete)

    def test_checkpoint_requires_completion_marker_and_hashes(self):
        root = self.root / 'checkpoints'
        good = root / 'checkpoint-10'
        good.mkdir(parents=True)
        (good / 'trainer_state.json').write_text('{}')
        atomic_json(good / 'complete.json', {'manifest_sha256': 'fingerprint',
                                             'files': hash_files(good, [good / 'trainer_state.json'])})
        (root / 'checkpoint-20').mkdir()
        self.assertEqual(runtime.latest_checkpoint(root, 'fingerprint', lambda _: None), str(good))
        (good / 'trainer_state.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed or incomplete'):
            runtime.latest_checkpoint(root, 'fingerprint', lambda _: None)

    def test_validation_selection_resumes_pending_epoch_before_training(self):
        store = self.store()
        model = FakeModel()
        state = SimpleNamespace(epoch=1., global_step=10)
        control = SimpleNamespace()
        callback = runtime.ValidationSelection(store, Tokenizer(), lambda model, state: {'selection_score': .7}, lambda _: None)
        callback.on_train_begin(None, state, control, model=model)
        pointer = json.loads((store.directory / 'best_adapter.json').read_text())
        self.assertEqual(pointer['selection_score'], .7)
        self.assertTrue((store.directory / 'validation/step-10.json').is_file())
        with patch.object(callback, 'scorer', side_effect=AssertionError('already committed')):
            callback.on_train_begin(None, state, control, model=model)
        self.assertTrue(model.training)
        poorer = runtime.ValidationSelection(store, Tokenizer(), lambda model, state: {'selection_score': .6}, lambda _: None)
        poorer.on_epoch_end(None, SimpleNamespace(epoch=2., global_step=20), control, model=model)
        self.assertEqual(poorer.best['selection_score'], .7)

    def test_model_metadata_fingerprint_detects_tokenizer_changes(self):
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        (model / 'tokenizer.json').write_text('{}')
        (model / 'weights.safetensors').write_bytes(b'dummy')
        before = runtime.model_fingerprint(model)
        (model / 'tokenizer.json').write_text('{"changed":true}')
        self.assertNotEqual(before, runtime.model_fingerprint(model))
        self.assertEqual(before['weight_verification'], 'size_mtime')

    def test_improved_best_prunes_only_previous_adapter(self):
        store = self.store()
        callback = runtime.ValidationSelection(store, Tokenizer(),
                                               lambda model, state: {'selection_score': state.epoch / 10}, lambda _: None)
        callback.on_epoch_end(None, SimpleNamespace(epoch=1., global_step=10), None, model=FakeModel())
        previous = store.directory / callback.best['path']
        callback.on_epoch_end(None, SimpleNamespace(epoch=2., global_step=20), None, model=FakeModel())
        self.assertFalse(previous.exists())
        self.assertTrue((store.directory / callback.best['path'] / 'complete.json').is_file())
        self.assertEqual(len(list((store.directory / 'best_adapters').glob('step-*'))), 1)
        self.assertTrue((store.directory / 'validation/step-10.json').is_file())
        self.assertTrue((store.directory / 'validation/step-20.json').is_file())

    def test_interruption_before_best_pointer_keeps_previous_adapter(self):
        store = self.store()
        callback = runtime.ValidationSelection(store, Tokenizer(),
                                               lambda model, state: {'selection_score': state.epoch / 10}, lambda _: None)
        callback.on_epoch_end(None, SimpleNamespace(epoch=1., global_step=10), None, model=FakeModel())
        previous = dict(callback.best)
        write_json = runtime.atomic_json

        def interrupted(path, value):
            if Path(path) == callback.pointer:
                raise OSError('interrupted before pointer commit')
            return write_json(path, value)

        with patch.object(runtime, 'atomic_json', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                callback.on_epoch_end(None, SimpleNamespace(epoch=2., global_step=20), None, model=FakeModel())
        self.assertEqual(callback.best, previous)
        self.assertEqual(json.loads(callback.pointer.read_text()), previous)
        self.assertTrue((store.directory / previous['path'] / 'complete.json').is_file())

    def test_best_cleanup_failure_does_not_invalidate_new_best(self):
        store = self.store()
        logs = []
        callback = runtime.ValidationSelection(store, Tokenizer(),
                                               lambda model, state: {'selection_score': state.epoch / 10}, logs.append)
        callback.on_epoch_end(None, SimpleNamespace(epoch=1., global_step=10), None, model=FakeModel())
        with patch.object(runtime.shutil, 'rmtree', side_effect=OSError('cleanup denied')):
            callback.on_epoch_end(None, SimpleNamespace(epoch=2., global_step=20), None, model=FakeModel())
        self.assertEqual(json.loads(callback.pointer.read_text())['selection_score'], .2)
        self.assertTrue((store.directory / 'validation/step-20.json').is_file())
        self.assertTrue(any('cleanup denied' in line for line in logs))

    def test_best_cleanup_refuses_outside_path_and_symlink(self):
        store = self.store()
        callback = runtime.ValidationSelection(store, Tokenizer(), lambda model, state: {'selection_score': .5}, lambda _: None)
        callback.on_epoch_end(None, SimpleNamespace(epoch=1., global_step=10), None, model=FakeModel())
        outside = self.root / 'outside'
        outside.mkdir()
        (outside / 'keep').write_text('untouched')
        previous = dict(callback.best, path='../outside')
        with patch.object(runtime.shutil, 'rmtree') as remove:
            callback._prune_previous(previous)
            link = store.directory / 'best_adapters/step-5-1234abcd'
            link.symlink_to(outside, target_is_directory=True)
            callback._prune_previous(dict(previous, path='best_adapters/step-5-1234abcd'))
            remove.assert_not_called()
        self.assertTrue((outside / 'keep').is_file())

    def test_training_rejects_an_environment_that_cannot_resume(self):
        with patch('transformers.utils.check_torch_load_is_safe', side_effect=ValueError('old torch')):
            with self.assertRaisesRegex(RuntimeError, 'PyTorch >= 2.6'):
                runtime.require_training_resume_support()

    def test_completed_adapter_loads_without_retraining(self):
        store = self.store()
        runtime.save_adapter(FakeModel(), Tokenizer(), store.directory / 'final_adapter',
                             store.manifest_sha256, train_seconds=12.)
        restored = FakeModel()
        with patch.object(runtime.PeftModel, 'from_pretrained', return_value=restored) as load, \
             patch.object(runtime, 'require_training_resume_support', side_effect=AssertionError('no training')):
            model, seconds = runtime.train_or_restore({}, {}, FakeModel(), Tokenizer(), {}, store,
                                                     SimpleNamespace(), lambda _: None)
        self.assertIs(model, restored)
        self.assertEqual(seconds, 12.)
        self.assertFalse(restored.training)
        self.assertFalse(load.call_args.kwargs['is_trainable'])

    def test_adapter_completion_detects_modified_weights(self):
        store = self.store()
        directory = store.directory / 'final_adapter'
        runtime.save_adapter(FakeModel(), Tokenizer(), directory, store.manifest_sha256, train_seconds=12.)
        (directory / 'adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed or incomplete'):
            runtime.train_or_restore({}, {}, FakeModel(), Tokenizer(), {}, store,
                                     SimpleNamespace(), lambda _: None)


if __name__ == '__main__':
    unittest.main()
