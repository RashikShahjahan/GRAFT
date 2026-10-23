import json
from pathlib import Path
import tempfile
import unittest

from modal_checkpoint import latest_training_checkpoint


class CheckpointTests(unittest.TestCase):
    def write_checkpoint(self, root, step):
        path = Path(root) / f'checkpoint-{step}'
        path.mkdir()
        for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
            (path / name).write_bytes(b'state')
        (path / 'trainer_state.json').write_text(json.dumps({'global_step': step}))
        (path / 'dual_lora_config.json').write_text('{}')
        return path

    def test_selects_highest_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(latest_training_checkpoint(root))
            self.write_checkpoint(root, 100)
            expected = self.write_checkpoint(root, 200)
            incomplete = self.write_checkpoint(root, 300)
            (incomplete / 'rng_state.pth').unlink()
            self.assertEqual(latest_training_checkpoint(root), str(expected))

    def test_rejects_partial_or_mismatched_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            expected = self.write_checkpoint(root, 100)
            corrupt = self.write_checkpoint(root, 200)
            (corrupt / 'trainer_state.json').write_text('{')
            mismatch = self.write_checkpoint(root, 300)
            (mismatch / 'trainer_state.json').write_text('{"global_step": 200}')
            self.assertEqual(latest_training_checkpoint(root), str(expected))


if __name__ == '__main__':
    unittest.main()
