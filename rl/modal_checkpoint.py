"""Locate complete adapter-training checkpoints for Modal worker retries."""

import json
from pathlib import Path


def latest_training_checkpoint(output_dir):
    required = ('model.safetensors', 'optimizer.pt', 'scheduler.pt',
                'trainer_state.json', 'dual_lora_config.json', 'rng_state.pth')
    candidates = []
    for path in Path(output_dir).glob('checkpoint-*'):
        suffix = path.name.removeprefix('checkpoint-')
        if not suffix.isdigit():
            continue
        if not all((path / name).is_file() and (path / name).stat().st_size for name in required):
            continue
        try:
            state = json.loads((path / 'trainer_state.json').read_text())
            json.loads((path / 'dual_lora_config.json').read_text())
        except (ValueError, OSError):
            continue
        if state.get('global_step') == int(suffix):
            candidates.append((int(suffix), path))
    return str(max(candidates)[1]) if candidates else None
