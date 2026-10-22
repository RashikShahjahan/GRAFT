"""The sweep's CLI must work without installed training dependencies."""

import subprocess
import sys
from pathlib import Path


def test_parser_without_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", "-c", """
import sys
from train_args import make_parser
args = make_parser().parse_args([
    '--max_steps', '5', '--retain_kl_coef', '0.05',
    '--mlp_config', 'm16', '--lora_config', 'r32', '--no_wandb',
])
assert args.max_steps == 5
assert args.retain_kl_coef == 0.05
assert args.mlp_config == 'm16'
assert args.lora_config == 'r32'
assert args.no_wandb
assert not {'train', 'torch', 'trl', 'transformers', 'vllm_utils'} & sys.modules.keys()
"""],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
