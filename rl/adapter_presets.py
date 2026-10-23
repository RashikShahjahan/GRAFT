"""Adapter presets shared by training, vLLM, and CPU-only launchers."""

LORA_PRESETS = {
    # alpha=rank always (scaling factor = 1)
    "r1":    {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 1, "lora_alpha": 1},
    "r2":    {"retain_rank": 2,  "forget_rank": 2,  "layer_stride": 1, "lora_alpha": 2},
    "r4":    {"retain_rank": 4,  "forget_rank": 4,  "layer_stride": 1, "lora_alpha": 4},
    "r8":    {"retain_rank": 8,  "forget_rank": 8,  "layer_stride": 1, "lora_alpha": 8},
    "r16":   {"retain_rank": 16, "forget_rank": 16, "layer_stride": 1, "lora_alpha": 16},
    "r32":   {"retain_rank": 32, "forget_rank": 32, "layer_stride": 1, "lora_alpha": 32},
    "r1h":   {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 2, "lora_alpha": 1},
    # Asymmetric: retain=32, forget varies
    "r32f16": {"retain_rank": 32, "forget_rank": 16, "layer_stride": 1, "lora_alpha": 32},
    "r32f4":  {"retain_rank": 32, "forget_rank": 4,  "layer_stride": 1, "lora_alpha": 32},
    "r32f1":  {"retain_rank": 32, "forget_rank": 1,  "layer_stride": 1, "lora_alpha": 32},
    "r32f0":  {"retain_rank": 32, "forget_rank": 0,  "layer_stride": 1, "lora_alpha": 32},
    # Legacy aliases (old results reference these names)
    "r1m":   {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 1, "lora_alpha": 1},
    "r8m":   {"retain_rank": 8,  "forget_rank": 8,  "layer_stride": 1, "lora_alpha": 8},
    "r32m":  {"retain_rank": 32, "forget_rank": 32, "layer_stride": 1, "lora_alpha": 32},
    "r1hm":  {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 2, "lora_alpha": 1},
}


MLP_PRESETS = {
    "m5":   {"retain_neurons": 5,   "forget_neurons": 5,   "layer_stride": 1},
    "m10":  {"retain_neurons": 10,  "forget_neurons": 10,  "layer_stride": 1},
    "m16":  {"retain_neurons": 16,  "forget_neurons": 16,  "layer_stride": 1},
    "m30":  {"retain_neurons": 30,  "forget_neurons": 30,  "layer_stride": 1},
    "m32":  {"retain_neurons": 32,  "forget_neurons": 32,  "layer_stride": 1},
    # Asymmetric splits of the canonical m16 budget (32 total neurons/layer):
    # strong-localization cells (2026-07-08). κ_F = 32/n_F (10.67 / 4.0).
    "m29f3": {"retain_neurons": 29, "forget_neurons": 3,  "layer_stride": 1},
    "m24f8": {"retain_neurons": 24, "forget_neurons": 8,  "layer_stride": 1},
    "m64":  {"retain_neurons": 64,  "forget_neurons": 64,  "layer_stride": 1},
    "m64_retain_only": {"retain_neurons": 64, "forget_neurons": 0, "layer_stride": 1},
    "m128": {"retain_neurons": 128, "forget_neurons": 128, "layer_stride": 1},
    "m256": {"retain_neurons": 256, "forget_neurons": 256, "layer_stride": 1},
}
