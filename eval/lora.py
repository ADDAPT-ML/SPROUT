"""Load a LoRA checkpoint written by sprout/train.py (lora_iteration_<t>/, the evaluated EMA weights)."""
import os

from peft import PeftConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file


def load_lora(unet, lora_dir):
    unet = get_peft_model(unet, PeftConfig.from_pretrained(lora_dir))
    state = {}
    for key, value in load_file(os.path.join(lora_dir, 'adapter_model.safetensors')).items():
        key = key.replace('.default.weight', '.weight').replace('.default.', '.')
        if '.lora_' in key and not key.startswith('base_model.model.'):
            key = f'base_model.model.{key}'
        state[key] = value
    result = set_peft_model_state_dict(unet, state)
    bad = [k for k in list(result.missing_keys) + list(result.unexpected_keys) if '.lora_' in k]
    if bad:
        raise ValueError(f'LoRA checkpoint did not load cleanly: {bad[:5]}')
    unet.eval()
    return unet
