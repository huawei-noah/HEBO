# MIT License
#
# Copyright (c) 2024, Huawei Technologies Co., Ltd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import shutil
import sys
import json

from safetensors import safe_open
from safetensors.torch import save_file


def load_model(file_path):
    model = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            model[key] = f.get_tensor(key)
    return model

def convert_checkpoint(source_dir, destination_dir):
    """
    Copy all contents from source_dir to destination_dir.

    :param source_dir: Path to the checkpoint directory to copy from.
    :param destination_dir: Path to the destination directory.
    """
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    # Ensure the destination directory exists, create it if it doesn't
    os.makedirs(destination_dir, exist_ok=True)

    # Copy the contents from source to destination directory
    try:
        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            destination_item = os.path.join(destination_dir, item)

            # If the item is a directory, copy it recursively
            if os.path.isdir(source_item):
                shutil.copytree(source_item, destination_item, dirs_exist_ok=True)
            else:
                shutil.copy2(source_item, destination_item)

        print(f"Successfully copied contents from '{source_dir}' to '{destination_dir}'.")

    except Exception as e:
        print(f"An error occurred while copying: {e}")
        sys.exit(1)

    with open(os.path.join(destination_dir, "config.json"), 'r', encoding='utf-8') as file:
        config = json.load(file)

    if "num_hidden_layers" in config:
        del config["num_hidden_layers"]
    if "_name_or_path" in config:
        del config["_name_or_path"]

    config["model_type"] = "moa_spec"
    config["architectures"] = ["MOASpecModel"]

    model = load_model(os.path.join(destination_dir, "model.safetensors"))

    assert model['self_attention.self_attn.q_proj.weight'].shape[1] == model['self_attention.self_attn.k_proj.weight'].shape[1]
    assert model['self_attention.self_attn.q_proj.weight'].shape[1] == config['hidden_size']

    del config["attention_bias"]
    del config["attention_dropout"]
    del config["bos_token_id"]
    del config["eos_token_id"]
    del config["initializer_range"]
    del config["intermediate_size"]
    del config["pretraining_tp"]
    del config["tie_word_embeddings"]
    del config["use_cache"]
    del config["num_key_value_heads"]

    config["self_attention_num_key_value_heads"] = model['self_attention.self_attn.k_proj.weight'].shape[0] // (model['self_attention.self_attn.q_proj.weight'].shape[1] // config['num_attention_heads'])
    config["self_attention_intermediate_size"] = model['self_attention.mlp.up_proj.weight'].shape[0]

    config["cross_attention_num_key_value_heads"] = model['cross_attention.self_attn.k_proj.weight'].shape[0] // (model['cross_attention.self_attn.q_proj.weight'].shape[1] // config['num_attention_heads'])
    config["cross_attention_intermediate_size"] = model['cross_attention.mlp.up_proj.weight'].shape[0]

    config["disable_LSA"] = not any(["layer_self_attention" in k for k in model.keys()])
    if not config["disable_LSA"]:
        config["kv_hidden_size"] = model['layer_self_attention.self_attn.q_proj.weight'].shape[0]
        config["layer_self_attention_num_key_value_heads"] = model['layer_self_attention.self_attn.k_proj.weight'].shape[0] // (model['layer_self_attention.self_attn.q_proj.weight'].shape[1] // config['num_attention_heads'])
        config["layer_self_attention_intermediate_size"] = model['layer_self_attention.mlp.up_proj.weight'].shape[0]

    with open(os.path.join(destination_dir, "config.json"), 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=2)

    print("Successfully write new config.json")

    save_file(model, os.path.join(destination_dir, "model.safetensors"))
    print("Successfully converted model.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_checkpoint.py <source_checkpoint_dir> <destination_dir>")
        sys.exit(1)

    source_checkpoint_dir = sys.argv[1]
    destination_dir = sys.argv[2]

    convert_checkpoint(source_checkpoint_dir, destination_dir)
