#!/usr/bin/env python3
import json
import argparse
import sys
from pathlib import Path

def convert_config(input_path, output_path):
    print(f"Reading original config from {input_path}")
    with open(input_path, 'r') as f:
        config = json.load(f)

    # 1. Add vLLM required fields
    if "custom_input_specs" not in config:
        print("Adding custom_input_specs...")
        config["custom_input_specs"] = [
            {
                "name": "combined_embeddings",
                "dim": 2048 # Default fallback, usually overridden by config if present
            }
        ]
        
        # Try to infer dim from config if possible
        if "talker_config" in config:
            tc = config["talker_config"]
            if "text_hidden_size" in tc:
                config["custom_input_specs"][0]["dim"] = tc["text_hidden_size"]
            elif "hidden_size" in tc:
                 config["custom_input_specs"][0]["dim"] = tc["hidden_size"]

    if "custom_outputs" not in config:
        print("Adding custom_outputs...")
        config["custom_outputs"] = ["codes", "next_input_embeddings"]

    # 2. Fix rope_scaling in talker_config
    if "talker_config" in config:
        tc = config["talker_config"]
        if "rope_scaling" in tc and tc["rope_scaling"] is not None:
            rs = tc["rope_scaling"]
            # Check for "interleaved" and add "mrope_interleaved"
            if rs.get("interleaved", False):
                if "mrope_interleaved" not in rs:
                    print("Adding mrope_interleaved=True to rope_scaling...")
                    rs["mrope_interleaved"] = True
            
            # Ensure "mrope_section" exists if it's mrope
            if "mrope_section" in rs:
                pass # Already there

    # 3. Add sampling parameters if missing (optional but good for explicit behavior)
    # vLLM defaults: do_sample=True, temperature=1.0, top_k=50, top_p=1.0
    # Original model hard defaults: temperature=0.9, repetition_penalty=1.05
    if "do_sample" not in config:
        config["do_sample"] = True
    if "temperature" not in config:
        config["temperature"] = 0.9
    if "top_k" not in config:
        config["top_k"] = 50
    if "top_p" not in config:
        config["top_p"] = 1.0
    if "repetition_penalty" not in config:
        config["repetition_penalty"] = 1.05

    print(f"Writing vLLM config to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS HF config to vLLM config")
    parser.add_argument("input_config", help="Path to original HF config.json")
    parser.add_argument("output_config", help="Path to output vLLM config.json")
    args = parser.parse_args()

    convert_config(args.input_config, args.output_config)
