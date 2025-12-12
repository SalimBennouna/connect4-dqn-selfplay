#!/usr/bin/env python3
import argparse
import torch

def load_checkpoint(path: str):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise ValueError("Unknown checkpoint format")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    sd = load_checkpoint(args.checkpoint)

    # Print all weight tensors
    for k, v in sd.items():
        if "weight" in k:
            print(f"Layer: {k}, Coefficients:")
            print(v)
            print()

if __name__ == "__main__":
    main()