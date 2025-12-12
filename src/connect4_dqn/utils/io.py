import os
import torch


def save_checkpoint(path, state_dict, cfg=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": state_dict, "cfg": cfg}, path)


def load_checkpoint(path, map_location="cpu"):
    obj = torch.load(path, map_location=map_location)
    return obj["state_dict"], obj.get("cfg")