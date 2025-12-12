#!/usr/bin/env python3
import argparse
import copy
import os
import subprocess
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sweep_batch_sizes.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    batch_sizes = cfg["sweep"]["batch_sizes"]
    base_run = cfg["output"].get("run_name", "sweep")
    ckpt_dir = cfg["output"].get("checkpoints_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # We reuse scripts/train.py by writing temporary per-sweep configs.
    tmp_dir = os.path.join("results", "_tmp_sweep_configs")
    os.makedirs(tmp_dir, exist_ok=True)

    for bs in batch_sizes:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["train"]["batch_size"] = int(bs)
        run_cfg["output"]["run_name"] = f"{base_run}_bs{bs}"

        tmp_path = os.path.join(tmp_dir, f"cfg_bs{bs}.yaml")
        with open(tmp_path, "w") as f:
            yaml.safe_dump(run_cfg, f, sort_keys=False)

        print(f"\n=== Running batch_size={bs} ===")
        subprocess.check_call(["python", "scripts/train.py", "--config", tmp_path])

if __name__ == "__main__":
    main()