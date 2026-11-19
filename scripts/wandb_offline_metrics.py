"""Read logged metrics from a local (offline) wandb run without syncing it.

The .wandb file is a LevelDB-style log (32 KiB blocks, 7-byte record headers) of wandb Record protobufs, after a
7-byte file header. This wandb version no longer ships a reader, so this is a minimal one.

Usage:
    python scripts/wandb_offline_metrics.py wandb/offline-run-<...>-<id> [--keys clipfrac approx_kl ...]
Prints one line per logged step for the requested keys (default: PPO/GRPO training diagnostics).
"""
import argparse
import glob
import json
import os
import struct

from wandb.proto import wandb_internal_pb2

BLOCK = 32 * 1024
HEADER = 7
FULL, FIRST, MIDDLE, LAST = 1, 2, 3, 4
DEFAULT_KEYS = ["epoch", "clipfrac", "approx_kl", "policy_loss", "zero_std_ratio", "reward_std_mean", "reward_ocr",
                "eval_reward_ocr"]


def records(path):
    with open(path, "rb") as f:
        data = f.read()
    pos, buf = HEADER, b""  # skip the file header
    while pos + HEADER <= len(data):
        block_left = BLOCK - (pos % BLOCK)
        if block_left < HEADER:  # block trailer padding
            pos += block_left
            continue
        _crc, length, rtype = struct.unpack("<IHB", data[pos:pos + HEADER])
        payload = data[pos + HEADER:pos + HEADER + length]
        pos += HEADER + length
        if rtype == 0 and length == 0:
            continue
        if rtype == FULL:
            yield payload
        elif rtype == FIRST:
            buf = payload
        elif rtype == MIDDLE:
            buf += payload
        elif rtype == LAST:
            yield buf + payload
            buf = b""


def history(run_dir):
    path = glob.glob(os.path.join(run_dir, "*.wandb"))[0]
    for raw in records(path):
        rec = wandb_internal_pb2.Record()
        try:
            rec.ParseFromString(raw)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in rec.history.item:
            key = item.key or "/".join(item.nested_key)
            try:
                row[key] = json.loads(item.value_json)
            except ValueError:
                pass
        yield row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    args = p.parse_args()
    print("step " + " ".join(f"{k:>16s}" for k in args.keys))
    for row in history(args.run_dir):
        if not any(k in row for k in args.keys):
            continue
        cells = []
        for k in args.keys:
            v = row.get(k)
            cells.append(f"{v:>16.4g}" if isinstance(v, (int, float)) else f"{'':>16s}")
        print(f"{row.get('_step', ''):>4} " + " ".join(cells))


if __name__ == "__main__":
    main()
