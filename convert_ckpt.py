"""Convert a ckpt.pt from train.py into files the Render server can load (needs only numpy; no PyTorch).

Usage:  python convert_ckpt.py --ckpt ckpt.pt --tokenizer tokenizer.json --out model2
Output: model2/orbis_weights.npz (float16 by default, so it stays under GitHub's 25 MB upload limit),
        model2/config.json, model2/tokenizer.json
"""
import argparse
import collections
import json
import pickle
import shutil
import zipfile
from pathlib import Path

import numpy as np

DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64}


class _Storage:
    def __init__(self, key, kind):
        self.key, self.kind = key, kind


class _Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def _rebuild_tensor(storage, storage_offset, size, stride, *rest):
    return _Tensor(storage, storage_offset, size, stride)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if name.endswith("Storage"):
            return name
        return lambda *a, **k: (module, name, a)

    def persistent_load(self, pid):
        _, kind, key, _loc, _n = pid
        return _Storage(key, kind)


def load_checkpoint(path):
    z = zipfile.ZipFile(path)
    pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
    prefix = pkl[: -len("data.pkl")]
    ck = _Unpickler(z.open(pkl)).load()
    weights = {}
    for name, t in ck["model"].items():
        if t.storage.kind not in DTYPES:
            raise SystemExit(f"Unsupported tensor type {t.storage.kind} for {name}")
        raw = np.frombuffer(z.read(f"{prefix}data/{t.storage.key}"), dtype=DTYPES[t.storage.kind])
        n = int(np.prod(t.size)) if t.size else 1
        arr = raw[t.offset: t.offset + n].reshape(t.size)
        weights[name] = np.array(arr, dtype=np.float32)
    return ck, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--out", default="model2")
    ap.add_argument("--fp32", action="store_true", help="store float32 (twice the size)")
    a = ap.parse_args()

    ck, w = load_checkpoint(a.ckpt)
    w.pop("lm_head.weight", None)          # tied to tok_emb.weight
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = np.float32 if a.fp32 else np.float16
    np.savez_compressed(out / "orbis_weights.npz", **{k: v.astype(dtype) for k, v in w.items()})
    cfg = dict(ck["config"])
    cfg.update({"eos_id": ck.get("eos_id", 1), "trained_steps": ck.get("step"), "val_loss": ck.get("val_loss")})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    shutil.copy(a.tokenizer, out / "tokenizer.json")
    n = sum(v.size for v in w.values())
    size_mb = (out / "orbis_weights.npz").stat().st_size / 1e6
    print(f"parameters: {n:,} | trained steps: {cfg['trained_steps']} | val loss: {cfg['val_loss']:.3f}")
    print(f"wrote {out}/ (weights {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
