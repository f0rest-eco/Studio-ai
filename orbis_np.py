"""NumPy inference for the Studio Orbis v2 model trained with train.py (no PyTorch needed).
Uses a KV cache, so generating is fast even on a small CPU. Needs the `tokenizers` package for text <-> ids.
"""
import json
from pathlib import Path

import numpy as np


def _rms(x, w, eps=1e-5):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _rope(x, cos, sin):
    # x: (H, T, head_dim); interleaved pairs, same as model.py
    out = np.empty_like(x)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def _sample(logits, temperature, top_k, top_p, rng):
    logits = logits.astype(np.float64) / max(temperature, 1e-5)
    if top_k and top_k < logits.size:
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf
    if top_p < 1.0:
        order = np.argsort(-logits)
        sl = logits[order]
        probs = _softmax(sl)
        sl[(np.cumsum(probs) - probs) > top_p] = -np.inf
        logits = np.full_like(logits, -np.inf)
        logits[order] = sl
    probs = _softmax(logits)
    return int(rng.choice(probs.size, p=probs))


class OrbisNP:
    def __init__(self, model_dir):
        d = Path(model_dir)
        self.cfg = json.loads((d / "config.json").read_text())
        z = np.load(d / "orbis_weights.npz")
        self.w = {k: z[k].astype(np.float32) for k in z.files}
        c = self.cfg
        self.n_layer, self.n_head, self.d_model = c["n_layer"], c["n_head"], c["d_model"]
        self.block_size, self.eos_id = c["block_size"], c.get("eos_id", 1)
        self.num_params = int(sum(v.size for v in self.w.values()))   # tied output head is not stored
        hd = self.d_model // self.n_head
        inv = 1.0 / (c.get("rope_base", 10000.0) ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
        fr = np.outer(np.arange(self.block_size, dtype=np.float32), inv)
        self.cos, self.sin = np.cos(fr).astype(np.float32), np.sin(fr).astype(np.float32)
        self.rng = np.random.default_rng()
        self._tok = None
        self._tok_path = d / "tokenizer.json"

    @property
    def tok(self):
        if self._tok is None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(str(self._tok_path))
        return self._tok

    def forward(self, ids, cache, pos0):
        """ids: list of token ids at positions pos0.. ; cache: per-layer (K, V) or None. Returns last-position logits."""
        w, H = self.w, self.n_head
        d = self.d_model
        hd = d // H
        T = len(ids)
        x = w["tok_emb.weight"][np.asarray(ids)]
        cos, sin = self.cos[pos0:pos0 + T][None], self.sin[pos0:pos0 + T][None]
        for l in range(self.n_layer):
            p = f"blocks.{l}."
            h = _rms(x, w[p + "n1.weight"])
            q, k, v = np.split(h @ w[p + "attn.qkv.weight"].T, 3, axis=1)
            q = _rope(q.reshape(T, H, hd).transpose(1, 0, 2), cos, sin)
            k = _rope(k.reshape(T, H, hd).transpose(1, 0, 2), cos, sin)
            v = v.reshape(T, H, hd).transpose(1, 0, 2)
            if cache[l] is not None:
                k = np.concatenate([cache[l][0], k], axis=1)
                v = np.concatenate([cache[l][1], v], axis=1)
            cache[l] = (k, v)
            S = k.shape[1]
            scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd)                  # (H, T, S)
            hidden = np.arange(S)[None, :] > (pos0 + np.arange(T))[:, None]     # key j hidden from query i if j > pos0+i
            scores = np.where(hidden[None], -1e30, scores)
            att = _softmax(scores) @ v                                          # (H, T, hd)
            x = x + att.transpose(1, 0, 2).reshape(T, d) @ w[p + "attn.proj.weight"].T
            h = _rms(x, w[p + "n2.weight"])
            x = x + (_silu(h @ w[p + "mlp.w1.weight"].T) * (h @ w[p + "mlp.w3.weight"].T)) @ w[p + "mlp.w2.weight"].T
        x = _rms(x[-1:], w["norm_f.weight"])
        return (x @ w["tok_emb.weight"].T)[0]

    def generate_ids(self, prompt_ids, max_new, temperature=0.8, top_k=40, top_p=0.95):
        ids = ([self.eos_id] + list(prompt_ids))[-(self.block_size - 1):]
        max_new = min(max_new, self.block_size - len(ids))
        out = []
        if max_new <= 0:
            return out
        cache = [None] * self.n_layer
        logits = self.forward(ids, cache, 0)
        pos = len(ids)
        for _ in range(max_new):
            nxt = _sample(logits, temperature, top_k, top_p, self.rng)
            if nxt == self.eos_id:
                break
            out.append(nxt)
            logits = self.forward([nxt], cache, pos)
            pos += 1
        return out

    def complete(self, prompt, max_new=60, temperature=0.8, top_k=40, top_p=0.95):
        ids = self.tok.encode(prompt).ids
        out = self.generate_ids(ids, min(max_new, 300), temperature, top_k, top_p)
        return self.tok.decode(out).strip()
