import json
import math
import os
import re
from pathlib import Path

# Keep numerical libraries from starting many threads on small free instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("ORBIS_MODEL_DIR", BASE / "model"))
WEIGHTS_PATH = Path(os.getenv("ORBIS_WEIGHTS", MODEL_DIR / "orbis_step6_weights.npz"))
TOKENIZER_PATH = Path(os.getenv("ORBIS_TOKENIZER", MODEL_DIR / "tokenizer.json"))

class ByteBPETokenizer:
    def __init__(self, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != "orbis-bpe-v1":
            raise RuntimeError("Unsupported Orbis tokenizer format")
        self.vocab_size = int(data["vocab_size"])
        self.merges = data["merges"]
        self.ranks, self.children = {}, {}
        for rank, item in enumerate(self.merges):
            left, right, new = int(item["left"]), int(item["right"]), int(item["new"])
            self.ranks[(left, right)] = rank
            self.children[new] = (left, right)

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            best_rank, best_pair = None, None
            for a, b in zip(tokens, tokens[1:]):
                rank = self.ranks.get((a, b))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, (a, b)
            if best_pair is None:
                break
            a, b = best_pair
            new_id = int(self.merges[best_rank]["new"])
            merged, i = [], 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == a and tokens[i+1] == b:
                    merged.append(new_id); i += 2
                else:
                    merged.append(tokens[i]); i += 1
            tokens = merged
        return tokens

    def decode(self, ids):
        output = bytearray()
        def emit(token_id):
            if 0 <= token_id < 256:
                output.append(token_id)
            elif token_id in self.children:
                left, right = self.children[token_id]
                emit(left); emit(right)
            else:
                output.extend(b"?")
        for token_id in ids:
            emit(int(token_id))
        return output.decode("utf-8", errors="replace")


def linear(x, prefix, weights):
    return x @ weights[prefix + ".weight"].T + weights[prefix + ".bias"]

def layer_norm(x, prefix, weights, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weights[prefix + ".weight"] + weights[prefix + ".bias"]

def gelu(x):
    # Tanh approximation is numerically close to PyTorch's exact GELU.
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

class OrbisNumpyModel:
    def __init__(self, weights_path, metadata_path):
        self.wfile = np.load(weights_path, allow_pickle=False)
        self.w = {k: self.wfile[k] for k in self.wfile.files}
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.vocab_size = int(meta["vocab_size"])
        self.block_size = int(meta["block_size"])
        self.d_model = int(meta["d_model"])
        self.n_heads = int(meta["n_heads"])
        self.n_layers = int(meta["n_layers"])
        self.head_dim = self.d_model // self.n_heads

    def __call__(self, ids):
        # ids is a 1-D list of token IDs. The network is the same Step 6 architecture.
        ids = ids[-self.block_size:]
        x = self.w["token.weight"][np.asarray(ids, dtype=np.int64)]
        x = x + self.w["pos.weight"][np.arange(len(ids), dtype=np.int64)]
        length = x.shape[0]
        for layer in range(self.n_layers):
            p = f"blocks.{layer}"
            h = layer_norm(x, p + ".ln1", self.w)
            qkv = linear(h, p + ".attn.qkv", self.w)
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(length, self.n_heads, self.head_dim).transpose(1, 0, 2)
            k = k.reshape(length, self.n_heads, self.head_dim).transpose(1, 0, 2)
            v = v.reshape(length, self.n_heads, self.head_dim).transpose(1, 0, 2)
            scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.head_dim)
            scores = np.where(np.triu(np.ones((length, length), dtype=bool), 1)[None, :, :], -1e9, scores)
            scores = scores - scores.max(axis=-1, keepdims=True)
            attn = np.exp(scores)
            attn = attn / attn.sum(axis=-1, keepdims=True)
            z = (attn @ v).transpose(1, 0, 2).reshape(length, self.d_model)
            x = x + linear(z, p + ".attn.out", self.w)
            h = layer_norm(x, p + ".ln2", self.w)
            h = gelu(linear(h, p + ".ff.0", self.w))
            h = linear(h, p + ".ff.2", self.w)
            x = x + h
        x = layer_norm(x, "ln", self.w)
        return x[-1] @ self.w["head.weight"].T


# ---- Orbis v2: the model you trained yourself with train.py (put converted files in model2/) ----
ORBIS2_DIR = Path(os.getenv("ORBIS2_DIR", BASE / "model2"))
orbis2 = None
if (ORBIS2_DIR / "orbis_weights.npz").exists():
    try:
        from orbis_np import OrbisNP
        orbis2 = OrbisNP(ORBIS2_DIR)
        orbis2.tok   # load the tokenizer now so a problem shows up at startup, not on the first request
        print(f"[orbis] v2 model loaded: {orbis2.num_params:,} parameters")
    except Exception as exc:
        orbis2 = None
        print(f"[orbis] could not load model2/: {type(exc).__name__}: {exc}")

# ---- Orbis Step 6 (the original 387K model); optional once v2 is in place ----
try:
    tokenizer = ByteBPETokenizer(TOKENIZER_PATH)
    model = OrbisNumpyModel(WEIGHTS_PATH, MODEL_DIR / "model_metadata.json")
except Exception as exc:
    tokenizer = model = None
    if orbis2 is None:
        raise
LOCAL_LABEL = (f"Studio Orbis v2 ({orbis2.num_params:,} parameters)" if orbis2
               else "Studio Orbis Step 6 (386,688 parameters)")
LOCAL_PARAMS = orbis2.num_params if orbis2 else 386688
app = FastAPI(title="Studio Orbis AI", version="0.7.1-render")
origins = [x.strip() for x in os.getenv("ORBIS_ALLOWED_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=["*"] if origins == ["*"] else origins,
                   allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])


# ---- Optional Gemma runtime (local folder or Hugging Face model repository) ----
# Keep this disabled until the large model weights and matching tokenizer are available.
GEMMA_MODEL = os.getenv("ORBIS_GEMMA_MODEL", "").strip()
GEMMA_TOKEN = os.getenv("HF_TOKEN", "").strip()
gemma_model = None
gemma_tokenizer = None
gemma_device = None
if GEMMA_MODEL:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        gemma_tokenizer = AutoTokenizer.from_pretrained(
            GEMMA_MODEL, token=GEMMA_TOKEN or None, trust_remote_code=False
        )
        # Let Transformers choose the checkpoint's stored dtype. CPU hosting still
        # needs several GB of RAM for a 1B model; this is not suitable for tiny Render instances.
        gemma_model = AutoModelForCausalLM.from_pretrained(
            GEMMA_MODEL, token=GEMMA_TOKEN or None, torch_dtype="auto",
            low_cpu_mem_usage=True, trust_remote_code=False
        )
        gemma_device = "cuda" if torch.cuda.is_available() else "cpu"
        gemma_model.to(gemma_device)
        gemma_model.eval()
        print(f"[orbis] Gemma model loaded from {GEMMA_MODEL} on {gemma_device}")
    except Exception as exc:
        gemma_model = gemma_tokenizer = None
        print(f"[orbis] Gemma model not loaded: {type(exc).__name__}: {exc}")


def gemma_generate(message, max_new_tokens=256, temperature=0.7):
    if gemma_model is None or gemma_tokenizer is None:
        raise RuntimeError("Gemma is not configured")
    import torch
    prompt = "You are Studio Orbis AI. Answer helpfully and clearly in English unless the user asks for another language.\n\nUser: " + message + "\nAssistant:"
    inputs = gemma_tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(gemma_device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = gemma_model.generate(
            **inputs, max_new_tokens=max(1, min(int(max_new_tokens), 1024)),
            do_sample=temperature > 0, temperature=max(0.1, float(temperature)),
            pad_token_id=gemma_tokenizer.eos_token_id
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return gemma_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ---- Backend config: point Studio Orbis at an open-source model (e.g. gpt-oss) ----
# Any OpenAI-compatible server works: Ollama, vLLM, LM Studio, or a hosted gpt-oss provider.
BACKEND_URL = os.getenv("ORBIS_BACKEND_URL", "").rstrip("/")      # e.g. http://localhost:11434/v1
BACKEND_MODEL = os.getenv("ORBIS_BACKEND_MODEL", "gpt-oss:20b")
BACKEND_KEY = os.getenv("ORBIS_BACKEND_API_KEY", "")
REASONING = os.getenv("ORBIS_REASONING_EFFORT", "")                # low | medium | high (optional)
SYSTEM_PROMPT = os.getenv("ORBIS_SYSTEM_PROMPT",
    "You are Studio Orbis AI, a helpful assistant. Answer clearly and concisely.")

HARD_REASONING = os.getenv("ORBIS_HARD_REASONING", "medium")      # reasoning effort for hard questions
EASY_REASONING = os.getenv("ORBIS_EASY_REASONING", "low")         # reasoning effort for easy rewrites
HARD_MAX_TOKENS = int(os.getenv("ORBIS_HARD_MAX_TOKENS", "1500"))   # server-side cap for hard answers
EASY_MAX_TOKENS = int(os.getenv("ORBIS_EASY_MAX_TOKENS", "600"))    # server-side cap for easy rewrites
ROUTER_MODE = os.getenv("ORBIS_ROUTER", "heuristic")              # heuristic | llm | always_hard | always_easy

REFINE_PROMPT = os.getenv("ORBIS_REFINE_PROMPT",
    "You are Studio Orbis AI. A small in-house model wrote a rough draft answer, which may be "
    "wrong, incomplete or nonsense. Rewrite it into a clear, correct, concise final answer to the "
    "user's question. Keep any useful ideas from the draft, but fix or discard anything wrong. "
    "Do not mention the draft.")

def backend_chat(messages, max_tokens=512, temperature=0.7, reasoning=""):
    headers = {"Content-Type": "application/json"}
    if BACKEND_KEY:
        headers["Authorization"] = f"Bearer {BACKEND_KEY}"
    body = {"model": BACKEND_MODEL, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature}
    if reasoning and reasoning.lower() != "off":   # set the env var to "off" if your provider rejects it
        body["reasoning_effort"] = reasoning
    r = httpx.post(BACKEND_URL + "/chat/completions", json=body, headers=headers, timeout=180)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()

HARD_KEYWORDS = ("why", "how do", "how does", "how can", "how would", "explain", "compare", "difference between",
                 "analy", "step by step", "prove", "derive", "calculate", "solve", "debug", "code", "function",
                 "algorithm", "write a", "essay", "summarize", "plan", "strategy", "pros and cons", "should i",
                 "design", "optimi", "implement", "translate", "recommend")

def is_hard_heuristic(text):
    """Cheap difficulty guess. Returns (is_hard, score)."""
    t = text.lower().strip()
    words = len(t.split())
    score = 0
    if words > 25: score += 2
    elif words > 12: score += 1
    if any(k in t for k in HARD_KEYWORDS): score += 2
    if re.search(r"\d\s*[-+*/^=x]\s*\d", t) or "```" in t or "def " in t or "{" in t: score += 2
    if t.count("?") > 1 or "\n" in t: score += 1
    return score >= 2, score

def is_hard_llm(text):
    """Ask the backend model to label the question EASY or HARD; fall back to the heuristic."""
    try:
        out = backend_chat(
            [{"role": "system", "content": "Classify the user's question. Reply with exactly one word: "
              "EASY (greeting, simple fact, short chat) or HARD (reasoning, multi-step, code, math, "
              "analysis, long or open-ended)."},
             {"role": "user", "content": text}],
            max_tokens=200, temperature=0.0, reasoning="low")
        up = out.upper()
        if "HARD" in up: return True, "llm"
        if "EASY" in up: return False, "llm"
    except Exception:
        pass
    return is_hard_heuristic(text)

def route_question(text):
    if ROUTER_MODE == "always_hard": return True, "forced"
    if ROUTER_MODE == "always_easy": return False, "forced"
    if ROUTER_MODE == "llm": return is_hard_llm(text)
    return is_hard_heuristic(text)

def tiny_generate(message, max_new_tokens=48, temperature=0.8, top_k=30):
    """Draft text from the homemade model (v2 if loaded, otherwise Step 6)."""
    if orbis2 is not None:
        return orbis2.complete(message.strip(), max_new_tokens, temperature, top_k)
    ids = tokenizer.encode("Question: " + message.strip() + "\nAnswer:")
    if not ids: ids = [32]
    generated = []
    for _ in range(max(1, min(max_new_tokens, 96))):
        logits = model(ids + generated) / temperature
        k = min(top_k, logits.shape[-1])
        top = np.argpartition(logits, -k)[-k:]
        vals = logits[top]
        vals = vals - vals.max()
        probs = np.exp(vals); probs /= probs.sum()
        generated.append(int(np.random.choice(top, p=probs)))
    return tokenizer.decode(generated).strip()

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int | None = Field(default=None, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)
    top_k: int = Field(default=30, ge=1, le=100)
    debug: bool = False   # include routing info and the tiny-model draft in the reply

@ app.get("/")
def root():
    return {"name":"Studio Orbis AI", "status":"online", "step":10,
            "model_parameters":LOCAL_PARAMS, "runtime":"NumPy CPU (no PyTorch runtime)",
            "message":"Homemade experimental model loaded. See /docs."}

@ app.get("/health")
def health():
    return {"status":"ok", "model_loaded":True, "runtime":"numpy-cpu" if gemma_model is None else f"transformers-{gemma_device}", "parameters":LOCAL_PARAMS, "local_model": LOCAL_LABEL, "gemma_model": GEMMA_MODEL or None, "gemma_loaded": gemma_model is not None, "backend": BACKEND_MODEL if BACKEND_URL else None}

@ app.get("/demo", response_class=HTMLResponse)
def demo():
    return """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Studio Orbis AI</title>
    <style>body{font:16px system-ui;max-width:760px;margin:40px auto;padding:0 16px;background:#10131a;color:#eef2ff}main{background:#1b2230;padding:22px;border-radius:16px}textarea{width:100%;min-height:90px;box-sizing:border-box;padding:12px;border-radius:10px;background:#111827;color:white}button{margin-top:10px;padding:11px 18px;border:0;border-radius:9px;background:#7c9cff;font-weight:700}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#111827;padding:14px;border-radius:10px;min-height:60px}</style></head>
    <body><main><h1>Studio Orbis AI</h1><p>Step 7 Render server test</p><textarea id="m" placeholder="Ask Studio Orbis..."></textarea><br><button id="b">Send</button><h3>Response</h3><pre id="o">Ready.</pre></main>
    <script>b.onclick=async()=>{let m=document.getElementById('m').value.trim();if(!m)return;b.disabled=true;o.textContent='Generating...';try{let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})});let d=await r.json();o.textContent=r.ok?d.response:(d.detail||JSON.stringify(d));}catch(e){o.textContent=String(e)}finally{b.disabled=false}}</script></body></html>"""

@ app.post("/chat")
def chat(request: ChatRequest):
    question = request.message.strip()

    # No OpenAI-compatible backend: use Gemma when configured, otherwise the homemade Orbis checkpoint.
    if not BACKEND_URL:
        try:
            if gemma_model is not None:
                answer = gemma_generate(question, request.max_new_tokens or 256, request.temperature)
                return {"response": answer or "(The model did not generate text. Try another prompt.)",
                        "route": "gemma", "model": GEMMA_MODEL}
            answer = tiny_generate(question, request.max_new_tokens or 48, request.temperature, request.top_k)
            return {"response": answer or "(The model did not generate text. Try another prompt.)",
                    "route": "local", "model": LOCAL_LABEL,
                    "note": "Experimental small homemade model; output may be incomplete or incorrect."}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Generation failed: {type(exc).__name__}")

    hard, why = route_question(question)

    # HARD -> gpt-oss answers directly.
    if hard:
        try:
            answer = backend_chat(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
                max_tokens=HARD_MAX_TOKENS, temperature=request.temperature,
                reasoning=HARD_REASONING)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Backend model error: {type(exc).__name__}")
        out = {"response": answer or "(Empty response from backend.)", "route": "hard", "model": BACKEND_MODEL}
        if request.debug: out["router"] = why
        return out

    # EASY -> Orbis drafts, gpt-oss rewrites ("translates") the draft into a good answer.
    try:
        draft = tiny_generate(question, request.max_new_tokens or 48, request.temperature, request.top_k)
    except Exception:
        draft = ""
    try:
        answer = backend_chat(
            [{"role": "system", "content": REFINE_PROMPT},
             {"role": "user", "content": f"Question: {question}\n\nRough draft:\n{draft or '(none)'}\n\nFinal answer:"}],
            max_tokens=EASY_MAX_TOKENS, temperature=request.temperature,
            reasoning=EASY_REASONING)
    except Exception as exc:
        # Backend down: degrade gracefully to the raw Orbis draft.
        if draft:
            return {"response": draft, "route": "easy-fallback", "model": LOCAL_LABEL,
                    "note": f"Backend unavailable ({type(exc).__name__}); showing raw draft."}
        raise HTTPException(status_code=502, detail=f"Backend model error: {type(exc).__name__}")
    out = {"response": answer or draft, "route": "easy", "model": f"{LOCAL_LABEL} draft + {BACKEND_MODEL}"}
    if request.debug: out.update({"router": why, "draft": draft})
    return out

@ app.post("/generate")
def generate_endpoint(request: ChatRequest):
    result = chat(request)
    return {"text":result["response"]}
