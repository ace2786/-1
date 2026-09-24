"""Ollama client: dual-model routing, JSON mode, retries (stability)."""
import asyncio
import json
import time
import httpx
from .config import settings
from .observe import Metrics, audit, log


class OllamaError(RuntimeError):
    pass


def _obs(name):
    class T:
        def __enter__(self_inner):
            self_inner.t0 = time.perf_counter()
            return self_inner

        def __exit__(self_inner, et, e, tb):
            Metrics.observe(name, time.perf_counter() - self_inner.t0)
            return False
    return T()


async def generate(prompt: str, *, system: str | None = None, model: str | None = None,
                   heavy: bool = False, json_mode: bool = False, temperature: float = 0.2,
                   timeout_s: float = 300.0, max_retries: int = 2) -> str:
    """Call Ollama /api/generate. heavy=True routes to the deep-analysis model."""
    if model is None:
        model = settings.heavy_model if heavy else settings.light_model
    body = {"model": model, "prompt": prompt, "stream": False,
            # think:false -> qwen3 hybrid models skip CoT chain (fast + pure JSON out)
            "think": False,
            "options": {"temperature": temperature}}
    if system:
        body["system"] = system
    if json_mode:
        body["format"] = "json"

    last_err = None
    for attempt in range(max_retries + 1):
        _t0 = time.perf_counter()
        try:
            with _obs("llm." + model):
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    r = await client.post(settings.ollama_base_url + "/api/generate", json=body)
                    if r.status_code != 200:
                        raise OllamaError("ollama %s: %s" % (r.status_code, r.text[:200]))
                    out = r.json().get("response", "")
            Metrics.inc("llm_calls")
            Metrics.inc("llm_calls." + model)
            audit("llm_generate", actor="system", model=model, attempt=attempt, ok=True,
                  duration_s=round(time.perf_counter() - _t0, 2))
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("llm_retry", model=model, attempt=attempt, error=str(e)[:200])
            audit("llm_generate", actor="system", model=model, attempt=attempt,
                  ok=False, error=str(e)[:300])
            await asyncio.sleep(1.5 * (attempt + 1))
    raise OllamaError("all attempts failed for %s: %s" % (model, last_err))


async def generate_json(prompt: str, **kw):
    """generate() with json_mode; tolerant of fenced/extra text.

    If the model returns an empty object/array (qwen3 sometimes does on long
    structured prompts), retry once with a stricter instruction suffix.
    """
    raw = await generate(prompt, json_mode=True, **kw)
    _probe = raw.strip().replace(" ", "")
    if _probe in ("{}", "{\"items\":[]}", "[]"):
        raw = await generate(prompt + "\n\n（上次输出为空。请重新仔细审阅卷宗材料，必须给出非空结果；若确实无矛盾/无发现，也要在 items 里给出一条说明性条目。）",
                             json_mode=True, **kw)
    s = raw.strip()
    if s.startswith("```"):
        parts = s.split("\n", 1)
        s = parts[1] if len(parts) > 1 else s
        s = s.rsplit("```", 1)[0]
    a, b = s.find("{"), s.rfind("}")
    la, lb = s.find("["), s.rfind("]")
    if a != -1 and b > a and (la == -1 or a < la):
        s = s[a:b + 1]
    elif la != -1 and lb > la:
        s = s[la:lb + 1]
    return json.loads(s)


_embed_fallback_warned = False


async def embed(texts: list[str]) -> list[list[float]]:
    """Batch embeddings; falls back to deterministic hashing vectors offline."""
    global _embed_fallback_warned
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            sem = asyncio.Semaphore(4)

            async def one(t: str):
                async with sem:
                    r = await client.post(settings.ollama_base_url + "/api/embeddings",
                                          json={"model": settings.embed_model, "prompt": t})
                    if r.status_code != 200:
                        raise OllamaError("embed failed: " + r.text[:120])
                    return r.json()["embedding"]
            vecs = await asyncio.gather(*[one(t) for t in texts])
        Metrics.inc("embed_calls")
        return vecs
    except Exception as e:  # noqa: BLE001
        if not _embed_fallback_warned:
            _embed_fallback_warned = True
            log.warning("embed_fallback_to_hashing", reason=str(e)[:150])
            audit("embed_fallback", actor="system", error=str(e)[:200])
        Metrics.inc("embed_fallback_calls")
        return [_hash_vec(t) for t in texts]


def _hash_vec(t: str, dim: int = 512) -> list[float]:
    """Deterministic bag-of-hashed-ngrams vector (offline, no model needed)."""
    import hashlib as _h
    import math
    v = [0.0] * dim
    tt = t.replace("\n", "")
    grams = [tt[i:i + 2] for i in range(max(1, len(tt) - 1))]
    for g in grams:
        idx = int.from_bytes(_h.md5(g.encode("utf-8")).digest()[:4], "big") % dim
        v[idx] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


async def ensure_models() -> dict:
    """Check which configured models are present; report missing."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(settings.ollama_base_url + "/api/tags")
            have = {m["name"] for m in r.json().get("models", [])}
    except Exception:  # noqa: BLE001
        return {"have": [], "missing": [settings.heavy_model, settings.light_model,
                                        settings.embed_model], "server": "down"}

    def present(want: str) -> bool:
        base = want.split(":")[0]
        return want in have or any(h.split(":")[0] == base for h in have)

    need = [settings.heavy_model, settings.light_model, settings.embed_model]
    return {"have": sorted(have), "missing": [n for n in need if not present(n)]}
