"""Embedding index + retrieval with page anchors (RAG core, traceability).

Design: no external vector DB process needed — numpy cosine over in-memory
matrices cached to disk. ChromaDB is optional; we keep the zero-service path
for stability & offline guarantee.
"""
import asyncio, json, hashlib
from pathlib import Path
import numpy as np
from ..config import settings
from ..llm import embed
from ..observe import Metrics, log
from .store import load_chunks, save_chunks, _case_dir


def _emb_cache(case_id: str) -> Path:
    return _case_dir(case_id) / "chunks.npz"


async def build_index(case_id: str):
    """Chunk parsed docs -> store chunks.jsonl -> embed -> cache vectors."""
    from .parsing import parse_document
    from .store import load_case, add_doc
    case = load_case(case_id)
    all_chunks: list[dict] = []
    for dmeta in case.docs:
        from .store import DocMeta
        dm = DocMeta(**dmeta) if isinstance(dmeta, dict) else dmeta
        base = _case_dir(case_id) / "documents"
        tj = base / (dm.doc_id + ".text.json")
        if not tj.exists():
            continue
        data = json.loads(tj.read_text(encoding="utf-8"))
        for pg in data["pages"]:
            text = pg.get("text", "")
            for ci, seg in enumerate(_chunk_text(text)):
                all_chunks.append({
                    "chunk_id": hashlib.md5(f'{dm.doc_id}:{pg['page']}:{ci}'.encode()).hexdigest()[:12],
                    "doc_id": dm.doc_id, "doc_name": dm.name,
                    "page": pg["page"], "start_line": ci * 30 + 1,
                    "text": seg,
                })
    save_chunks(case_id, all_chunks)
    if not all_chunks:
        return 0
    vecs = await embed([c["text"] for c in all_chunks])
    np.savez_compressed(_emb_cache(case_id),
                        ids=np.array([c["chunk_id"] for c in all_chunks]),
                        vecs=np.asarray(vecs, dtype=np.float32))
    Metrics.inc("index_builds"); log.info("index_built", case_id=case_id, chunks=len(all_chunks))
    return len(all_chunks)


def _chunk_text(text: str, max_chars: int = 600) -> list[str]:
    """Split on paragraph/line boundaries keeping ~<=max_chars per chunk."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                out.append(buf)
            while len(p) > max_chars:
                out.append(p[:max_chars]); p = p[max_chars:]
            buf = p
    if buf:
        out.append(buf)
    return out


_state: dict[str, tuple[list[dict], dict[str, np.ndarray]]] = {}


def _load(case_id: str):
    chunks = load_chunks(case_id)
    cache = _emb_cache(case_id)
    if not chunks or not cache.exists():
        return None
    z = np.load(cache, allow_pickle=False)
    id2vec = {str(i): v for i, v in zip(z["ids"], z["vecs"])}
    _state[case_id] = (chunks, id2vec)
    return _state[case_id]


async def search(case_id: str, query: str, k: int = 6) -> list[dict]:
    """Return top-k chunks with scores; each carries doc/page anchor fields."""
    st = _load(case_id) or _state.get(case_id)
    if not st:
        return []
    chunks, id2vec = st
    qv = np.asarray((await embed([query]))[0], dtype=np.float32)
    scored = []
    for c in chunks:
        v = id2vec.get(c["chunk_id"])
        if v is None:
            continue
        s = float(np.dot(qv, v) / (np.linalg.norm(qv) * np.linalg.norm(v) + 1e-9))
        scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    res = [{**c, "score": round(s, 4)} for s, c in scored[:k]]
    Metrics.inc("rag_searches")
    return res


def get_chunk(case_id: str, chunk_id: str) -> dict | None:
    for c in load_chunks(case_id):
        if c["chunk_id"] == chunk_id:
            return c
    return None