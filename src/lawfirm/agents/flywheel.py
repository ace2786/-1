"""Data flywheel: accumulate human feedback, retrieve relevant corrections.

feedback records (flywheel.jsonl per case): {ts, question, answer, rating(+1/-1), correction}
On every ask(), top-K semantically similar *corrections* are retrieved and injected
as expert guidance; the response carries `used_memories` so the UI can show a badge.
"""
import json
from pathlib import Path
from ..config import settings


def _fp(case_id: str) -> Path:
    from ..rag.store import _case_dir as cd
    return cd(case_id) / "flywheel.jsonl"


def record(case_id: str, question: str, answer: str, rating: int, correction: str = "") -> dict:
    rec = {"ts": __import__("datetime").datetime.now().isoformat(),
           "question": question, "answer": answer[:500], "rating": rating, "correction": correction}
    fp = _fp(case_id)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_all() -> list[dict]:
    """Every feedback record across all cases (small-scale demo corpus)."""
    out = []
    base = Path(settings.cases_dir)
    if not base.exists():
        return out
    for fp in base.glob("*/flywheel.jsonl"):
        try:
            case_id = fp.parent.name
            for ln in fp.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    r["case_id"] = case_id
                    out.append(r)
        except Exception:  # noqa: BLE001
            continue
    return out


async def recall(query: str, k: int = 3) -> list[dict]:
    """Rank stored corrections by embedding similarity to query (fallback: keyword overlap)."""
    items = [r for r in load_all() if r.get("rating") == -1 and r.get("correction")]
    if not items:
        return []
    try:
        from ..llm import embed
        import numpy as np
        qv = np.asarray((await embed([query]))[0], dtype=np.float32)
        vecs = await embed([i["question"] + " " + i["correction"] for i in items])
        scored = []
        for it, v in zip(items, vecs):
            v = np.asarray(v, dtype=np.float32)
            if v.shape != qv.shape:
                raise ValueError("dim mismatch")
            s = float(np.dot(qv, v) / (np.linalg.norm(qv) * np.linalg.norm(v) + 1e-9))
            scored.append((s, it))
        scored.sort(key=lambda x: -x[0])
        return [{**it, "similarity": round(s, 3)} for s, it in scored[:k] if s > 0.25]
    except Exception:  # noqa: BLE001
        # offline fallback: token-overlap scoring
        qt = set(query)
        out = []
        for it in items:
            ct = set(it["question"] + it["correction"])
            j = len(qt & ct) / max(1, len(qt | ct))
            if j > 0.08:
                out.append({**it, "similarity": round(j, 3)})
        out.sort(key=lambda x: -x["similarity"])
        return out[:k]
