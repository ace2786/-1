"""FastAPI app: REST API + health/metrics/audit observability. Local-first security."""
import hashlib, json, uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..config import settings
from ..observe import audit, log, Metrics, traced
from ..rag.store import (new_case, load_case, list_cases, DocMeta,
                         add_doc, all_analyses, _case_dir, save_analysis)
from ..rag.parsing import parse_document, detect_kind
from ..rag.retrieval import build_index, search as rag_search, get_chunk
from ..analysis.reports import run_analysis, run_all, ANALYSIS_KEYS
from ..agents.react import ask as agent_ask
from .. import llm

from ..web.gui import mount as _mount_gui  # noqa: E402

app = FastAPI(title="律所办案辅助系统", version="0.1.0")


async def check_token(request: Request):
    """Optional bearer gate; enabled when LAWFIRM_API_TOKEN is set."""
    if settings.api_token and request.headers.get("X-API-Token") != settings.api_token:
        raise HTTPException(401, "missing/invalid X-API-Token")


class CaseCreate(BaseModel):
    title: str
    charge: str = ""
    suspect: str = ""


class AskBody(BaseModel):
    question: str
    page_context: str | None = None
    heavy: bool = False


class AnalyzeBody(BaseModel):
    key: str | None = None


class FeedbackBody(BaseModel):
    question: str
    answer: str
    rating: int          # 1 good / -1 bad
    correction: str = ""


@app.get("/api/audit/history")
async def audit_history(limit: int = 200, event: str = "", _=Depends(check_token)):
    """Recent audit records across day files (newest first), optional event filter."""
    files = sorted(Path(settings.audit_dir).glob("audit-*.jsonl"), reverse=True)
    out = []
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception:  # noqa: BLE001
            continue
        for ln in reversed(lines):
            if len(out) >= limit:
                break
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if event and r.get("event") != event:
                continue
            out.append(r)
        if len(out) >= limit:
            break
    return {"count": len(out), "records": out}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    html = (Path(__file__).parent.parent / "web" / "templates" / "dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/health")
async def health():
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(settings.ollama_base_url + "/api/version")
        return {"status": "ok" if r.status_code == 200 else "degraded",
                "ollama": r.json().get("version", "?") if r.status_code == 200 else "DOWN"}
    except Exception as e:  # noqa: BLE001
        return {"status": "down", "error": str(e)[:120]}


@app.get("/api/models")
async def models():
    return await llm.ensure_models()


class SentencingBody(BaseModel):
    charge: str
    amount_yuan: float = 0
    victims: int = 0
    factors: list[str] = []


@app.post("/api/sentencing")
async def sentencing(body: SentencingBody):
    """Deterministic statutory sentencing suggestion (offline rule engine)."""
    from ..analysis.sentencing import advise, RULES
    audit("sentencing_query", actor="user", **body.model_dump())
    return {"supported_charges": list(RULES), "result": advise(
        body.charge, body.amount_yuan, body.victims, body.factors)}


class ModelSwitch(BaseModel):
    heavy: str | None = None
    light: str | None = None
    embed: str | None = None


@app.get("/api/flywheel/count")
async def flywheel_count():
    from ..agents.flywheel import load_all
    return {"count": len(load_all())}


@app.get("/api/models/installed")
async def models_installed():
    """List all models available on the local Ollama for switch dropdowns."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get(settings.ollama_base_url + "/api/tags")
        return {"models": [m["name"] for m in r.json().get("models", [])]}
    except Exception as e:  # noqa: BLE001
        return {"models": [], "error": str(e)[:120]}


@app.post("/api/models/set", dependencies=[Depends(check_token)])
async def models_set(body: ModelSwitch):
    """Hot-switch routing targets at runtime (persisted to lawfirm.local.env)."""
    changed = {}
    for field_, val in (("heavy_model", body.heavy), ("light_model", body.light),
                        ("embed_model", body.embed)):
        if val:
            setattr(settings, field_, val)
            env_key = "LAWFIRM_" + field_.upper()
            changed[env_key] = val
    # persist so restarts keep the choice
    envf = Path(__file__).resolve().parents[3] / "lawfirm.local.env"
    lines = {}
    if envf.exists():
        for ln in envf.read_text(encoding="utf-8").splitlines():
            if "=" in ln and not ln.strip().startswith("#"):
                k, v = ln.split("=", 1)
                lines[k.strip()] = v.strip()
    lines.update(changed)
    envf.write_text("\n".join(f"{k}={v}" for k, v in lines.items()) + "\n", encoding="utf-8")
    audit("model_switch", actor="user", **changed)
    Metrics.inc("model_switches")
    return {"ok": True, "active": {"heavy": settings.heavy_model,
            "light": settings.light_model, "embed": settings.embed_model}}


@app.get("/api/metrics")
async def metrics(window_min: int = 30, _=Depends(check_token)):
    """Live in-process counters + history derived from audit JSONL (cross-restart)."""
    from datetime import datetime, timedelta, timezone
    snap = Metrics.snapshot()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_min)
    llm_n, tool_n, avg_llm_s, series = 0, 0, None, {}
    durs = []
    files = sorted(Path(settings.audit_dir).glob("audit-*.jsonl"), reverse=True)[:2]
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception:  # noqa: BLE001
            continue
        for ln in reversed(lines):
            try:
                r = json.loads(ln)
                ts = datetime.fromisoformat(r["ts"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:  # noqa: BLE001
                continue
            if ts < cutoff:
                break
            ev = r.get("event")
            if ev == "llm_generate" and r.get("ok"):
                llm_n += 1
                if r.get("duration_s"):
                    durs.append(r["duration_s"])
                minute = ts.strftime("%H:%M")
                series[minute] = series.get(minute, 0) + 1
            elif ev == "agent_tool_call":
                tool_n += 1
    if durs:
        avg_llm_s = round(sum(durs) / len(durs), 2)
    snap["live"] = {"llm_calls_window": llm_n, "tool_calls_window": tool_n,
                    "avg_llm_seconds": avg_llm_s, "window_min": window_min,
                    "spark": [series[k] for k in sorted(series)][-30:]}
    return snap


@app.get("/api/audit/today")
async def audit_today(limit: int = 50, _=Depends(check_token)):
    from datetime import datetime
    p = Path(settings.audit_dir) / f'audit-{datetime.now().strftime('%Y%m%d')}.jsonl'
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(x) for x in lines]


@app.post("/api/cases", dependencies=[Depends(check_token)])
async def create_case(body: CaseCreate):
    c = new_case(body.title, body.charge, body.suspect)
    audit("case_created", case_id=c.case_id, title=c.title)
    return {"case_id": c.case_id}


@app.get("/api/cases")
async def get_cases():
    return [{"case_id": c.case_id, "title": c.title, "charge": c.charge,
             "suspect": c.suspect, "n_docs": len(c.docs), "updated_at": c.updated_at}
            for c in list_cases()]


@app.get("/api/cases/{case_id}")
async def get_case(case_id: str):
    return load_case(case_id).__dict__


@app.post("/api/cases/{case_id}/documents", dependencies=[Depends(check_token)])
async def upload(case_id: str, file: UploadFile = File(...)):
    case = load_case(case_id)
    name = Path(file.filename or "unnamed").name
    kind = detect_kind(name)
    if Path(name).suffix.lower() not in {".pdf", ".txt", ".md", ".xlsx", ".xls", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(400, f"不支持的文件类型: {name}")
    doc_id = uuid.uuid4().hex[:12]
    dest = _case_dir(case_id) / "documents" / f'{doc_id}{Path(name).suffix.lower()}'
    size, h = 0, hashlib.sha256()
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(413, f"超过大小限制 {settings.max_upload_mb}MB")
            h.update(chunk); f.write(chunk)
    dm = DocMeta(doc_id=doc_id, name=name, kind=kind, size=size, sha256=h.hexdigest())
    try:
        with traced("parse.document", event="doc_parsed", case_id=case_id):
            parse_document(case_id, dm)
    except Exception as e:  # noqa: BLE001
        dm.status, dm.error = "failed", str(e)[:200]
    add_doc(case, dm)
    n_chunks = await build_index(case_id)
    audit("doc_uploaded", case_id=case_id, doc=name, sha256=dm.sha256[:12], chunks=n_chunks)
    return {"doc_id": doc_id, "status": dm.status, "pages": dm.pages,
            "chunks": n_chunks, "error": dm.error}


@app.get("/api/cases/{case_id}/doc/{doc_id}")
async def doc_text(case_id: str, doc_id: str):
    """Parsed page-anchored text of one document (for the in-page viewer)."""
    from ..rag.store import _case_dir as cd
    tj = cd(case_id) / "documents" / f"{doc_id}.text.json"
    if not tj.exists():
        raise HTTPException(404, "parsed text not found")
    return json.loads(tj.read_text(encoding="utf-8"))


@app.get("/files/{case_id}/{doc_id}")
async def raw_file(case_id: str, doc_id: str, name: str = ""):
    """Serve original uploaded file (images preview inline). Local-only binding keeps this safe."""
    from pathlib import Path as _P
    from fastapi.responses import FileResponse
    base = (_case_dir(case_id) / "documents").resolve()
    cands = [f for f in base.glob(doc_id + "*") if f.suffix.lower() != ".json"]
    if not cands:
        raise HTTPException(404, "file not found")
    target = cands[0].resolve()
    if base not in target.parents:  # path traversal guard
        raise HTTPException(400, "bad path")
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".pdf": "application/pdf", ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
    return FileResponse(str(target), media_type=media.get(target.suffix.lower(), "application/octet-stream"),
                        filename=name or target.name)


@app.post("/api/cases/{case_id}/analyze", dependencies=[Depends(check_token)])
async def analyze(case_id: str, body: AnalyzeBody):
    load_case(case_id)
    if body.key:
        if body.key not in ANALYSIS_KEYS:
            raise HTTPException(400, f"key must be one of {ANALYSIS_KEYS}")
        return await run_analysis(case_id, body.key)
    return await run_all(case_id)


@app.get("/api/cases/{case_id}/analyses")
async def analyses(case_id: str):
    data = all_analyses(case_id)
    meta = {}
    for k, v in data.items():
        if isinstance(v, dict) and v.get("applicable") is False:
            meta[k] = {"applicable": False}
    return {"data": data, "meta": meta}


@app.get("/api/cases/{case_id}/chunk/{chunk_id}")
async def chunk_source(case_id: str, chunk_id: str):
    c = get_chunk(case_id, chunk_id)
    if not c:
        raise HTTPException(404, "chunk not found")
    return c


@app.get("/api/cases/{case_id}/search")
async def search_ep(case_id: str, q: str, k: int = 6):
    return await rag_search(case_id, q, k=k)


@app.post("/api/cases/{case_id}/ask", dependencies=[Depends(check_token)])
async def ask_ep(case_id: str, body: AskBody):
    return await agent_ask(body.question, case_id,
                           page_context=body.page_context, heavy=body.heavy)


@app.post("/api/cases/{case_id}/feedback", dependencies=[Depends(check_token)])
async def feedback(case_id: str, body: FeedbackBody):
    """Data flywheel: store ratings & corrections for few-shot retrieval later."""
    from ..agents.flywheel import record
    record(case_id, body.question, body.answer, body.rating, body.correction)
    audit("feedback_recorded", case_id=case_id, rating=body.rating)
    return {"saved": True}


_mount_gui(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.bind_host, port=8000)