"""FastAPI app: REST API + health/metrics/audit observability. Local-first security."""
import hashlib, json, uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
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


@app.get("/api/metrics")
async def metrics(_=Depends(check_token)):
    return Metrics.snapshot()


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
    return all_analyses(case_id)


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
    from ..rag.store import _case_dir as cd
    fp = cd(case_id) / "flywheel.jsonl"
    rec = body.model_dump(); rec["ts"] = __import__("datetime").datetime.now().isoformat()
    with open(fp, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    audit("feedback_recorded", case_id=case_id, rating=body.rating)
    return {"saved": True}


_mount_gui(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.bind_host, port=8000)