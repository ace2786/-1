"""Case-file storage: JSON-on-disk model, one dir per case.

data/cases/<case_id>/meta.json          案件元信息
data/cases/<case_id>/documents/<doc_id> 原始文件 + 解析出的 .text.json
data/cases/<case_id>/analysis/*.json    各分析产物（证据/时间线/矛盾/…）
data/cases/<case_id>/chunks.jsonl       RAG 分块（带页码锚点，供溯源跳转）
"""
import json, re, uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from ..config import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DocMeta:
    doc_id: str
    name: str
    kind: str            # pdf | text | xlsx | image
    size: int
    sha256: str
    uploaded_at: str = field(default_factory=_now)
    pages: int = 0
    status: str = "pending"   # pending|parsed|failed
    error: str | None = None


@dataclass
class Case:
    case_id: str
    title: str
    charge: str = ""
    suspect: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    docs: list = field(default_factory=list)     # list[DocMeta dict]
    notes: str = ""


def new_case(title: str, charge: str = "", suspect: str = "") -> Case:
    c = Case(case_id=uuid.uuid4().hex[:12], title=title, charge=charge, suspect=suspect)
    d = _case_dir(c.case_id)
    (d / "documents").mkdir(parents=True, exist_ok=True)
    (d / "analysis").mkdir(parents=True, exist_ok=True)
    save_case(c)
    return c


def _case_dir(case_id: str) -> Path:
    return Path(settings.cases_dir) / case_id


def save_case(case: Case):
    case.updated_at = _now()
    p = _case_dir(case.case_id)
    p.mkdir(parents=True, exist_ok=True)
    (p / "meta.json").write_text(json.dumps(asdict(case), ensure_ascii=False, indent=2), encoding="utf-8")


def load_case(case_id: str) -> Case:
    p = _case_dir(case_id) / "meta.json"
    if not p.exists():
        raise FileNotFoundError(f"case {case_id} not found")
    return Case(**json.loads(p.read_text(encoding="utf-8")))


def list_cases() -> list[Case]:
    out = []
    base = Path(settings.cases_dir)
    for d in sorted(base.iterdir()):
        if (d / "meta.json").exists():
            try:
                out.append(load_case(d.name))
            except Exception:  # noqa: BLE001
                pass
    return out


def add_doc(case: Case, doc: DocMeta):
    case.docs = [d for d in case.docs if d["doc_id"] != doc.doc_id]
    case.docs.append(asdict(doc))
    save_case(case)


def save_analysis(case_id: str, key: str, payload):
    """Persist an analysis artifact; key e.g. evidence/timeline/contradictions/..."""
    safe = re.sub(r"[^a-z0-9_]", "_", key.lower())
    p = _case_dir(case_id) / "analysis" / f"{safe}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"key": safe, "saved_at": _now(), "data": payload}, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_analysis(case_id: str, key: str):
    safe = re.sub(r"[^a-z0-9_]", "_", key.lower())
    p = _case_dir(case_id) / "analysis" / f"{safe}.json"
    return json.loads(p.read_text(encoding="utf-8"))["data"] if p.exists() else None


def all_analyses(case_id: str) -> dict:
    d = _case_dir(case_id) / "analysis"
    out = {}
    if d.exists():
        for f in d.glob("*.json"):
            try:
                out[f.stem] = json.loads(f.read_text(encoding="utf-8"))["data"]
            except Exception:  # noqa: BLE001
                pass
    return out


def save_chunks(case_id: str, chunks: list[dict]):
    """chunks: [{chunk_id, doc_id, doc_name, page, start_line, text}] -> jsonl"""
    p = _case_dir(case_id) / "chunks.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_chunks(case_id: str) -> list[dict]:
    p = _case_dir(case_id) / "chunks.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def doc_path(case_id: str, doc_id: str) -> Path:
    return _case_dir(case_id) / "documents" / doc_id