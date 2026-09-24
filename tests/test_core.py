"""Pytest suite: offline-safe tests (no LLM calls). Run: .venv/bin/python -m pytest tests/ -q"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest  # noqa: E402
from lawfirm.rag.store import new_case, load_case, list_cases, DocMeta, add_doc, save_analysis, load_analysis, all_analyses  # noqa: E402
from lawfirm.rag.parsing import detect_kind, parse_text, sha256_file  # noqa: E402
from lawfirm.analysis.reports import ANALYSIS_KEYS  # noqa: E402


def test_case_lifecycle(tmp_path, monkeypatch):
    from lawfirm.config import settings
    monkeypatch.setattr(settings, "cases_dir", tmp_path)
    c = new_case("测试案", "诈骗罪", "李四")
    assert c.case_id and len(c.title) > 0
    loaded = load_case(c.case_id)
    assert loaded.title == "测试案"
    assert any(x.case_id == c.case_id for x in list_cases())


def test_detect_kind():
    assert detect_kind("a.pdf") == "pdf"
    assert detect_kind("流水.XLSX") == "xlsx"
    assert detect_kind("x.exe") == "unknown"


def test_parse_text_pages(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("行1\n行2\n行3\n", encoding="utf-8")
    pages, n = parse_text(p)
    assert n >= 1 and "行1" in pages[0]["text"]


def test_sha256(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    h = sha256_file(p)
    assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_analysis_roundtrip(tmp_path, monkeypatch):
    from lawfirm.config import settings
    monkeypatch.setattr(settings, "cases_dir", tmp_path)
    c = new_case("分析存储案")
    save_analysis(c.case_id, "evidence", {"items": [{"title": "t"}]})
    assert load_analysis(c.case_id, "evidence")["items"][0]["title"] == "t"
    assert "evidence" in all_analyses(c.case_id)


def test_analysis_keys_complete():
    assert set(ANALYSIS_KEYS) == {"evidence", "timeline", "contradictions", "irrelevant",
                                  "summary", "trial_strategy", "bank_flow"}


def test_audit_written(tmp_path, monkeypatch):
    from lawfirm.config import settings
    from lawfirm.observe import audit
    monkeypatch.setattr(settings, "audit_dir", tmp_path)
    audit("unit_event", case_id="x", ok=True)
    files = list(tmp_path.glob("audit-*.jsonl"))
    assert files
    rec = json.loads(files[0].read_text(encoding="utf-8").splitlines()[-1])
    assert rec["event"] == "unit_event"
