"""Structured case-file analysis: 7 report types via heavy model, JSON contracts.

Every item carries citations [{doc_name, page, chunk_id}] so the GUI can render
clickable source jumps (溯源跳转).
"""
import json
from ..llm import generate_json, ensure_models
from ..observe import traced, audit
from ..rag.store import load_chunks, save_analysis, load_case
from ..config import settings

ANALYSIS_KEYS = ["evidence", "timeline", "contradictions", "irrelevant",
                 "summary", "trial_strategy", "bank_flow"]

_PROMPTS = {
    "evidence": "你是刑事律师助理。从卷宗中提炼证据要点，区分书证/物证/言词证据。",
    "timeline": "你是刑事案件时间线分析师。梳理案件完整事件时序。",
    "contradictions": "你是卷宗质证专家。找出卷宗内相互冲突的事实陈述（如供述时间与司法鉴定不一致）。",
    "irrelevant": "你是材料筛查员。判断哪些文档与本案无关并给出排除理由。",
    "summary": "你是资深刑辩律师。生成全案卷宗摘要（当事人、案情梗概、涉案金额、关键节点）。",
    "trial_strategy": "你是庭审顾问。基于卷宗给出庭审思路：争议焦点、质证要点、辩护方向。",
    "bank_flow": "你是司法会计。分析银行流水：资金规模、频率异常、可疑对手方、吸存起止时间推断。",
}

_SCHEMA = {
    "evidence": '{"items":[{"type":"书证|物证|言词证据","title":"","detail":"","citations":[{"doc_name":"","page":0,"quote":"原文摘录"}]}]}',
    "timeline": '{"items":[{"date":"","event":"","significance":"","citations":[{"doc_name":"","page":0,"quote":""}]}]}',
    "contradictions": '{"items":[{"topic":"","claim_a":{"text":"","source_doc":"","page":0},"claim_b":{"text":"","source_doc":"","page":0},"risk":"高危|中风险","analysis":""}]}',
    "irrelevant": '{"items":[{"doc_name":"","reason":""}],"kept_docs":[]}',
    "summary": '{"parties":"","overview":"","amount":"","key_points":[],"citations":[{"doc_name":"","page":0,"quote":""}]}',
    "trial_strategy": '{"focuses":[""],"cross_exam":[""],"defense_lines":[""],"risks":[""]}',
    "bank_flow": '{"period":"","total_in":"","total_out":"","anomalies":[{"desc":"","evidence":""}],"counterparties":[{"name":"","note":""}],"conclusion":""}',
}



async def _pick_model(heavy: bool) -> str | None:
    """Route to heavy model; fall back to light automatically if not pulled yet."""
    from ..llm import ensure_models
    st = await ensure_models()
    want = [key for key in ("heavy_model", "light_model")]
    if heavy and not any(m == __import__("lawfirm.config", fromlist=["settings"]).settings.heavy_model or
                         m.split(":")[0] == __import__("lawfirm.config", fromlist=["settings"]).settings.heavy_model.split(":")[0]
                         for m in st["have"]):
        return None  # light via generate default
    return None


def _material(case_id: str, max_chars: int = 12000) -> tuple[str, list[dict]]:
    """Concatenate deduped chunks (by doc_id+page) up to char budget."""
    chunks = load_chunks(case_id)
    buf, used, seen = [], [], set()
    total = 0
    for c in chunks:
        key = (c["doc_id"], c["page"])
        if key in seen:
            continue
        seen.add(key)
        seg = "【" + str(c["doc_name"]) + "·第" + str(c["page"]) + "页】" + str(c["text"])
        if total + len(seg) > max_chars and used:
            break
        buf.append(seg); used.append(c); total += len(seg)
    return "\n\n".join(buf), used


def _fix_claims(payload, used_chunks):
    """Attach chunk anchors to contradictions' claim_a/claim_b too."""
    def find_chunk(doc_hint, text):
        t = (text or "").strip()[:24]
        for c in used_chunks:
            if t and t in c["text"] and (not doc_hint or doc_hint in c["doc_name"]):
                return c
        for c in used_chunks:
            if doc_hint and doc_hint in c["doc_name"]:
                return c
        return None
    if isinstance(payload, dict):
        for it in payload.get("items", []) if isinstance(payload.get("items"), list) else []:
            for side in ("claim_a", "claim_b"):
                cl = it.get(side)
                if isinstance(cl, dict):
                    c = find_chunk(cl.get("source_doc", ""), cl.get("text", ""))
                    if c:
                        cl["chunk_id"] = c["chunk_id"]
                        cl["page"] = c["page"]
                        cl["source_doc"] = c["doc_name"]


def _fix_citations(payload, used_chunks):
    """Attach nearest chunk_id to each citation by quote matching (traceability glue)."""
    def walk(o):
        if isinstance(o, dict):
            if "citations" in o and isinstance(o["citations"], list):
                for cit in o["citations"]:
                    q = (cit.get("quote") or "").strip()[:30]
                    for c in used_chunks:
                        if q and q in c["text"]:
                            cit.setdefault("chunk_id", c["chunk_id"]); break
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(payload)
    return payload


async def run_analysis(case_id: str, key: str, *, heavy: bool = True) -> dict:
    assert key in ANALYSIS_KEYS
    case = load_case(case_id)
    material, used = _material(case_id)
    if not material.strip():
        return {"items": [], "error": "该案卷尚无已解析文本，请先上传并解析材料"}
    hint = ""
    if key in ("summary", "trial_strategy", "bank_flow"):
        hint = "\n注意：顶层JSON对象的每个字段都必须填写（没有信息就填空字符串或空数组），items不是这些键的返回结构。"
    prompt = (f"{_PROMPTS[key]}\n罪名参考：{case.charge or '未知'}\n\n"
              f"严格输出JSON，形如 {_SCHEMA[key]} 。只依据给定卷宗内容，禁止编造；"
              f"每条结论必须附citations，quote字段填卷宗原文片段。{hint}\n\n===卷宗材料===\n{material}")
    with traced(f"analysis.{key}", event="analysis_run", case_id=case_id):
        from ..llm import ensure_models
        st = await ensure_models()
        _hb = settings.heavy_model.split(":")[0]
        have_heavy = any(x.split(":")[0] == _hb and x != settings.light_model for x in st["have"])
        data = await generate_json(prompt, heavy=(heavy and have_heavy), temperature=0.1)
    _fix_citations(data, used)
    _fix_claims(data, used)
    save_analysis(case_id, key, data)
    return data


async def run_all(case_id: str) -> dict:
    out = {}
    for key in ANALYSIS_KEYS:
        try:
            out[key] = await run_analysis(case_id, key)
        except Exception as e:  # noqa: BLE001
            out[key] = {"items": [], "error": str(e)[:200]}
            audit("analysis_fail", case_id=case_id, key=key, error=str(e)[:300])
    return out