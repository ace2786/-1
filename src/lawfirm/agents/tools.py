"""Agent tool registry: retrieval, doc reading, analysis results, bank-flow stats.

Each tool is offline-safe and audited upstream in the ReAct loop.
"""
import json
from ..rag.retrieval import search as rag_search
from ..rag.store import load_analysis, load_chunks, _case_dir
from ..observe import log


TOOL_SPECS = [
    {"name": "search_case", "desc": "在本案全部卷宗中语义检索，返回最相关原文片段(含文档/页码)",
     "params": {"query": "str 检索语句"}},
    {"name": "read_doc_page", "desc": "读取指定文档指定页的原文内容",
     "params": {"doc_name": "str", "page": "int"}},
    {"name": "get_analysis", "desc": "获取已生成的分析结果。key可选: evidence/timeline/contradictions/irrelevant/summary/trial_strategy/bank_flow",
     "params": {"key": "str"}},
    {"name": "list_docs", "desc": "列出本案所有卷宗材料及页数", "params": {}},
    {"name": "bank_stats", "desc": "对银行流水表格做统计计算(收支合计/极值/高频对手方)，输入sheet名可空",
     "params": {"column_hint": "str 金额列名提示，可空"}},
]


async def execute(name: str, args: dict, case_id: str) -> str:
    try:
        if name == "search_case":
            res = await rag_search(case_id, args.get("query", ""), k=5)
            return json.dumps([{"doc": r["doc_name"], "page": r["page"],
                                "score": r["score"], "text": r["text"][:400]} for r in res],
                              ensure_ascii=False)
        if name == "read_doc_page":
            want_doc, want_pg = args.get("doc_name", ""), int(args.get("page", 1))
            for c in load_chunks(case_id):
                if c["doc_name"] == want_doc and c["page"] == want_pg:
                    return c["text"][:3000]
            # fuzzy: substring match
            for c in load_chunks(case_id):
                if want_doc in c["doc_name"] and c["page"] == want_pg:
                    return c["text"][:3000]
            return f"未找到 {want_doc} 第{want_pg}页"
        if name == "get_analysis":
            data = load_analysis(case_id, args.get("key", ""))
            return json.dumps(data, ensure_ascii=False)[:6000] if data else "该分析尚未生成"
        if name == "list_docs":
            case = __import__("lawfirm.rag.store", fromlist=["load_case"]).load_case(case_id)
            return json.dumps([{"name": d["name"], "kind": d["kind"], "pages": d["pages"],
                                "status": d["status"]} for d in case.docs], ensure_ascii=False)
        if name == "bank_stats":
            return _bank_stats(case_id, args.get("column_hint", ""))
        return f"未知工具: {name}"
    except Exception as e:  # noqa: BLE001
        log.warning("tool_fail", tool=name, error=str(e)[:200])
        return f"工具执行失败: {e}"


def _bank_stats(case_id: str, col_hint: str) -> str:
    """Deterministic pandas stats over parsed xlsx text (offline, no LLM math)."""
    import pandas as pd, io, re
    from ..rag.store import doc_path
    base = _case_dir(case_id) / "documents"
    for tj in base.glob("*.text.json"):
        data = json.loads(tj.read_text(encoding="utf-8"))
        if not data["pages"]:
            continue
        head = data["pages"][0]["text"]
        if "\t" not in head:
            continue
        rows = [r.split("\t") for r in head.splitlines() if r.strip()]
        if len(rows) < 2:
            continue
        df = pd.DataFrame(rows[1:], columns=rows[0])
        amt_col = None
        for c in df.columns:
            if col_hint and col_hint in c:
                amt_col = c; break
        if not amt_col:
            for c in df.columns:
                if any(k in c for k in ("金额", "发生额", "amount", "收入", "支出")):
                    amt_col = c; break
        if not amt_col:
            continue
        nums = pd.to_numeric(df[amt_col].astype(str).str.replace(",", ""), errors="coerce").dropna()
        if nums.empty:
            continue
        out = {"doc": data["name"], "amount_column": amt_col, "n_rows": int(len(nums)),
               "sum": float(nums.sum()), "max": float(nums.max()), "min": float(nums.min())}
        # counterparties frequency if a column looks like names
        for c in df.columns:
            if any(k in c for k in ("对方", "户名", "counterparty")):
                vc = df[c].value_counts().head(8)
                out["top_counterparties"] = {str(k): int(v) for k, v in vc.items()}
                break
        return json.dumps(out, ensure_ascii=False)
    return "本案没有可统计的银行流水表格文件"