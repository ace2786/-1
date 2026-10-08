"""Assemble case dossier analyses into a lawyer-ready review report."""
import json
from datetime import datetime
from ..config import settings
from .reports import ANALYSIS_KEYS
from .sentencing import advise
from ..rag.store import load_case, all_analyses, _case_dir

_LABELS = {
    "summary": "一、卷宗摘要", "evidence": "二、证据要点", "timeline": "三、案件时间线",
    "contradictions": "四、矛盾排查", "irrelevant": "五、无效材料过滤",
    "trial_strategy": "六、庭审思路", "bank_flow": "七、银行流水专项分析",
}


def _cites(item) -> str:
    cs = item.get("citations") or []
    return "；".join(f"{c.get('doc_name','')}·P{c.get('page','')}" for c in cs)


def build_markdown(case_id: str) -> str:
    case = load_case(case_id)
    data = all_analyses(case_id)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [f"# 案件审查报告工作底稿", "",
         f"| | |", "|---|---|",
         f"| 案件名称 | {case.title} |",
         f"| 涉嫌罪名 | {case.charge or '—'} |",
         f"| 犯罪嫌疑人 | {case.suspect or '—'} |",
         f"| 卷宗材料 | {len(case.docs)} 份（{'、'.join(d['name'] for d in case.docs)}） |",
         f"| 生成时间 | {now} |",
         f"| 生成方式 | 全离线本地大模型（律所办案辅助系统 v0.1） |", "",
         "> ⚠️ 本报告由 AI 辅助生成，所有结论均附卷宗来源，仅供办案参考，须由执业律师复核。", ""]

    def _flat(key):
        """Normalize report payload: some models nest fields under items[0]."""
        v = data.get(key)
        if not isinstance(v, dict):
            return None
        if not any(x for k, x in v.items() if k != "items") and isinstance(v.get("items"), list) \
                and v["items"] and isinstance(v["items"][0], dict):
            merged = dict(v["items"][0])
            merged.update({k: x for k, x in v.items() if k != "items"})
            return merged
        return v

    s = _flat("summary")
    if s:
        L += [_LABELS["summary"], "", f"**当事人**：{s.get('parties','')}", "", s.get("overview",""), ""]
        if s.get("amount"):
            L += [f"**涉案金额**：{s['amount']}", ""]
        for p in s.get("key_points", []):
            L.append(f"- {p}")
        L.append("")

    ev = data.get("evidence", {}).get("items", [])
    if ev:
        L += [_LABELS["evidence"], "", "| 类别 | 要点 | 说明 | 来源 |", "|---|---|---|---|"]
        L += [f"| {i.get('type','')} | {i.get('title','')} | {i.get('detail','')[:80]} | {_cites(i)} |" for i in ev]
        L.append("")

    tl = data.get("timeline", {}).get("items", [])
    if tl:
        L += [_LABELS["timeline"], "", "| 时间 | 事件 | 意义 | 来源 |", "|---|---|---|---|"]
        L += [f"| {i.get('date','')} | {i.get('event','')} | {i.get('significance','')} | {_cites(i)} |" for i in tl]
        L.append("")

    ct = data.get("contradictions", {}).get("items", [])
    L += [_LABELS["contradictions"], ""]
    if ct:
        for n, i in enumerate(ct, 1):
            a, b = i.get("claim_a", {}), i.get("claim_b", {})
            L += [f"**矛盾{n+1}【{i.get('risk','')}】{i.get('topic','')}**", "",
                  f"- A方：「{a.get('text','')}」（{a.get('source_doc','')}·P{a.get('page','')}）",
                  f"- B方：「{b.get('text','')}」（{b.get('source_doc','')}·P{b.get('page','')}）",
                  f"- 分析：{i.get('analysis','')}", ""]
    else:
        L += ["未发现明显事实矛盾。", ""]

    ir = data.get("irrelevant", {}).get("items", [])
    if ir:
        L += [_LABELS["irrelevant"], ""] + [f"- **{i.get('doc_name','')}**：{i.get('reason','')}" for i in ir] + [""]

    ts = _flat("trial_strategy")
    if ts:
        names = {"focuses": "争议焦点", "cross_exam": "质证要点", "defense_lines": "辩护方向", "risks": "风险提示"}
        L += [_LABELS["trial_strategy"], ""]
        for k, cn in names.items():
            if ts.get(k):
                L += [f"**{cn}**"] + [f"- {x}" for x in ts[k]] + [""]

    bf = _flat("bank_flow")
    if bf and bf.get("applicable") is not False:
        L += [_LABELS["bank_flow"], "",
              f"期间：{bf.get('period','')}　流入：{bf.get('total_in','')}　流出：{bf.get('total_out','')}", ""]
        for x in bf.get("anomalies", []):
            L.append(f"- **{x.get('desc','')}**：{x.get('evidence','')}")
        if bf.get("conclusion"):
            L += ["", f"结论：{bf['conclusion']}"]
        L.append("")

    # sentencing appendix (rule engine, deterministic)
    amt = 0
    try:
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*万", (data.get("summary") or {}).get("amount", ""))
        if m:
            amt = int(float(m.group(1)) * 10000)
    except Exception:  # noqa: BLE001
        pass
    sent = advise(case.charge, amount_yuan=amt, victims=len(case.docs)) if case.charge else None
    if sent and not sent.get("error"):
        lo, hi = sent["adjusted_range_months"]
        L += ["## 附录：量刑区间参考（确定性法条引擎）", "",
              f"依据《{sent['article']}》{sent['tier']}：{sent['tier_note']}；"
              f"考虑已识别情节后建议区间约 **{lo:.0f} ~ {hi:.0f} 个月**。",
              "", *[f"> {c}" for c in sent["caveats"]], ""]

    L += ["---", "", "*本报告由律所办案辅助系统离线生成；每次模型调用与工具执行均可在审计日志中回溯。*"]
    return "\n".join(L)


def export_file(case_id: str, fmt: str = "md", out_path: str | None = None) -> str:
    md = build_markdown(case_id)
    if out_path is None:
        out_path = str(_case_dir(case_id) / f"review_report.{fmt}")
    Path_p = __import__("pathlib").Path
    Path_p(out_path).write_text(md, encoding="utf-8")
    return out_path
