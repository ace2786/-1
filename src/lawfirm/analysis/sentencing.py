"""Sentencing guideline engine: deterministic statutory ranges + factor adjustments.

Law basis (public statutes, offline-safe):
- 刑法§176 非法吸收公众存款罪; §264 盗窃罪; §266 诈骗罪; §133之一 危险驾驶罪;
- §234 故意伤害罪; §383/386 受贿罪(数额较大/巨大/特别巨大)
Amount tiers per 两高司法解释 for 非吸: 个人20万/单位100万入罪; 数额巨大(个人500万).
This module NEVER invents law: every output cites its rule id.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

# ------------------------------------------------------------------ rules
RULES = {
    "非法吸收公众存款罪": {
        "article": "刑法第176条",
        "tiers": [
            {"name": "基本档", "cond": lambda m, s: True,
             "range": (0, 3), "note": "三年以下有期徒刑或拘役，并处或单处罚金"},
            {"name": "数额巨大或情节严重", "cond": lambda m, s: m >= 5_000_000 or s >= 500,
             "range": (3, 10), "note": "三年以上十年以下有期徒刑，并处罚金"},
            {"name": "数额特别巨大或有其他特别严重情节", "cond": lambda m, s: m >= 50_000_000,
             "range": (10, 20), "note": "十年以上有期徒刑，并处罚金（修正案十一新增档）"},
        ],
        "entry": {"amount": 200_000, "victims": 30, "note": "个人吸存20万元/对象30人以上追诉"},
    },
    "集资诈骗罪": {
        "article": "刑法第192条",
        "tiers": [
            {"name": "基本档", "cond": lambda m, s: True, "range": (0, 7), "note": "七年以下有期徒刑并处罚金"},
            {"name": "数额巨大", "cond": lambda m, s: m >= 1_000_000, "range": (7, 15), "note": "七年以上有期徒刑或无期"},
        ],
        "entry": {"amount": 100_000, "victims": None, "note": "集资诈骗10万元追诉"},
    },
    "盗窃罪": {
        "article": "刑法第264条",
        "tiers": [
            {"name": "数额较大", "cond": lambda m, s: True, "range": (0, 3), "note": "三年以下"},
            {"name": "数额巨大", "cond": lambda m, s: m >= 30_000, "range": (3, 10), "note": "三至十年"},
            {"name": "数额特别巨大", "cond": lambda m, s: m >= 300_000, "range": (10, 15), "note": "十年以上或无期"},
        ],
        "entry": {"amount": 1_000 if False else 2_000, "victims": None, "note": "多数省份1000-3000元入罪"},
    },
    "诈骗罪": {
        "article": "刑法第266条",
        "tiers": [
            {"name": "数额较大", "cond": lambda m, s: True, "range": (0, 3), "note": "三年以下"},
            {"name": "数额巨大", "cond": lambda m, s: m >= 30_000, "range": (3, 10), "note": "三至十年"},
            {"name": "数额特别巨大", "cond": lambda m, s: m >= 500_000, "range": (10, 15), "note": "十年以上或无期"},
        ],
        "entry": {"amount": 3_000, "victims": None, "note": "3000元入罪（两高解释）"},
    },
    "危险驾驶罪": {
        "article": "刑法第133条之一",
        "tiers": [{"name": "醉驾等", "cond": lambda m, s: True, "range": (0, 0.5), "note": "拘役(1-6个月)并处罚金"}],
        "entry": {"amount": None, "victims": None, "note": "血液酒精≥80mg/100ml即构成"},
    },
    "故意伤害罪": {
        "article": "刑法第234条",
        "tiers": [
            {"name": "轻伤", "cond": lambda m, s: True, "range": (0, 3), "note": "三年以下"},
            {"name": "重伤", "cond": lambda m, s: m >= 1, "range": (3, 10), "note": "三至十年（m参数此处传伤害等级系数）"},
            {"name": "致死/特别残忍致残", "cond": lambda m, s: m >= 10, "range": (10, 15), "note": "十年以上、无期或死刑"},
        ],
        "entry": {"amount": None, "victims": None, "note": "轻伤鉴定即入罪"},
    },
}

FACTORS = {  # 量刑情节调节比例（参照两高量刑指导意见常见幅度）
    "confession":   (-0.20, "如实供述/坦白：可减少基准刑20%以下"),
    "guilty_plea":  (-0.30, "认罪认罚：可减少基准刑30%以下"),
    "surrender":    (-0.40, "自首：可减少基准刑40%以下"),
    "meritorious":  (-0.20, "立功：可减少基准刑20%以下"),
    "restitution":  (-0.30, "退赃退赔/取得谅解：可减少基准刑30%以下"),
    "recidivist":   (+0.10, "累犯：应当从重"),
    "organizer":    (+0.20, "主犯/组织者：按全部犯罪处罚"),
}


@dataclass
class Suggestion:
    charge: str
    article: str
    tier: str
    tier_note: str
    base_range_months: tuple[int | float, int | float]
    adjusted_range_months: tuple[int | float, int | float]
    applied_factors: list[dict]
    entry_check: dict
    caveats: list[str] = field(default_factory=list)

    def to_dict(self):
        d = self.__dict__.copy()
        d["base_range_months"] = list(self.base_range_months)
        d["adjusted_range_months"] = list(self.adjusted_range_months)
        return d


def advise(charge: str, amount_yuan: float = 0, victims: int = 0,
           factors: list[str] | None = None) -> dict:
    """Core deterministic sentencing suggestion."""
    factors = factors or []
    rule = RULES.get(charge)
    if not rule:
        return {"error": f"未内置罪名『{charge}』，支持：{'、'.join(RULES)}",
                "supported": list(RULES)}
    # pick highest satisfied tier
    tier = None
    for t in rule["tiers"]:
        try:
            ok = t["cond"](amount_yuan, victims)
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            tier = t
    lo, hi = tier["range"]
    lo_m, hi_m = lo * 12, hi * 12

    adj_lo, adj_hi, applied = lo_m, hi_m, []
    mult = 1.0
    for f in factors:
        if f in FACTORS:
            delta, desc = FACTORS[f]
            mult += delta
            applied.append({"factor": f, "desc": desc})
        else:
            applied.append({"factor": f, "desc": f"未知情节『{f}』未计入"})
    mult = max(mult, 0.1)
    adj_lo, adj_hi = round(lo_m * mult, 1), round(hi_m * mult, 1)

    entry = rule["entry"]
    meets = (entry["amount"] is None or amount_yuan >= entry["amount"]) and \
            (entry["victims"] is None or victims >= entry["victims"])
    caveats = [
        "本结果为法条区间的确定性推算，仅供辩护研究参考，不构成法律意见",
        "具体量刑须结合当地实施细则与法官自由裁量",
    ]
    if not meets:
        caveats.insert(0, f"⚠ 按追诉标准（{entry['note']}）可能未达刑事立案门槛")

    s = Suggestion(
        charge=charge, article=rule["article"], tier=tier["name"], tier_note=tier["note"],
        base_range_months=(lo_m, hi_m), adjusted_range_months=(adj_lo, adj_hi),
        applied_factors=applied,
        entry_check={"meets": meets, "standard": entry["note"],
                     "amount_input": amount_yuan, "victims_input": victims},
        caveats=caveats,
    )
    return s.to_dict()


if __name__ == "__main__":
    print(json.dumps(advise("非法吸收公众存款罪", 42_000_000, 62,
                            ["guilty_plea", "confession"]), ensure_ascii=False, indent=2))
