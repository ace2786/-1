#!/usr/bin/env python3
"""Seed the built-in demo case: 张某某涉嫌非法吸收公众存款案 (3 docs)."""
import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lawfirm.rag.store import new_case, load_case, DocMeta, add_doc, _case_dir  # noqa: E402
from lawfirm.rag.parsing import parse_document, detect_kind  # noqa: E402
from lawfirm.rag.retrieval import build_index  # noqa: E402
import hashlib  # noqa: E402
import uuid  # noqa: E402

DOCS = {
    "doc1_起诉意见书.txt": "起诉意见书\n犯罪嫌疑人张某某，于2021年6月起以投资养老项目为名向不特定公众吸收资金。\n被害人李某陈述：2021年6月与张某某签订合同并转账50万元。\n询问笔录：张某某供述其2021年6月开始吸收存款。\n",
    "doc2_司法会计鉴定.txt": "司法会计鉴定意见书\n经鉴定，2021年3月15日受害人王某向张某个人账户转入资金80万元，属吸收存款性质。\n截至2022年12月，涉案账户共流入资金4200万元。\n",
    "doc3_物业催缴单.txt": "物业催缴单\n业主张三：您名下房屋2023年度物业费3200元尚未缴纳，请于月底前缴清。\n",
}


def main():
    case = new_case("张某某涉嫌非法吸收公众存款案", "非法吸收公众存款罪", "张某某")
    for name, content in DOCS.items():
        doc_id = uuid.uuid4().hex[:12]
        path = _case_dir(case.case_id) / "documents" / f"{doc_id}.txt"
        path.write_text(content, encoding="utf-8")
        dm = DocMeta(doc_id=doc_id, name=name, kind=detect_kind(name),
                     size=path.stat().st_size,
                     sha256=hashlib.sha256(content.encode()).hexdigest())
        parse_document(case.case_id, dm)
        add_doc(case, dm)
    n = asyncio.run(build_index(case.case_id))
    print(json.dumps({"case_id": case.case_id, "chunks": n}, ensure_ascii=False))


if __name__ == "__main__":
    main()
