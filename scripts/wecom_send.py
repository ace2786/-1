#!/usr/bin/env python3
"""Send the 5-min talk + repo link to WeCom (企业微信) group robot webhook.

Usage:
    export WECOM_WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"
    python scripts/wecom_send.py <repo_url> [path/to/talk.md]

Robot setup: 企微群 → 设置 → 群机器人 → 添加 → 复制 webhook 地址。
Markdown limit is 4096 bytes; we auto-split into sequential messages.
"""
import json
import os
import sys
from pathlib import Path
import urllib.request

MD_LIMIT = 3800


def split_md(text: str, limit: int = MD_LIMIT):
    chunks, buf = [], ""
    for line in text.splitlines(keepends=True):
        if len(buf.encode("utf-8")) + len(line.encode("utf-8")) > limit:
            chunks.append(buf)
            buf = line
        else:
            buf += line
    if buf:
        chunks.append(buf)
    return chunks


def send(webhook: str, md: str) -> bool:
    body = json.dumps({"msgtype": "markdown", "markdown": {"content": md}}).encode("utf-8")
    req = urllib.request.Request(webhook, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            resp = json.loads(r.read().decode())
        ok = resp.get("errcode") == 0
        if not ok:
            print("wecom error:", resp)
        return ok
    except Exception as e:  # noqa: BLE001
        print("send failed:", e)
        return False


def main():
    webhook = os.environ.get("WECOM_WEBHOOK", "")
    if not webhook:
        print("请先设置环境变量 WECOM_WEBHOOK（群机器人 webhook 地址）")
        sys.exit(1)
    repo = sys.argv[1] if len(sys.argv) > 1 else "(仓库地址待填)"
    talk_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent.parent / "docs" / "TALK.md"
    talk = talk_path.read_text(encoding="utf-8")
    head = f"**📦 项目仓库**：{repo}\n\n---\n\n"
    all_md = head + talk
    chunks = split_md(all_md)
    for i, c in enumerate(chunks, 1):
        prefix = f"（{i}/{len(chunks)}）\n\n" if len(chunks) > 1 else ""
        if not send(webhook, prefix + c):
            sys.exit(2)
        print(f"message {i}/{len(chunks)} sent ✓")
    print("全部发送完成 ✅")


if __name__ == "__main__":
    main()
