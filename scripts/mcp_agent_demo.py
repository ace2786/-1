"""Demo: an external agent connects to `lawfirm mcp` via stdio and calls tools.

This is exactly what 'MCP版本以供智能体调用' means — any MCP-capable client
(Claude Desktop, Cursor, or this script acting as the agent) can discover and
invoke our legal-analysis toolbox over the standard protocol.
"""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # 1) spawn our own MCP server as a subprocess (stdio transport)
    params = StdioServerParameters(command=".venv/bin/lawfirm", args=["mcp"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()

            # 2) agent discovers available tools
            tools = await sess.list_tools()
            print("== 智能体发现工具箱 ==")
            for t in tools.tools:
                print(f"  🔧 {t.name}: {t.description[:46]}")

            # 3) agent autonomously calls tools to answer a user question
            print("\n== 智能体自主调用 ==")
            r = await sess.call_tool("cases_list", {})
            print("call cases_list ->", r.content[0].text[:80], "...")

            import json
            cases = json.loads(r.content[0].text)
            fraud = next(c for c in cases if "诈骗" in c["title"])
            print(f"\n选定案件: {fraud['title']} ({fraud['case_id']})")

            r2 = await sess.call_tool("search_case_files",
                                      {"case_id": fraud["case_id"], "query": "预付款去向", "k": 2})
            hits = json.loads(r2.content[0].text)
            print("\n== search_case_files('预付款去向') 返回 ==")
            for h in hits:
                print(f"  📄 {h['doc_name']}·P{h['page']} score={h['score']}")
                print(f"     {h['snippet'][:70]}...")

            r3 = await sess.call_tool("sentencing_advise",
                                      {"charge": "诈骗罪", "amount_yuan": 850000, "factors": ""})
            sent = json.loads(r3.content[0].text)
            print(f"\n== sentencing_advise(诈骗85万) ==\n  档位: {sent['tier']} | 基准刑(月): {sent['base_range_months']}")

            print("\n✅ 结论：外部智能体可通过 MCP 协议完整调用本系统全部能力。")


asyncio.run(main())
