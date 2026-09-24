"""MCP server (stdio): exposes case tools to any MCP-capable agent.

Run: lawfirm mcp   (or python -m lawfirm.mcp_server.server)
"""
import asyncio, json
from pathlib import Path


def _tools():
    from mcp.server.fastmcp import FastMCP
    from ..rag.store import list_cases, load_case, all_analyses
    from ..rag.retrieval import search as rag_search
    from ..analysis.reports import run_analysis, ANALYSIS_KEYS
    from ..agents.react import ask as agent_ask

    mcp = FastMCP("lawfirm-case-analysis")

    @mcp.tool()
    async def cases_list() -> str:
        '''List all legal cases (id/title/charge/docs count).'''
        return json.dumps([{"case_id": c.case_id, "title": c.title, "charge": c.charge,
                            "n_docs": len(c.docs)} for c in list_cases()], ensure_ascii=False)

    @mcp.tool()
    async def case_detail(case_id: str) -> str:
        '''Get full case metadata including documents.'''
        return json.dumps(load_case(case_id).__dict__, ensure_ascii=False)

    @mcp.tool()
    async def case_analyses(case_id: str) -> str:
        '''Return all generated analysis reports for a case.'''
        return json.dumps(all_analyses(case_id), ensure_ascii=False)

    @mcp.tool()
    async def analyze_key(case_id: str, key: str) -> str:
        '''Run one analysis: evidence|timeline|contradictions|irrelevant|summary|trial_strategy|bank_flow.'''
        if key not in ANALYSIS_KEYS:
            return json.dumps({"error": f"bad key; choose {ANALYSIS_KEYS}"})
        return json.dumps(await run_analysis(case_id, key), ensure_ascii=False)

    @mcp.tool()
    async def search_case_files(case_id: str, query: str, k: int = 5) -> str:
        '''Semantic search over the case file corpus; results carry doc/page anchors.'''
        res = await rag_search(case_id, query, k=k)
        return json.dumps([{kk: r[kk] for kk in ("doc_name", "page", "score", "chunk_id")} | {"snippet": r["text"][:300]} for r in res], ensure_ascii=False)

    @mcp.tool()
    async def ask_assistant(case_id: str, question: str, heavy: bool = False) -> str:
        '''Ask the ReAct assistant about a case; returns answer + step trace.'''
        out = await agent_ask(question, case_id, heavy=heavy)
        return json.dumps(out, ensure_ascii=False)

    return mcp


async def main():
    mcp = _tools()
    await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())