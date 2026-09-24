"""Typer CLI: lawfirm <cmd> — same core as GUI/MCP (thin-shell architecture)."""
import asyncio, json, sys
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="律所办案辅助系统 CLI（离线本地大模型）", no_args_is_help=True)
console = Console()


def _run(coro):
    return asyncio.run(coro)


@app.command()
def models():
    '''列出/检查本地模型状态。'''
    from ..llm import ensure_models
    st = _run(ensure_models())
    console.print(f'[green]已有[/green]: {', '.join(st['have']) or '无'}')
    if st["missing"]:
        console.print(f'[yellow]缺失[/yellow]: {', '.join(st['missing'])}  (ollama pull ...)')


@app.command()
def new_case(title: str, charge: str = "", suspect: str = ""):
    '''新建案件。'''
    from ..rag.store import new_case as nc
    c = nc(title, charge, suspect)
    console.print(f'案件已创建: [bold]{c.case_id}[/bold] {title}')


@app.command()
def cases():
    '''列出全部案件。'''
    from ..rag.store import list_cases
    t = Table("case_id", "案件", "罪名", "材料数")
    for c in list_cases():
        t.add_row(c.case_id, c.title, c.charge, str(len(c.docs)))
    console.print(t)


@app.command()
def add_doc(case_id: str, path: Path):
    '''添加卷宗文件（PDF/xlsx/txt/图片）。'''
    from ..rag.store import load_case, DocMeta, add_doc as ad, _case_dir, detect_kind
    from ..rag.parsing import parse_document
    from ..rag.retrieval import build_index
    import hashlib, uuid
    case = load_case(case_id)
    doc_id = uuid.uuid4().hex[:12]
    ext = path.suffix.lower()
    dest = _case_dir(case_id) / "documents" / f"{doc_id}{ext}"
    dest.write_bytes(path.read_bytes())
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    dm = DocMeta(doc_id=doc_id, name=path.name, kind=detect_kind(path.name),
                 size=path.stat().st_size, sha256=h)
    parse_document(case_id, dm)
    ad(case, dm)
    n = _run(build_index(case_id))
    console.print(f'解析完成: {path.name} pages={dm.pages} chunks={n}')


@app.command()
def analyze(case_id: str, key: str = typer.Option(None, help="evidence|timeline|contradictions|irrelevant|summary|trial_strategy|bank_flow；缺省=全部")):
    '''运行卷宗分析（深度模型）。'''
    async def _go():
        from ..analysis.reports import run_analysis, run_all
        if key:
            return {key: await run_analysis(case_id, key)}
        return await run_all(case_id)

    out = _run(_go())
    console.print_json(json.dumps(out, ensure_ascii=False))


@app.command()
def ask(question: str, case_id: str, heavy: bool = False):
    '''向AI助手提问（ReAct多步工具调用，带执行轨迹）。'''
    from ..agents.react import ask as agent_ask
    res = _run(agent_ask(question, case_id, heavy=heavy))
    console.print(Panel(res["answer"], title="AI助手"))
    console.print(f'[dim]执行步骤: {len(res['steps'])}[/dim]')
    for s in res["steps"]:
        icon = "🔧" if s["type"] == "tool" else "✅"
        detail = f'{s.get('action','')} {json.dumps(s.get('input',{}), ensure_ascii=False)}' if s["type"] == "tool" else ""
        console.print(f'  {icon} step{s['step']} {detail}')


@app.command()
def search(query: str, case_id: str, k: int = 5):
    '''卷宗语义检索（返回原文锚点）。'''
    from ..rag.retrieval import search as rs
    for r in _run(rs(case_id, query, k=k)):
        console.print(f'[cyan]{r['doc_name']}·P{r['page']}[/cyan] score={r['score']}')
        console.print(f'  {r['text'][:120]}...')


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    '''启动 Web GUI/API 服务。'''
    import uvicorn
    from ..api.server import app as fastapi_app
    uvicorn.run(fastapi_app, host=host, port=port)  # server module already mounts GUI at import


@app.command("mcp")
def mcp_stdio():
    '''以 MCP stdio server 模式运行（供智能体接入）。'''
    from ..mcp_server.server import main as mcp_main
    _run(mcp_main())


if __name__ == "__main__":
    app()