"""Server-rendered GUI (Jinja2) mounted onto the API app. Zero-build frontend."""
from pathlib import Path
from fastapi import Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from ..rag.store import list_cases, load_case, all_analyses
from ..config import settings

TEMPLATES = Path(__file__).parent / "templates"
templates = TemplateResponse = None  # set by mount()


def mount(app):
    global templates
    templates = Jinja2Templates(directory=str(TEMPLATES))

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse(request, "home.html", {"cases": list_cases()})

    @app.get("/case/{case_id}", response_class=HTMLResponse)
    async def case_page(request: Request, case_id: str):
        case = load_case(case_id)
        return templates.TemplateResponse(request, "case.html", {"case": case.__dict__,
            "analyses": all_analyses(case_id),
            "analysis_keys": ["evidence", "timeline", "contradictions", "irrelevant",
                              "summary", "trial_strategy", "bank_flow"],
            "models": {"heavy": settings.heavy_model, "light": settings.light_model},
        })

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        from ..llm import ensure_models
        st = await ensure_models()
        return templates.TemplateResponse(request, "settings.html", {
            "s": settings, "st": st,
            "heavy": settings.heavy_model, "light": settings.light_model,
            "embed": settings.embed_model,
        })