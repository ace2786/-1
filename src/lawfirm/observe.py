"""Structured logging + metrics + audit trail (observability layer)."""
import json, time, threading
from pathlib import Path
from datetime import datetime, timezone
import structlog
from .config import settings

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(ensure_ascii=False),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        structlog.stdlib.get_level_from_number(settings.log_level) if hasattr(structlog.stdlib,"get_level_from_number") else 20
    ),
)

log = structlog.get_logger("lawfirm")


class Metrics:
    """Thread-safe in-process counters; exposed via /api/metrics."""
    _lock = threading.Lock()
    counters: dict[str, int] = {}
    timers: dict[str, list[float]] = {}

    @classmethod
    def inc(cls, name: str, n: int = 1):
        with cls._lock:
            cls.counters[name] = cls.counters.get(name, 0) + n

    @classmethod
    def observe(cls, name: str, seconds: float):
        with cls._lock:
            cls.timers.setdefault(name, []).append(round(seconds, 3))

    @classmethod
    def snapshot(cls) -> dict:
        with cls._lock:
            out = {"counters": dict(cls.counters), "timers": {}}
            for k, v in cls.timers.items():
                out["timers"][k] = {"n": len(v), "avg_s": round(sum(v)/len(v), 3), "max_s": max(v)}
            return out


def audit(event: str, case_id: str | None = None, actor: str = "user", **payload):
    """Append-only audit log: every tool call / agent step / mutation lands here.

    One JSONL file per day under data/audit/. This is the evidence chain the
    spec asks for (审计日志/执行轨迹).
    """
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "case_id": case_id,
        "actor": actor,
        **payload,
    }
    path = Path(settings.audit_dir) / f"audit-{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


class traced:
    """Context manager: time a block, record metric + audit entry."""
    def __init__(self, name: str, event: str | None = None, **kw):
        self.name, self.event, self.kw = name, event or name, kw

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        Metrics.observe(self.name, dt)
        status = "error" if exc else "ok"
        audit(self.event, status=status, duration_s=round(dt, 3), error=str(exc) if exc else None, **self.kw)
        log.info(self.event, duration_s=round(dt, 3), status=status)
        return False