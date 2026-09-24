"""Document parsing: PDF (pypdf), Excel (openpyxl/pandas), text, images(OCR hook).

Every parse produces page-anchored plain text so downstream citations can jump
back to the original location (溯源 requirement).
"""
import hashlib, json, re, unicodedata
from pathlib import Path
from dataclasses import asdict
from ..observe import log, Metrics
from .store import DocMeta


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def detect_kind(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {".pdf": "pdf", ".txt": "text", ".md": "text",
            ".xlsx": "xlsx", ".xls": "xlsx",
            ".png": "image", ".jpg": "image", ".jpeg": "image"}.get(ext, "unknown")


def parse_pdf(path: Path) -> tuple[list[dict], int]:
    """Return (pages:[{page,text}], total_pages)."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for i, pg in enumerate(reader.pages, start=1):
        try:
            txt = pg.extract_text() or ""
        except Exception as e:  # noqa: BLE001
            log.warning("pdf_page_fail", page=i, error=str(e)[:120])
            txt = ""
        pages.append({"page": i, "text": _clean(txt)})
    scanned = sum(1 for p in pages if len(p["text"].strip()) < 20)
    if scanned and scanned >= max(1, len(pages) // 2):
        # mostly image-based pdf -> try OCR fallback
        ocr = _ocr_pdf_fallback(path)
        if ocr:
            return ocr, len(ocr)
    return pages, len(pages)


def _ocr_pdf_fallback(path: Path) -> list[dict] | None:
    """Try pdftoppm+tesseract if available (offline OCR). Returns None if toolchain absent."""
    import shutil, subprocess, tempfile
    if not (shutil.which("pdftoppm") and shutil.which("tesseract")):
        log.info("ocr_toolchain_missing hint='apt-get install poppler-utils tesseract-ocr tesseract-ocr-chi-sim'")
        return None
    out_pages = []
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", "200", "-png", str(path), str(Path(td) / "pg")], check=True)
        for img in sorted(Path(td).glob("*.png")):
            m = re.search(r"(\d+)\.png$", img.name)
            page_no = int(m.group(1)) if m else len(out_pages) + 1
            r = subprocess.run(["tesseract", str(img), "stdout", "-l", "chi_sim+eng"],
                               capture_output=True, text=True)
            out_pages.append({"page": page_no, "text": _clean(r.stdout)})
    Metrics.inc("ocr_pages", len(out_pages))
    return out_pages or None


def parse_xlsx(path: Path) -> tuple[list[dict], int]:
    import pandas as pd
    book = pd.read_excel(str(path), sheet_name=None, dtype=str)
    pages = []
    for sheet, df in book.items():
        rows = df.fillna("").astype(str)
        lines = ["\t".join(rows.columns)] + ["\t".join(r) for r in rows.values.tolist()]
        # chunk big sheets into pseudo-pages of 80 rows for citation granularity
        body = lines[1:]
        for pi in range(0, max(1, len(body)), 80):
            seg = body[pi:pi + 80]
            pages.append({"page": pi // 80 + 1, "sheet": sheet,
                          "text": lines[0] + "\n" + "\n".join(seg)})
    return pages, len(pages)


def parse_text(path: Path) -> tuple[list[dict], int]:
    txt = path.read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()
    pages = [{"page": 1, "text": _clean(txt)}] if len(lines) <= 400 else \
            [{"page": i + 1, "text": _clean("\n".join(lines[j:j + 400]))}
             for i, j in enumerate(range(0, len(lines), 400))]
    return pages, len(pages)


def _clean(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def parse_document(case_id: str, doc: DocMeta) -> list[dict]:
    """Parse stored upload into page-anchored blocks; persist as <doc_id>.text.json"""
    from .store import doc_path
    base = doc_path(case_id, doc.doc_id)
    src = base.with_suffix(base.suffix)  # original saved with its own ext under documents/
    # uploads are stored as documents/<doc_id><ext>; find it
    cands = list(base.parent.glob(doc.doc_id + "*"))
    src = cands[0] if cands else src
    kind = doc.kind
    if kind == "pdf":
        pages, n = parse_pdf(src)
    elif kind == "xlsx":
        pages, n = parse_xlsx(src)
    elif kind == "text":
        pages, n = parse_text(src)
    elif kind == "image":
        # single-image OCR via tesseract if present
        import shutil, subprocess
        if shutil.which("tesseract"):
            r = subprocess.run(["tesseract", str(src), "stdout", "-l", "chi_sim+eng"],
                               capture_output=True, text=True)
            pages, n = [{"page": 1, "text": _clean(r.stdout)}], 1
        else:
            pages, n = [{"page": 1, "text": ""}], 1
            doc.error = "缺少离线OCR工具链(tesseract)，图片未识别"
    else:
        pages, n = [], 0
        doc.error = f"不支持的文件类型: {kind}"
    doc.pages = n
    doc.status = "parsed" if any(p["text"].strip() for p in pages) else ("failed" if doc.error else "parsed")
    out = base.with_suffix(".text.json")
    out.write_text(json.dumps({"doc_id": doc.doc_id, "name": doc.name, "pages": pages},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    Metrics.inc("docs_parsed")
    return pages