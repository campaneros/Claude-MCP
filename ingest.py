"""
ingest.py

Converte i PDF in ./docs in Markdown strutturato e ne estrae l'indice delle
sezioni, cosi' che Claude possa navigare i documenti per struttura invece che
per sola similarita' vettoriale.

Produce:
    md/<nome>.md        Markdown del documento
    index/<nome>.json   albero delle sezioni con pagine e dimensioni
    index/map.json      mappa compatta dell'intero corpus
    db/                 indice vettoriale (ricerca di ripiego)

L'indicizzazione e' INCREMENTALE: vengono processati solo i file nuovi o
modificati; i file rimossi da ./docs vengono ripuliti.

Uso:
    python ingest.py               # conversione veloce (pymupdf4llm)
    python ingest.py --marker      # formule in LaTeX; lento, richiede marker-pdf
    python ingest.py --rebuild     # azzera tutto e riparte da capo
    python ingest.py --no-vectors  # solo struttura, niente indice vettoriale
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import pymupdf
from pymupdf4llm.helpers.pymupdf_rag import to_markdown
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
SLIDES_DIR = BASE_DIR / "slides"
MD_DIR = BASE_DIR / "md"
INDEX_DIR = BASE_DIR / "index"
DB_DIR = BASE_DIR / "db"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
MAP_PATH = INDEX_DIR / "map.json"
FTS_PATH = INDEX_DIR / "fts.sqlite"
COLLECTION_NAME = "documentazione"

# Versioni della pipeline. Il manifest registra con quale versione ogni file e'
# stato processato: cosi' un miglioramento del codice non resta invisibile sui
# documenti gia' indicizzati, che altrimenti risulterebbero "aggiornati" solo
# perche' il PDF non e' cambiato.
#   PIPELINE_VERSION -> conversione e struttura (richiede riconversione completa)
#   FIGURES_VERSION  -> rilevamento figure (si aggiorna da solo, in secondi,
#                       perche' non dipende dal convertitore)
PIPELINE_VERSION = 3
FIGURES_VERSION = 2

# Versione dello schema degli indici. Aumentandola, i documenti gia'
# indicizzati con una versione precedente vengono aggiornati alla prima
# esecuzione utile, senza riconvertirli da capo.
#   1 -> indice iniziale
#   2 -> aggiunto il rilevamento delle figure
INDEX_VERSION = 2

# Marcatore di pagina nel Markdown: permette di risalire alla pagina di ogni
# riga senza inquinare il testo che verra' mostrato al modello.
PAGE_MARK = "<!--page:{n}-->"
PAGE_MARK_RE = re.compile(r"^<!--page:(\d+)-->$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_entry(fingerprint: str) -> dict:
    return {"hash": fingerprint,
            "pipeline": PIPELINE_VERSION,
            "figures": FIGURES_VERSION}


def entry_hash(entry) -> str:
    """Hash di una voce di manifest, accettando anche il vecchio formato."""
    return entry if isinstance(entry, str) else (entry or {}).get("hash", "")


def entry_version(entry, key: str) -> int:
    """Versione registrata per una voce di manifest.

    Per le voci scritte prima del versionamento si assume che testo e struttura
    siano validi e che il solo rilevamento figure sia da rifare. E' una scelta
    deliberata: riconvertire il testo costerebbe ore di Marker, aggiornare le
    figure costa secondi. Chi vuole comunque rifare tutto usa --rebuild.
    """
    if isinstance(entry, str):
        return PIPELINE_VERSION if key == "pipeline" else 0
    return (entry or {}).get(key, 0)


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# Un titolo numerato tipico dei documenti tecnici: "3 Chapter", "2.1.4 Timing".
# Serve a ritrovare l'inizio del titolo quando finisce in coda a una riga che
# contiene anche corpo del testo.
NUMBERED_TITLE_RE = re.compile(r"(?:(?<=^)|(?<=[.\s]))(\d+(?:\.\d+)*)\s+([A-Z][^\n]*)$")


def split_heading_line(content: str) -> tuple[str, str, str]:
    """Separa una riga di heading in (corpo_prima, titolo, corpo_dopo).

    Il convertitore puo' fondere heading e testo adiacente in una sola riga, in
    due modi: "**Titolo** corpo..." oppure "corpo... Titolo". Se il titolo non
    viene isolato, finisce nella mappa e nei path delle sezioni, sprecando
    contesto e rendendo illeggibili le citazioni.

    Euristica, non magia: ci si affida al grassetto quando c'e', altrimenti al
    fatto che i documenti tecnici numerano le sezioni. Nel dubbio si restituisce
    tutto come titolo, lasciando che clean_heading lo tronchi.
    """
    bold = re.search(r"\*\*(.+?)\*\*", content)
    if bold:
        return content[:bold.start()].strip(), bold.group(1).strip(), content[bold.end():].strip()

    m = NUMBERED_TITLE_RE.search(content)
    if m and m.start() > 0:
        title = f"{m.group(1)} {m.group(2)}".strip()
        if len(title) < 100:
            return content[:m.start()].strip(), title, ""

    return "", content.strip(), ""


def clean_heading(text: str) -> str:
    """Normalizza un titolo di sezione.

    Rimuove il grassetto che il convertitore applica ai titoli e tronca quelli
    abnormi: un "titolo" di 300 caratteri e' quasi sempre corpo del testo
    scambiato per heading, e in mappa sprecherebbe contesto.
    """
    text = re.sub(r"\*+", "", text).strip()
    if len(text) > 150:
        text = text[:150].rsplit(" ", 1)[0] + "..."
    return text
    """Normalizza un titolo di sezione.

    Rimuove il grassetto che pymupdf4llm applica ai titoli e tronca quelli
    abnormi: un "titolo" di 300 caratteri e' quasi sempre corpo del testo
    scambiato per heading, e in mappa sprecherebbe contesto.
    """
    text = re.sub(r"\*+", "", text).strip()
    if len(text) > 150:
        text = text[:150].rsplit(" ", 1)[0] + "..."
    return text


# --------------------------------------------------------------------------
# Conversione PDF -> Markdown
# --------------------------------------------------------------------------

_MARKER_CONVERTER = None


def convert_pymupdf(pdf_path: Path) -> str:
    """Conversione veloce, senza modelli ML. Le formule display vanno perse."""
    with pymupdf.open(str(pdf_path)) as doc:
        pages = to_markdown(doc, page_chunks=True, show_progress=False)
    parts = []
    for page in pages:
        parts.append(PAGE_MARK.format(n=page["metadata"]["page"]))
        parts.append(page["text"].strip())
    return "\n".join(parts)


# Marker, con paginate_output attivo, separa le pagine con una riga del tipo
# "{0}------------------------------------------------" (indice da 0).
MARKER_PAGE_RE = re.compile(r"^\{(\d+)\}-{10,}\s*$")


def convert_marker(pdf_path: Path) -> str:
    """Conversione con Marker: formule in LaTeX e numeri di pagina preservati.

    Lenta (minuti per documento, molto di piu' su un TDR) e richiede il download
    una tantum di alcuni GB di modelli. L'opzione paginate_output fa emettere a
    Marker un separatore di pagina, che qui viene tradotto nel marcatore interno
    usato dal resto della pipeline: senza questo le citazioni perderebbero il
    numero di pagina.
    """
    global _MARKER_CONVERTER
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError as exc:
        raise RuntimeError(
            "Marker non e' installato. Installalo con 'pip install marker-pdf' "
            "oppure usa 'python ingest.py --fast'."
        ) from exc

    if _MARKER_CONVERTER is None:
        _MARKER_CONVERTER = PdfConverter(
            artifact_dict=create_model_dict(),
            config={
                "paginate_output": True,   # indispensabile per le citazioni
                "extract_images": False,   # non salviamo le immagini
            },
        )

    text, _, _ = text_from_rendered(_MARKER_CONVERTER(str(pdf_path)))

    out, seen_page = [], False
    for line in text.splitlines():
        m = MARKER_PAGE_RE.match(line.strip())
        if m:
            # Marker numera le pagine da 0, il resto della pipeline da 1.
            out.append(PAGE_MARK.format(n=int(m.group(1)) + 1))
            seen_page = True
        else:
            out.append(line)
    if not seen_page:
        # Nessun separatore trovato: meglio un documento senza pagine che
        # nessun documento, ma va segnalato perche' le citazioni ne risentono.
        print(f"\nATTENZIONE: nessun separatore di pagina in {pdf_path.name}; "
              "le citazioni indicheranno solo la sezione.", file=sys.stderr)
        out.insert(0, PAGE_MARK.format(n=1))
    return "\n".join(out)


# --------------------------------------------------------------------------
# Estrazione della struttura
# --------------------------------------------------------------------------

def parse_outline(markdown: str) -> tuple[list[str], list[dict]]:
    """Estrae righe pulite e albero delle sezioni da un Markdown con marcatori.

    Ogni sezione ha: id, level, title, path, line_start, line_end,
    page_start, page_end, n_chars. 'path' include i titoli dei livelli superiori.
    """
    lines: list[str] = []
    page_of_line: list[int] = []
    current_page = 1

    for raw in markdown.splitlines():
        m = PAGE_MARK_RE.match(raw.strip())
        if m:
            current_page = int(m.group(1))
            continue
        lines.append(raw)
        page_of_line.append(current_page)

    # Il convertitore puo' fondere heading e testo adiacente in una sola riga:
    # qui il titolo viene isolato e il corpo reinserito come righe normali.
    split_lines: list[str] = []
    split_pages: list[int] = []
    for line, page in zip(lines, page_of_line):
        m = HEADING_RE.match(line)
        if m:
            hashes, content = m.group(1), m.group(2)
            before, title, after = split_heading_line(content)
            if before or after:
                if before:
                    split_lines.append(before)
                    split_pages.append(page)
                split_lines.append(f"{hashes} {title}")
                split_pages.append(page)
                if after:
                    split_lines.append(after)
                    split_pages.append(page)
                continue
        split_lines.append(line)
        split_pages.append(page)
    lines, page_of_line = split_lines, split_pages

    headings = [
        (i, len(m.group(1)), clean_heading(m.group(2)))
        for i, line in enumerate(lines)
        if (m := HEADING_RE.match(line))
    ]

    sections = []
    stack: list[tuple[int, str]] = []
    for idx, (line_no, level, title) in enumerate(headings):
        # La sezione finisce dove ne inizia una di livello uguale o superiore.
        end = len(lines)
        for next_line, next_level, _ in headings[idx + 1:]:
            if next_level <= level:
                end = next_line
                break

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        sections.append({
            "id": f"s{idx + 1}",
            "level": level,
            "title": title,
            "path": " > ".join(t for _, t in stack),
            "line_start": line_no,
            "line_end": end,
            "page_start": page_of_line[line_no] if page_of_line else 1,
            "page_end": page_of_line[min(end, len(page_of_line)) - 1] if page_of_line else 1,
            "n_chars": len("\n".join(lines[line_no:end])),
        })

    return lines, sections


def guess_title(sections: list[dict], meta_title: str, fallback: str) -> str:
    """Titolo del documento.

    I metadati del PDF, quando ci sono, sono piu' affidabili del primo heading:
    su un TDR il primo heading e' spesso "1 Introduction", che non identifica
    il documento nella mappa.
    """
    meta_title = (meta_title or "").strip()
    # Molti generatori lasciano metadati inutili: vanno scartati.
    meta_title = re.sub(r"^Microsoft (Word|PowerPoint) - ", "", meta_title).strip()
    junk = {"untitled", "untitled document", "document", "document1",
            "presentation", "no title", "-"}
    if (len(meta_title) > 3
            and meta_title.lower() not in junk
            and not meta_title.lower().endswith((".pdf", ".doc", ".docx", ".tex"))):
        return meta_title
    for s in sections:
        if s["level"] == 1:
            return s["title"]
    return sections[0]["title"] if sections else fallback


def first_paragraph(lines: list[str], limit: int = 400) -> str:
    """Primo blocco di prosa: fa da abstract nella mappa del corpus."""
    buf: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or HEADING_RE.match(s) or s.startswith(("|", "<!--")):
            if buf:
                break
            continue
        buf.append(s)
        if sum(len(b) for b in buf) > limit:
            break
    text = " ".join(buf)
    return text[:limit] + "..." if len(text) > limit else text


# --------------------------------------------------------------------------
# Indice vettoriale di ripiego
# --------------------------------------------------------------------------

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end - overlap if end - overlap > start else end
    return chunks


def vector_records(name: str, lines: list[str], sections: list[dict],
                   kind: str = "document"):
    """Chunk ancorati alle sezioni: ogni chunk sa da quale sezione proviene.

    Viene indicizzato solo il corpo proprio di ogni sezione (fino all'inizio
    della prima sottosezione), per non duplicare il testo dei genitori.
    """
    ids, texts, metas = [], [], []
    for sec in sections:
        children = [s for s in sections
                    if s["line_start"] > sec["line_start"]
                    and s["line_end"] <= sec["line_end"]]
        body_end = min((c["line_start"] for c in children), default=sec["line_end"])
        body = "\n".join(lines[sec["line_start"]:body_end]).strip()
        if not body:
            continue
        for i, piece in enumerate(chunk_text(body)):
            ids.append(hashlib.sha256(f"{name}::{sec['id']}::{i}".encode()).hexdigest()[:24])
            texts.append(piece)
            metas.append({
                "source": name,
                "kind": kind,
                "section_id": sec["id"],
                "section": sec["path"][:300],
                "page": sec["page_start"],
            })
    return ids, texts, metas


def figure_records(name: str, figures: list[dict], kind: str):
    """Chunk ricercabili a partire dalle figure.

    Didascalia e testo interno al grafico (etichette degli assi, legenda,
    annotazioni) diventano un'unita' a se'. E' quello che rende trovabile un
    risultato mostrato solo in un plot: la curva no, ma "Stochastic term =
    8.2%" scritto accanto sì. Il chunk e' marcato come figura, cosi' chi legge
    sa che sono frammenti tipografici e non prosa.
    """
    ids, texts, metas = [], [], []
    for fig in figures:
        pieces = [p for p in (fig.get("caption", ""), fig.get("text", "")) if p]
        if not pieces:
            continue
        body = f"[Figura p. {fig['page']}] " + " | ".join(pieces)
        ids.append(hashlib.sha256(f"{name}::fig::{fig['id']}".encode()).hexdigest()[:24])
        texts.append(body)
        metas.append({
            "source": name,
            "kind": kind,
            "section_id": fig["id"],
            "section": (fig.get("caption") or f"Figura a p. {fig['page']}")[:300],
            "page": fig["page"],
        })
    return ids, texts, metas


def build_fts(processed: list[Path], all_sources: set[str],
              all_indexed: Optional[list[Path]] = None) -> None:
    """Costruisce l'indice BM25 (SQLite FTS5) per la ricerca lessicale.

    Complementare a quello vettoriale: sul lessico tecnico molto specifico
    (sigle, nomi di file, formule, identificativi tipo 'CMS-TDR-015' o
    'PbWO4') il match esatto dei termini batte nettamente gli embedding, che
    su questi token sono quasi ciechi.

    Non richiede dipendenze: FTS5 e' incluso nel modulo sqlite3 standard.
    """
    import sqlite3

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(FTS_PATH)
    try:
        # Schema con 'kind': se esiste una tabella creata da una versione
        # precedente, senza quella colonna, va ricreata da zero.
        existing = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if existing and "kind" not in existing[0]:
            con.execute("DROP TABLE chunks")
            existing = None
            print("Indice BM25 ricreato per aggiornamento di schema.",
                  file=sys.stderr)
        con.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
            "  source, kind UNINDEXED, section_id UNINDEXED, section,"
            "  page UNINDEXED, text,"
            "  tokenize='unicode61 remove_diacritics 2'"
            ")"
        )
        rebuilt_schema = existing is None
        # Ripulisce i documenti non piu' presenti e quelli riconvertiti.
        rows = con.execute("SELECT DISTINCT source FROM chunks").fetchall()
        stale = {r[0] for r in rows} - all_sources
        for name in stale | {p.name for p in (all_indexed or processed)}:
            con.execute("DELETE FROM chunks WHERE source = ?", (name,))

        targets = processed
        if rebuilt_schema and all_indexed:
            targets = all_indexed
        for pdf_path in tqdm(targets, desc="BM25", unit="pdf"):
            data = load_json(INDEX_DIR / f"{pdf_path.stem}.json", None)
            if not data:
                continue
            kind = data.get("kind", "document")
            _, texts, metas = vector_records(
                pdf_path.name, data["lines"], data["sections"], kind)
            _, ftexts, fmetas = figure_records(
                pdf_path.name, data.get("figures", []), kind)
            texts += ftexts
            metas += fmetas
            con.executemany(
                "INSERT INTO chunks "
                "(source, kind, section_id, section, page, text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(m["source"], m["kind"], m["section_id"], m["section"],
                  str(m["page"]), t) for t, m in zip(texts, metas)],
            )
        con.commit()
    finally:
        con.close()


def build_vectors(processed: list[Path]) -> None:
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("chromadb non installato: salto l'indice vettoriale.", file=sys.stderr)
        return

    client = chromadb.PersistentClient(
        path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    for pdf_path in tqdm(processed, desc="Vettori", unit="pdf"):
        data = load_json(INDEX_DIR / f"{pdf_path.stem}.json", None)
        if not data:
            continue
        collection.delete(where={"source": pdf_path.name})
        kind = data.get("kind", "document")
        ids, texts, metas = vector_records(
            pdf_path.name, data["lines"], data["sections"], kind)
        fids, ftexts, fmetas = figure_records(
            pdf_path.name, data.get("figures", []), kind)
        ids += fids; texts += ftexts; metas += fmetas
        for i in range(0, len(ids), 200):
            collection.upsert(ids=ids[i:i + 200], documents=texts[i:i + 200],
                              metadatas=metas[i:i + 200])


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

CAPTION_RE = re.compile(
    r"^\s*((?:Fig(?:ure|\.)?|Tab(?:le|\.)?|Figura|Tabella)\s*\.?\s*\d+[.:)]?\s+.{5,})",
    re.IGNORECASE,
)

# Soglie per il rilevamento delle figure. Un plot vettoriale e' fatto di
# centinaia di piccoli tratti: si riconosce dalla densita', non dal tipo.
MIN_FIGURE_AREA_FRAC = 0.02   # frazione minima della pagina
MIN_DRAW_ITEMS = 8            # tratti minimi perche' sia un grafico e non un bordo
FIGURE_MERGE_MARGIN = 12      # pt di tolleranza nel fondere regioni vicine
FIGURE_DPI = 130


def _overlaps(a, b, margin: float = 0.0) -> bool:
    """Sovrapposizione fra due rettangoli, tolleranti quelli degeneri.

    Rect.intersects() di PyMuPDF restituisce False se uno dei due ha larghezza
    o altezza nulla, e assi, tacche e linee di griglia dei grafici sono
    esattamente cosi'. Usarlo farebbe sparire la maggior parte degli elementi
    di un plot dal conteggio, e la figura non verrebbe riconosciuta.
    """
    return (a.x0 - margin <= b.x1 and b.x0 <= a.x1 + margin
            and a.y0 - margin <= b.y1 and b.y0 <= a.y1 + margin)


def _merge_rects(rects: list, margin: float = FIGURE_MERGE_MARGIN) -> list:
    """Fonde rettangoli che si toccano o quasi, iterando fino a stabilita'."""
    boxes = [pymupdf.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out: list = []
        for box in boxes:
            for i, other in enumerate(out):
                if _overlaps(box, other, margin):
                    out[i] = pymupdf.Rect(
                        min(other.x0, box.x0), min(other.y0, box.y0),
                        max(other.x1, box.x1), max(other.y1, box.y1))
                    changed = True
                    break
            else:
                out.append(box)
        boxes = out
    return boxes


def detect_figures(page, page_num: int) -> list[dict]:
    """Individua le regioni-figura di una pagina.

    Nei paper di fisica i grafici sono quasi sempre grafica VETTORIALE, non
    immagini incorporate: page.get_images() non li trova. Vengono quindi
    raggruppati gli elementi di disegno vicini, e si tiene un raggruppamento
    solo se e' abbastanza grande e abbastanza fitto da essere un grafico e non
    una riga di tabella o un bordo.

    Per ogni regione si raccolgono la didascalia (cercata sotto, poi sopra) e
    il testo interno: etichette degli assi, legenda, annotazioni. Quel testo
    contiene spesso i numeri che interessano davvero.
    """
    page_rect = page.rect
    page_area = (page_rect.width * page_rect.height) or 1

    candidates = []
    counts = []

    # Immagini incorporate (plot raster, foto, schemi esportati come PNG).
    for img in page.get_images(full=True):
        try:
            for r in page.get_image_rects(img[0]):
                candidates.append(r)
                counts.append((r, MIN_DRAW_ITEMS))  # sempre significative
        except Exception:
            continue

    # Grafica vettoriale.
    draw_rects = []
    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"])
        if r.width < 2 and r.height < 2:
            continue
        # Un rettangolo grande quanto la pagina e' uno sfondo, non una figura.
        if (r.width * r.height) / page_area > 0.9:
            continue
        draw_rects.append(r)
    candidates.extend(draw_rects)

    figures = []
    for region in _merge_rects(candidates):
        area_frac = (region.width * region.height) / page_area
        if area_frac < MIN_FIGURE_AREA_FRAC:
            continue
        n_items = sum(1 for r in draw_rects if _overlaps(region, r))
        has_image = any(_overlaps(region, r) for r, _ in counts)
        if not has_image and n_items < MIN_DRAW_ITEMS:
            continue

        inner = page.get_text("text", clip=region, sort=True).strip()
        caption = _find_caption(page, region)
        figures.append({
            "page": page_num,
            "bbox": [round(v, 1) for v in (region.x0, region.y0, region.x1, region.y1)],
            "caption": caption,
            "text": " ".join(inner.split())[:1200],
            "n_items": n_items,
        })

    figures.sort(key=lambda f: (f["bbox"][1], f["bbox"][0]))
    return figures


def _find_caption(page, region) -> str:
    """Didascalia associata a una regione: prima sotto, poi sopra."""
    below = pymupdf.Rect(region.x0 - 40, region.y1,
                         region.x1 + 40, region.y1 + 90)
    above = pymupdf.Rect(region.x0 - 40, max(region.y0 - 90, 0),
                         region.x1 + 40, region.y0)
    for area in (below, above):
        text = page.get_text("text", clip=area, sort=True)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            if CAPTION_RE.match(line):
                # Le didascalie proseguono su piu' righe.
                chunk = " ".join(lines[i:i + 4])
                return " ".join(chunk.split())[:400]
    return ""


def slide_title(page) -> str:
    """Titolo di una slide: il testo con il carattere piu' grande, in alto.

    Le presentazioni non hanno una gerarchia di heading, quindi il titolo va
    dedotto dalla tipografia. In caso di dubbio si ripiega sulla prima riga
    non vuota: meglio un titolo mediocre che nessun titolo, perche' e' cio' che
    finisce nell'indice e permette di orientarsi.
    """
    try:
        data = page.get_text("dict")
    except Exception:
        return ""

    best_size, best_text, best_top = 0.0, "", None
    page_height = page.rect.height or 1
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(sp.get("text", "") for sp in line.get("spans", [])).strip()
            if not text or len(text) > 200:
                continue
            size = max((sp.get("size", 0) for sp in line.get("spans", [])), default=0)
            top = line.get("bbox", [0, 0, 0, 0])[1] / page_height
            # Solo il terzo superiore: i numeri di pagina e i loghi in fondo
            # possono avere caratteri grandi ma non sono titoli.
            if top > 0.35:
                continue
            if size > best_size:
                best_size, best_text, best_top = size, text, top

    if best_text:
        return best_text[:150]
    for line in (page.get_text("text") or "").splitlines():
        if line.strip():
            return line.strip()[:150]
    return ""


def strip_repeated_lines(pages: list[list[str]], threshold: float = 0.6) -> set[str]:
    """Individua le righe che ricorrono nella maggioranza delle slide.

    Nelle presentazioni piede e intestazione (nome del relatore, conferenza,
    data, numerazione) si ripetono su ogni slide. Indicizzarli significa
    riempire la ricerca di match inutili su quei termini, quindi vanno tolti.
    Con poche slide la statistica non e' affidabile e non si tocca nulla.
    """
    if len(pages) < 4:
        return set()
    counts: dict[str, int] = {}
    for lines in pages:
        for line in set(l.strip() for l in lines if l.strip()):
            counts[line] = counts.get(line, 0) + 1
    minimum = max(3, int(len(pages) * threshold))
    # Le righe lunghe che si ripetono sono probabilmente contenuto reale
    # (una citazione ripetuta), non decorazione: si conservano.
    return {line for line, n in counts.items() if n >= minimum and len(line) <= 120}


def backfill_figures(pdf_files: list[Path], corpus_map: dict,
                     force: bool = False) -> list[Path]:
    """Aggiunge le figure ai documenti indicizzati prima che esistessero.

    Il rilevamento delle figure lavora sul PDF originale e non dipende dalla
    conversione in Markdown: puo' quindi essere fatto a posteriori in pochi
    secondi, invece di riconvertire tutto con Marker (che su un corpus di
    centinaia di pagine significherebbe ore).

    Ritorna i file aggiornati, che vanno reindicizzati in BM25 e nei vettori
    perche' i chunk-figura entrino nella ricerca.
    """
    updated = []
    todo = []
    for pdf_path in pdf_files:
        data = load_json(INDEX_DIR / f"{pdf_path.stem}.json", None)
        if data is None:
            continue
        if force or data.get("index_version", 1) < INDEX_VERSION:
            todo.append((pdf_path, data))

    if not todo:
        return []

    print(f"Aggiornamento indice: {len(todo)} documenti indicizzati con una "
          "versione precedente. Rilevo le figure senza riconvertirli.")

    for pdf_path, data in tqdm(todo, desc="Figure", unit="pdf"):
        try:
            figures = []
            with pymupdf.open(str(pdf_path)) as doc:
                for i, page in enumerate(doc, start=1):
                    for fig in detect_figures(page, i):
                        fig["id"] = f"f{len(figures) + 1}"
                        figures.append(fig)
        except Exception as exc:
            print(f"\nATTENZIONE: figure non rilevate in {pdf_path.name} "
                  f"({type(exc).__name__}: {exc}).", file=sys.stderr)
            continue

        data["figures"] = figures
        data["index_version"] = INDEX_VERSION
        save_json(INDEX_DIR / f"{pdf_path.stem}.json", data)
        if pdf_path.name in corpus_map:
            corpus_map[pdf_path.name]["n_figures"] = len(figures)
        updated.append(pdf_path)

    print(f"Figure aggiunte: {sum(len(load_json(INDEX_DIR / f'{p.stem}.json', {}).get('figures', [])) for p in updated)} "
          f"in {len(updated)} documenti.")
    return updated


def process_slides(pdf_path: Path) -> dict:
    """Converte una presentazione: una unita' per slide, senza gerarchia.

    Le presentazioni non hanno sezioni annidate; forzarle nel modello dei
    documenti produrrebbe alberi degeneri. Ogni slide diventa una "sezione" di
    livello 1, e la navigazione avviene per intervalli di slide.
    """
    figures = []
    with pymupdf.open(str(pdf_path)) as doc:
        n_pages = doc.page_count
        meta_title = (doc.metadata or {}).get("title", "")
        raw = []
        for i, page in enumerate(doc, start=1):
            raw.append((slide_title(page),
                        (page.get_text("text", sort=True) or "").splitlines()))
            for fig in detect_figures(page, i):
                fig["id"] = f"f{len(figures) + 1}"
                figures.append(fig)

    boilerplate = strip_repeated_lines([body for _, body in raw])

    lines: list[str] = []
    sections: list[dict] = []
    parts: list[str] = []

    for i, (title, body_lines) in enumerate(raw, start=1):
        clean = []
        for line in body_lines:
            stripped = line.strip()
            if not stripped or stripped in boilerplate:
                continue
            # Il titolo compare anche nel testo della pagina: nell'intestazione
            # c'e' gia', ripeterlo raddoppierebbe il peso di quei termini.
            if title and stripped == title.strip():
                continue
            clean.append(stripped)
        body = "\n".join(clean)

        heading = f"# Slide {i}" + (f" — {title}" if title else "")
        parts.append(PAGE_MARK.format(n=i))
        parts.append(heading)
        parts.append(body)

        start = len(lines)
        lines.append(heading)
        lines.extend(clean)
        sections.append({
            "id": f"s{i}",
            "level": 1,
            "title": f"Slide {i}" + (f" — {title}" if title else ""),
            "path": f"Slide {i}" + (f" — {title}" if title else ""),
            "line_start": start,
            "line_end": len(lines),
            "page_start": i,
            "page_end": i,
            "n_chars": len(heading) + len(body),
        })

    if not any(l.strip() for l in lines) and not figures:
        raise ValueError(
            "nessun testo ne' figure estratti: presentazione probabilmente "
            "scansionata (servirebbe un OCR, es. ocrmypdf)"
        )

    stem = pdf_path.stem
    MD_DIR.mkdir(parents=True, exist_ok=True)
    (MD_DIR / f"{stem}.md").write_text("\n".join(parts))

    save_json(INDEX_DIR / f"{stem}.json", {
        "source": pdf_path.name,
        "kind": "slides",
        "index_version": INDEX_VERSION,
        "figures": figures,
        "lines": lines,
        "sections": sections,
        "n_pages": n_pages,
    })

    titles = [s["title"] for s in sections]
    return {
        "source": pdf_path.name,
        "kind": "slides",
        "title": guess_title([], meta_title, stem),
        "n_pages": n_pages,
        "n_sections": len(sections),
        "n_figures": len(figures),
        "abstract": first_paragraph(lines),
        "slide_titles": titles,
        "top_sections": [],
        "converter": "pymupdf",
    }


def refresh_figures(pdf_path: Path, corpus_map: dict) -> int:
    """Ricalcola solo le figure di un file gia' indicizzato.

    Il rilevamento figure lavora direttamente sul PDF e non dipende dal
    convertitore usato per il testo: aggiornarlo costa secondi, non le ore di
    una riconversione con Marker. Struttura e markdown restano intatti.

    Ritorna il numero di figure trovate.
    """
    index_path = INDEX_DIR / f"{pdf_path.stem}.json"
    data = load_json(index_path, None)
    if data is None:
        raise FileNotFoundError(f"indice mancante per {pdf_path.name}")

    figures = []
    with pymupdf.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc, start=1):
            for fig in detect_figures(page, i):
                fig["id"] = f"f{len(figures) + 1}"
                figures.append(fig)

    data["figures"] = figures
    save_json(index_path, data)
    if pdf_path.name in corpus_map:
        corpus_map[pdf_path.name]["n_figures"] = len(figures)
    return len(figures)


def process_pdf(pdf_path: Path, use_marker: bool) -> dict:
    """Converte un PDF, salva markdown e indice, ritorna la voce di mappa."""
    markdown = convert_marker(pdf_path) if use_marker else convert_pymupdf(pdf_path)
    lines, sections = parse_outline(markdown)

    if not "".join(lines).strip():
        raise ValueError(
            "nessun testo estratto: probabilmente e' una scansione "
            "(servirebbe un OCR, es. ocrmypdf)"
        )

    stem = pdf_path.stem
    MD_DIR.mkdir(parents=True, exist_ok=True)
    (MD_DIR / f"{stem}.md").write_text(markdown)

    figures = []
    with pymupdf.open(str(pdf_path)) as doc:
        n_pages = doc.page_count
        meta_title = (doc.metadata or {}).get("title", "")
        for i, page in enumerate(doc, start=1):
            for fig in detect_figures(page, i):
                fig["id"] = f"f{len(figures) + 1}"
                figures.append(fig)

    save_json(INDEX_DIR / f"{stem}.json", {
        "source": pdf_path.name,
        "kind": "document",
        "index_version": INDEX_VERSION,
        "figures": figures,
        "lines": lines,
        "sections": sections,
        "n_pages": n_pages,
    })

    return {
        "source": pdf_path.name,
        "kind": "document",
        "title": guess_title(sections, meta_title, stem),
        "n_pages": n_pages,
        "n_sections": len(sections),
        "n_figures": len(figures),
        "abstract": first_paragraph(lines),
        "top_sections": [
            {"id": s["id"], "title": s["title"], "page": s["page_start"]}
            for s in sections if s["level"] <= 2
        ],
        "converter": "marker" if use_marker else "pymupdf4llm",
    }


def report(corpus_map: dict) -> None:
    """Riepilogo finale, con i segnali di allarme utili a capire se qualcosa
    non ha funzionato."""
    docs = [e for e in corpus_map.values() if e.get("kind", "document") == "document"]
    decks = [e for e in corpus_map.values() if e.get("kind") == "slides"]
    n_fig = sum(e.get("n_figures", 0) for e in corpus_map.values())
    tot_sec = sum(e["n_sections"] for e in docs)

    print(f"\nFatto. {len(docs)} documenti ({sum(e['n_pages'] for e in docs)} pagine, "
          f"{tot_sec} sezioni)"
          + (f", {len(decks)} presentazioni "
             f"({sum(e['n_pages'] for e in decks)} slide)" if decks else "")
          + (f", {n_fig} figure" if n_fig else "") + ".")

    if docs and tot_sec == 0:
        print("ATTENZIONE: nessuna sezione rilevata. Se i PDF non hanno titoli "
              "distinguibili per dimensione del carattere, la navigazione per "
              "struttura non funziona e resta solo la ricerca.", file=sys.stderr)

    # Un documento lungo senza figure e' sospetto: quasi ogni paper o TDR ne ha.
    sospetti = [e for e in corpus_map.values()
                if e["n_pages"] >= 15 and e.get("n_figures", 0) == 0]
    if sospetti:
        print(f"\nNota: {len(sospetti)} documenti lunghi risultano senza figure:",
              file=sys.stderr)
        for e in sorted(sospetti, key=lambda x: -x["n_pages"])[:8]:
            print(f"  {e['source']} ({e['n_pages']} pp.)", file=sys.stderr)
        print("Se contengono grafici, abbassa MIN_FIGURE_AREA_FRAC o "
              "MIN_DRAW_ITEMS in ingest.py e rilancia.", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Converte e indicizza i PDF in ./docs")
    ap.add_argument("--fast", action="store_true",
                    help="usa pymupdf4llm invece di Marker: pochi secondi per "
                         "documento, ma perde le formule")
    ap.add_argument("--rebuild", action="store_true",
                    help="azzera tutto e riconverte da capo")
    ap.add_argument("--no-vectors", action="store_true",
                    help="salta l'indice vettoriale di ripiego")
    ap.add_argument("--refresh-figures", action="store_true",
                    help="ricalcola le figure di tutti i file gia' indicizzati "
                         "senza riconvertire il testo: da usare dopo aver "
                         "regolato le soglie di rilevamento")
    args = ap.parse_args()
    use_marker = not args.fast

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    doc_files = sorted(DOCS_DIR.glob("*.pdf"))
    slide_files = sorted(SLIDES_DIR.glob("*.pdf"))
    pdf_files = doc_files + slide_files
    is_slides = {p.name for p in slide_files}

    dupes = {p.name for p in doc_files} & is_slides
    if dupes:
        print(f"Errore: stessi nomi in docs/ e slides/: {', '.join(sorted(dupes))}. "
              "I nomi devono essere univoci fra le due cartelle.", file=sys.stderr)
        return 1

    if not pdf_files:
        print(f"Nessun PDF trovato. Metti i documenti in {DOCS_DIR} e le "
              f"presentazioni in {SLIDES_DIR}, poi rilancia.")
        return 1

    if args.rebuild:
        for d in (MD_DIR, INDEX_DIR, DB_DIR):
            if d.exists():
                for f in sorted(d.rglob("*"), reverse=True):
                    if f.is_file():
                        f.unlink()
        print("Indice azzerato: riconversione completa.")

    manifest = load_json(MANIFEST_PATH, {})
    corpus_map = load_json(MAP_PATH, {})
    current = {p.name: file_fingerprint(p) for p in pdf_files}

    removed_sources: set[str] = set()
    for name in set(manifest) - set(current):
        manifest.pop(name, None)
        corpus_map.pop(name, None)
        (MD_DIR / f"{Path(name).stem}.md").unlink(missing_ok=True)
        (INDEX_DIR / f"{Path(name).stem}.json").unlink(missing_ok=True)
        removed_sources.add(name)
        print(f"Rimosso dall'indice: {name}")

    # Tre stati possibili per ogni file:
    #  - riconversione completa: il PDF e' cambiato, o la pipeline di
    #    conversione e' stata aggiornata;
    #  - solo figure: testo e struttura vanno bene, ma il rilevamento figure
    #    e' di una versione precedente (costa secondi);
    #  - niente da fare.
    todo, figs_only = [], []
    for pdf in pdf_files:
        entry = manifest.get(pdf.name)
        if entry_hash(entry) != current[pdf.name]:
            todo.append(pdf)
        elif entry_version(entry, "pipeline") != PIPELINE_VERSION:
            todo.append(pdf)
        elif (args.refresh_figures
              or entry_version(entry, "figures") != FIGURES_VERSION):
            figs_only.append(pdf)
    if removed_sources and not todo and not figs_only:
        try:
            build_fts([], set(current), pdf_files)
        except Exception:
            pass
    if figs_only:
        print(f"{len(figs_only)} file: aggiorno solo le figure "
              "(il testo e' gia' convertito, non serve rifarlo).")
        for pdf in tqdm(figs_only, desc="Figure", unit="pdf"):
            try:
                n = refresh_figures(pdf, corpus_map)
            except Exception as exc:
                print(f"\nATTENZIONE: figure non aggiornate per {pdf.name} "
                      f"({type(exc).__name__}: {exc}). Serve --rebuild per "
                      "questo file.", file=sys.stderr)
                continue
            entry = manifest.get(pdf.name)
            manifest[pdf.name] = {
                "hash": current[pdf.name],
                "pipeline": entry_version(entry, "pipeline") or PIPELINE_VERSION,
                "figures": FIGURES_VERSION,
            }
            tqdm.write(f"  {pdf.name}: {n} figure")
            save_json(MANIFEST_PATH, manifest)
            save_json(MAP_PATH, corpus_map)

    if not todo and not figs_only:
        print(f"Tutti i {len(pdf_files)} PDF sono gia' aggiornati. Niente da fare.")
        save_json(MANIFEST_PATH, manifest)
        save_json(MAP_PATH, corpus_map)
        return 0

    if not todo:
        save_json(MANIFEST_PATH, manifest)
        save_json(MAP_PATH, corpus_map)
        # Gli indici di ricerca vanno rifatti per i file toccati: le figure
        # sono anche unita' ricercabili.
        try:
            build_fts(figs_only, set(current), pdf_files)
        except Exception as exc:
            print(f"\nATTENZIONE: indice BM25 non aggiornato ({exc}).", file=sys.stderr)
        if not args.no_vectors:
            try:
                build_vectors(figs_only)
            except Exception as exc:
                print(f"\nATTENZIONE: indice vettoriale non aggiornato ({exc}).",
                      file=sys.stderr)
        report(corpus_map)
        return 0

    print(f"{len(pdf_files) - len(todo)} file gia' aggiornati, {len(todo)} da "
          f"(ri)convertire con {'Marker' if use_marker else 'pymupdf4llm'}.")
    if use_marker:
        try:
            import marker  # noqa: F401
        except ImportError:
            print("Marker non e' installato. Installalo con "
                  "'pip install marker-pdf', oppure usa 'python ingest.py --fast' "
                  "per la conversione veloce senza formule.", file=sys.stderr)
            return 1

        tot_pages = 0
        for p in [x for x in todo if x.name not in is_slides]:
            try:
                with pymupdf.open(str(p)) as d:
                    tot_pages += d.page_count
            except Exception:
                pass
        print(f"{tot_pages} pagine di documenti da processare con Marker "
              "(le presentazioni usano sempre l'estrazione veloce). "
              "qualche secondo per pagina: puo' voler dire ore su un corpus "
              "grande. Il progresso viene salvato dopo ogni documento, quindi "
              "puoi interrompere con Ctrl+C e riprendere rilanciando lo script.")
        print("NOTA: al primo avvio Marker scarica alcuni GB di modelli. La "
              "fase di download puo' restare silenziosa per parecchi minuti: "
              "non e' un blocco. Serve connessione di rete solo questa volta.")

    processed = []
    try:
        for pdf_path in tqdm(todo, desc="Documenti", unit="pdf"):
            started = time.monotonic()
            slides = pdf_path.name in is_slides
            try:
                entry = (process_slides(pdf_path) if slides
                         else process_pdf(pdf_path, use_marker))
            except RuntimeError as exc:
                # Marker assente o non inizializzabile: inutile insistere.
                print(f"\n{exc}", file=sys.stderr)
                break
            except Exception as exc:
                if use_marker and not slides:
                    # Un singolo PDF che manda in crisi Marker non deve far
                    # perdere il documento: si ripiega sul convertitore veloce.
                    print(f"\nATTENZIONE: Marker ha fallito su {pdf_path.name} "
                          f"({type(exc).__name__}: {exc}). "
                          "Riprovo con pymupdf4llm (senza formule).",
                          file=sys.stderr)
                    try:
                        entry = process_pdf(pdf_path, use_marker=False)
                    except Exception as exc2:
                        print(f"ATTENZIONE: {pdf_path.name} saltato "
                              f"({type(exc2).__name__}: {exc2}).", file=sys.stderr)
                        continue
                else:
                    print(f"\nATTENZIONE: {pdf_path.name} saltato "
                          f"({type(exc).__name__}: {exc}).", file=sys.stderr)
                    continue

            corpus_map[pdf_path.name] = entry
            manifest[pdf_path.name] = manifest_entry(current[pdf_path.name])
            processed.append(pdf_path)
            # Salvataggio dopo OGNI documento: una conversione con Marker puo'
            # durare ore, e un'interruzione non deve far perdere il lavoro.
            save_json(MANIFEST_PATH, manifest)
            save_json(MAP_PATH, corpus_map)
            if use_marker:
                tqdm.write(f"  {pdf_path.name}: {entry['n_pages']} pagine, "
                           f"{entry['n_sections']} sezioni, "
                           f"{time.monotonic() - started:.0f}s")
    except KeyboardInterrupt:
        print(f"\nInterrotto. {len(processed)} documenti completati e salvati; "
              "rilancia lo script per riprendere da dove eri.", file=sys.stderr)

    save_json(MANIFEST_PATH, manifest)
    save_json(MAP_PATH, corpus_map)

    if processed:
        # BM25 non richiede modelli ne' rete: si costruisce sempre.
        try:
            build_fts(processed + figs_only, set(current), pdf_files)
        except Exception as exc:
            print(f"\nATTENZIONE: indice BM25 non costruito "
                  f"({type(exc).__name__}: {exc}).", file=sys.stderr)

    if not args.no_vectors and processed:
        # L'indice strutturale (la parte importante) e' gia' su disco: se la
        # ricerca vettoriale non si costruisce, il sistema resta utilizzabile
        # tramite mappa e navigazione per sezioni.
        try:
            build_vectors(processed + figs_only)
        except Exception as exc:
            print(f"\nATTENZIONE: indice vettoriale non costruito "
                  f"({type(exc).__name__}: {exc}).\n"
                  "Causa tipica: il download del modello di embedding e' fallito "
                  "(serve rete al primo avvio).\n"
                  "La navigazione per struttura funziona comunque; solo "
                  "docs_search resta indisponibile. Rilancia per riprovare.",
                  file=sys.stderr)

    report(corpus_map)
    return 0


if __name__ == "__main__":
    sys.exit(main())
