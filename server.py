"""
server.py

MCP server che espone a Claude Desktop la documentazione convertita da
ingest.py.

Idea centrale: la MAPPA dell'intero corpus (titolo, abstract e sezioni di ogni
documento) viene caricata una volta sola nelle istruzioni del server e resta in
contesto per tutta la conversazione. Claude quindi sa sempre cosa possiede e
naviga per struttura ("apri la sezione 2.1 di quel TDR") invece di sparare
ricerche vettoriali alla cieca.

Tool esposti:
    - docs_outline        albero completo delle sezioni di un documento
    - docs_read_section   legge una sezione (con guardia sulle dimensioni)
    - docs_read_pages     legge un intervallo di pagine
    - docs_search         ricerca semantica, come ripiego

Avvio (transport stdio, gestito da Claude Desktop):
    python server.py
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Annotated, Any, Optional

# Con transport stdio, stdout e' il canale del protocollo JSON-RPC: qualsiasi
# print o log su stdout corromperebbe la comunicazione. Va silenziato PRIMA di
# importare le librerie che potrebbero loggare.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
for _name in ("chromadb", "httpx", "onnxruntime"):
    logging.getLogger(_name).setLevel(logging.ERROR)

from mcp.server.mcpserver import Image, MCPServer  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402
from pydantic import Field  # noqa: E402

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "index"
DB_DIR = BASE_DIR / "db"
MAP_PATH = INDEX_DIR / "map.json"
FTS_PATH = INDEX_DIR / "fts.sqlite"
COLLECTION_NAME = "documentazione"

# Una sezione piu' grande di questo viene restituita troncata, elencando le
# sottosezioni: in un TDR una sezione di primo livello puo' valere decine di
# pagine e saturerebbe il contesto.
MAX_SECTION_CHARS = 12000
MAX_PAGES_PER_READ = 25
# Tetto alla mappa inserita nelle istruzioni: oltre, si degrada a soli titoli.
MAX_MAP_CHARS = 40000
# Spazio minimo garantito all'elenco delle presentazioni, anche quando i
# documenti hanno gia' consumato tutto il budget.
MIN_DECK_CHARS = 2000
# Risoluzione con cui vengono rese le figure. Piu' alta = piu' leggibile ma
# piu' costosa in contesto; 130 dpi e' un buon compromesso su un plot.
FIGURE_DPI = 130
DOCS_DIR = BASE_DIR / "docs"
SLIDES_DIR = BASE_DIR / "slides"

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)

_doc_cache: dict[str, dict] = {}
_chroma_collection: Any = None


# --------------------------------------------------------------------------
# Accesso ai dati
# --------------------------------------------------------------------------

def find_pdf(source: str) -> Optional[Path]:
    """Percorso del PDF originale, necessario per generare le figure."""
    for folder in (DOCS_DIR, SLIDES_DIR):
        candidate = folder / source
        if candidate.exists():
            return candidate
    return None


def kind_of(entry: dict) -> str:
    """Tipo di una voce di mappa: 'document' o 'slides'."""
    return entry.get("kind", "document")


def split_corpus(corpus: dict) -> tuple[dict, dict]:
    """Separa documenti e presentazioni."""
    docs = {k: v for k, v in corpus.items() if kind_of(v) == "document"}
    decks = {k: v for k, v in corpus.items() if kind_of(v) == "slides"}
    return docs, decks


def load_map() -> dict:
    """Mappa del corpus prodotta da ingest.py ({} se assente)."""
    if MAP_PATH.exists():
        try:
            return json.loads(MAP_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def load_doc(source: str) -> Optional[dict]:
    """Indice completo di un documento (righe + sezioni), con cache."""
    if source in _doc_cache:
        return _doc_cache[source]
    path = INDEX_DIR / f"{Path(source).stem}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    _doc_cache[source] = data
    return data


def resolve_source(source: str, kind: Optional[str] = None) -> Optional[str]:
    """Accetta il nome esatto, senza estensione o con differenze di maiuscole.

    Con 'kind' la ricerca si limita a documenti o presentazioni, cosi' che i
    tool dei documenti non aprano per sbaglio una presentazione e viceversa.
    """
    corpus = load_map()
    if kind:
        corpus = {k: v for k, v in corpus.items() if kind_of(v) == kind}
    if source in corpus:
        return source
    wanted = Path(source).stem.lower()
    for name in corpus:
        if Path(name).stem.lower() == wanted:
            return name
    matches = [n for n in corpus if wanted in n.lower()]
    return matches[0] if len(matches) == 1 else None


def unknown_source_msg(source: str, kind: Optional[str] = None) -> str:
    corpus = load_map()
    if not corpus:
        return ("L'indice e' vuoto: l'utente deve mettere i documenti in ./docs, "
                "le presentazioni in ./slides, ed eseguire 'python ingest.py'.")
    docs, decks = split_corpus(corpus)
    stem = Path(source).stem.lower()

    def belongs(pool: dict) -> bool:
        return any(Path(n).stem.lower() == stem for n in pool)

    if kind == "slides":
        if belongs(docs):
            return (f"'{source}' e' un documento, non una presentazione: "
                    "usa docs_read_section o docs_outline.")
        return (f"Presentazione '{source}' non trovata. Disponibili: "
                + (", ".join(sorted(decks)) if decks else "nessuna"))
    if kind == "document":
        if belongs(decks):
            return (f"'{source}' e' una presentazione, non un documento: "
                    "usa slides_read o slides_search.")
        return (f"Documento '{source}' non trovato. Disponibili: "
                + ", ".join(sorted(docs)))
    return (f"'{source}' non trovato. Disponibili: " + ", ".join(sorted(corpus)))


def find_section(sections: list[dict], wanted: str):
    """Risolve una sezione da id ('s12') o da parte del titolo.

    Ritorna il dict della sezione, None se non trovata, o la lista dei
    candidati se la richiesta e' ambigua.
    """
    key = wanted.strip().lower()
    exact = next((s for s in sections if s["id"].lower() == key), None)
    if exact:
        return exact
    candidates = [s for s in sections if key in s["title"].lower()]
    if not candidates:
        candidates = [s for s in sections if key in s["path"].lower()]
    if not candidates:
        return None
    return candidates[0] if len(candidates) == 1 else candidates


def get_collection():
    """Collection Chroma per la ricerca di ripiego (None se non disponibile)."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    if not DB_DIR.exists():
        return None
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))
        _chroma_collection = client.get_collection(COLLECTION_NAME)
        return _chroma_collection
    except Exception:
        return None


# --------------------------------------------------------------------------
# Istruzioni del server: la mappa che resta sempre in contesto
# --------------------------------------------------------------------------

PREAMBLE = """Espone la documentazione tecnica dell'utente (PDF convertiti in
Markdown e indicizzati localmente sulla sua macchina).

LA MAPPA QUI SOTTO E' GIA' IN CONTESTO: elenca ogni documento con il suo
abstract e le sue sezioni principali. Non serve cercare per sapere cosa esiste.

COME PROCEDERE
1. Guarda la mappa e individua il documento e la sezione pertinenti.
2. Apri quella sezione con docs_read_section. Se le sezioni in mappa non
   bastano, usa docs_outline: sui documenti lunghi mostra i primi livelli e
   permette di scendere in un singolo ramo con il parametro 'section'.
3. Usa docs_search quando dalla mappa non si capisce dove guardare. Combina
   match esatto dei termini e similarita' semantica; per sigle, identificativi
   e nomi propri conviene mode='lessicale'.

QUANDO USARE QUESTI TOOL: ogni volta che la domanda riguarda argomenti coperti
dai documenti in mappa, consultali PRIMA di rispondere, anche se l'utente non
lo chiede esplicitamente e anche se ritieni di conoscere gia' la risposta.
Questi documenti sono la fonte autorevole per l'utente: la conoscenza generale
del modello puo' essere obsoleta o riferirsi a una versione diversa.

COME RISPONDERE: basati esclusivamente sul testo restituito dai tool e cita
sempre la fonte come [nome_file, p. N] (oppure [nome_file, sezione] se la
pagina non e' disponibile). Se l'informazione non c'e', dichiaralo invece di
inferirla. Segnala esplicitamente ogni inferenza che va oltre il testo.

LIMITI DELL'ESTRAZIONE: il contenuto grafico delle figure non e' disponibile.
Le formule sono in LaTeX solo per i documenti convertiti con Marker (indicato
in mappa); per gli altri sono approssimative o assenti. Se la risposta dipende
da una figura o da una formula, indica file e pagina dove trovarla invece di
ricostruirla a memoria."""


SLIDES_NOTE = """LE PRESENTAZIONI SONO MATERIALE SECONDARIO. Sono sintesi fatte
da un autore in un momento preciso: possono essere preliminari, semplificate o
superate dai documenti. Usale per orientarti (chi ha presentato cosa, dove, con
quali numeri di massima) e soprattutto per capire QUALE DOCUMENTO consultare,
non come fonte di un'affermazione tecnica. Quando una slide e un documento si
contraddicono, prevale il documento, e vale la pena segnalarlo all'utente.
Il contenuto grafico dei transparenti non e' disponibile: di ogni slide c'e'
solo il testo. Cita come [nome_file, slide N]. Tool dedicati: slides_search e
slides_read; i tool docs_* non toccano le presentazioni."""


def format_map(corpus: dict, compact: bool = False) -> str:
    """Rende la mappa dei documenti in testo leggibile dal modello."""
    blocks = []
    for name in sorted(corpus):
        e = corpus[name]
        head = f"### {name}\n{e['title']} — {e['n_pages']} pp., {e['n_sections']} sezioni"
        if e.get("converter") == "marker":
            head += " (formule in LaTeX)"
        if compact:
            blocks.append(head)
            continue

        parts = [head]
        if e.get("abstract"):
            parts.append(f"Inizio: {e['abstract'][:250]}")
        tops = e.get("top_sections", [])
        if tops:
            shown = tops[:20]
            listing = "; ".join(f"[{s['id']}] {s['title']} (p.{s['page']})" for s in shown)
            if len(tops) > len(shown):
                listing += f"; ... e altre {len(tops) - len(shown)} (usa docs_outline)"
            parts.append("Sezioni: " + listing)
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def format_decks(decks: dict, budget: int) -> str:
    """Rende le presentazioni entro il budget di caratteri rimasto.

    Le presentazioni sono materiale secondario: ricevono solo lo spazio che i
    documenti non hanno usato. Cosi' aggiungere decine di deck non puo' mai
    far degradare la mappa dei documenti, che e' la parte che serve davvero
    per rispondere.
    """
    lines = []
    for name in sorted(decks):
        e = decks[name]
        lines.append(f"- {name} — {e['title']} ({e['n_pages']} slide)")

    base = "\n".join(lines)
    if len(base) <= budget:
        # Se avanza spazio, si aggiungono i titoli delle slide: aiutano molto
        # a capire se un deck e' pertinente senza doverlo aprire.
        detailed = []
        for name in sorted(decks):
            e = decks[name]
            titles = [t.split(" — ", 1)[-1] for t in e.get("slide_titles", [])]
            titles = [t for t in titles if t and not t.startswith("Slide ")]
            entry = f"- {name} — {e['title']} ({e['n_pages']} slide)"
            if titles:
                entry += "\n  Slide: " + "; ".join(titles[:25])
                if len(titles) > 25:
                    entry += f"; ... e altre {len(titles) - 25}"
            detailed.append(entry)
        full = "\n".join(detailed)
        if len(full) <= budget:
            return full
        return base
    return (base[:max(budget - 80, 0)].rsplit("\n", 1)[0]
            + f"\n(... e altre presentazioni; {len(decks)} in totale.)")


def build_instructions() -> str:
    corpus = load_map()
    if not corpus:
        return (PREAMBLE + "\n\nATTENZIONE: nessun documento indicizzato. "
                "L'utente deve mettere i documenti in ./docs (e le eventuali "
                "presentazioni in ./slides) ed eseguire 'python ingest.py'.")

    docs, decks = split_corpus(corpus)

    # I documenti hanno la precedenza assoluta sul budget di contesto.
    doc_body = format_map(docs) if docs else ""
    if len(doc_body) > MAX_MAP_CHARS:
        doc_body = (format_map(docs, compact=True)
                    + "\n\n(Mappa in forma compatta: troppi documenti per "
                      "elencarne le sezioni. Usa docs_outline su un documento "
                      "per vederne la struttura.)")

    out = [PREAMBLE]
    if docs:
        out.append(f"=== DOCUMENTI ({len(docs)}) ===\n\n{doc_body}")
    if decks:
        remaining = max(MAX_MAP_CHARS - len(doc_body), MIN_DECK_CHARS)
        out.append(SLIDES_NOTE
                   + f"\n\n=== PRESENTAZIONI ({len(decks)}) ===\n\n"
                   + format_decks(decks, remaining))
    return "\n\n".join(out)


mcp = MCPServer(name="docs_mcp", instructions=build_instructions())


# --------------------------------------------------------------------------
# Tool
# --------------------------------------------------------------------------

@mcp.tool(name="docs_outline", title="Struttura di un documento", annotations=READ_ONLY)
async def docs_outline(
    source: Annotated[str, Field(
        description="Nome del file, es. 'CMS-TDR-015.pdf' (estensione opzionale).",
        min_length=1)],
    section: Annotated[Optional[str], Field(
        description=("Limita l'albero a una sola sezione e alle sue "
                     "sottosezioni: id ('s12') o parte del titolo. Su documenti "
                     "molto lunghi conviene partire dalla struttura generale e "
                     "poi scendere qui nel ramo che interessa."))] = None,
    max_level: Annotated[Optional[int], Field(
        description=("Profondita' massima da mostrare (1 = solo capitoli). "
                     "Se omesso viene scelta automaticamente: sui documenti con "
                     "molte sezioni l'albero viene limitato per non saturare il "
                     "contesto, e il messaggio indica come espanderlo."),
        ge=1, le=6)] = None,
) -> str:
    """Struttura delle sezioni di un documento, con pagine e dimensioni.

    Da usare quando le sezioni presenti nella mappa non bastano a capire dove
    si trova un argomento. Su un TDR l'albero completo puo' contenere centinaia
    di voci: per questo la profondita' viene limitata in automatico, e si scende
    nel dettaglio ramo per ramo con il parametro 'section'.

    Returns:
        str: una riga per sezione, indentata per livello, nel formato
             '[id] Titolo (p. N-M, X caratteri)'. La dimensione indica quanto
             testo restituira' docs_read_section su quella sezione.
    """
    resolved = resolve_source(source, kind="document")
    if not resolved:
        return unknown_source_msg(source, kind="document")
    data = load_doc(resolved)
    if not data:
        return f"Indice mancante per '{resolved}'. Rilancia 'python ingest.py'."

    sections = data["sections"]
    if not sections:
        return (f"'{resolved}' non ha sezioni riconosciute ({data['n_pages']} pagine). "
                "Usa docs_read_pages o docs_search su questo documento.")

    scope_label = ""
    root = None
    if section:
        root = find_section(sections, section)
        if root is None:
            return (f"Sezione '{section}' non trovata in {resolved}. "
                    "Chiama docs_outline senza 'section' per vedere la struttura.")
        if isinstance(root, list):
            elenco = "; ".join(f"[{s['id']}] {s['path']}" for s in root[:12])
            return f"'{section}' e' ambiguo in {resolved}. Candidati: {elenco}."
        sections = [s for s in sections
                    if s["line_start"] >= root["line_start"]
                    and s["line_end"] <= root["line_end"]]
        scope_label = f" — ramo '{root['path']}'"

    # Profondita' automatica: su alberi grandi si mostra meno, per non
    # spendere migliaia di token in un elenco che serve solo a orientarsi.
    auto = max_level is None
    if auto:
        base = min(s["level"] for s in sections)
        max_level = base if len(sections) > 200 else base + 1 if len(sections) > 60 else 6

    shown = [s for s in sections if s["level"] <= max_level]
    hidden = len(sections) - len(shown)

    lines = [f"{resolved}{scope_label} — {data['n_pages']} pagine, "
             f"{len(sections)} sezioni nel perimetro\n"]
    base_level = min(s["level"] for s in shown) if shown else 1
    for s in shown:
        indent = "  " * (s["level"] - base_level)
        pages = (f"p. {s['page_start']}" if s["page_start"] == s["page_end"]
                 else f"p. {s['page_start']}-{s['page_end']}")
        n_sub = sum(1 for c in sections
                    if c["line_start"] > s["line_start"]
                    and c["line_end"] <= s["line_end"])
        suffix = f", {n_sub} sottosezioni" if n_sub and s["level"] == max_level else ""
        lines.append(f"{indent}[{s['id']}] {s['title']} "
                     f"({pages}, {s['n_chars']} caratteri{suffix})")

    if hidden:
        lines.append(
            f"\n[{hidden} sottosezioni non mostrate oltre il livello {max_level}. "
            "Per vederle: docs_outline con section='<id del ramo>', "
            "oppure max_level piu' alto.]")
    return "\n".join(lines)


@mcp.tool(name="docs_read_section", title="Leggi una sezione", annotations=READ_ONLY)
async def docs_read_section(
    source: Annotated[str, Field(
        description="Nome del file, es. 'CMS-TDR-015.pdf'.", min_length=1)],
    section: Annotated[str, Field(
        description=("Identificativo della sezione ('s12', come mostrato in mappa "
                     "e in docs_outline) oppure parte del titolo ('Timing "
                     "performance'). La corrispondenza sul titolo ignora "
                     "maiuscole e accetta sottostringhe."),
        min_length=1)],
) -> str:
    """Legge il testo completo di una sezione, sottosezioni incluse.

    E' il tool principale: la mappa dice dove guardare, questo apre il punto
    giusto. Se la sezione supera la soglia di dimensione viene restituita
    troncata insieme all'elenco delle sottosezioni, cosi' da poter scendere di
    livello invece di caricare decine di pagine.

    Returns:
        str: intestazione con file, percorso della sezione e pagine, seguita
             dal testo Markdown della sezione.
    """
    resolved = resolve_source(source, kind="document")
    if not resolved:
        return unknown_source_msg(source, kind="document")
    data = load_doc(resolved)
    if not data:
        return f"Indice mancante per '{resolved}'. Rilancia 'python ingest.py'."

    sections = data["sections"]
    match = find_section(sections, section)
    if match is None:
        return (f"Sezione '{section}' non trovata in {resolved}. "
                "Usa docs_outline per vedere la struttura.")
    if isinstance(match, list):
        elenco = "; ".join(f"[{s['id']}] {s['path']}" for s in match[:12])
        return (f"'{section}' e' ambiguo in {resolved}. Candidati: {elenco}. "
                "Richiama il tool con l'id preciso.")

    lines = data["lines"]
    text = "\n".join(lines[match["line_start"]:match["line_end"]]).strip()
    pages = (f"p. {match['page_start']}" if match["page_start"] == match["page_end"]
             else f"p. {match['page_start']}-{match['page_end']}")
    header = f"# {resolved} — {match['path']} ({pages})\n"

    if len(text) <= MAX_SECTION_CHARS:
        return header + "\n" + text

    children = [s for s in sections
                if s["line_start"] > match["line_start"]
                and s["line_end"] <= match["line_end"]
                and s["level"] == match["level"] + 1]
    body_end = min((c["line_start"] for c in children), default=match["line_end"])
    own = "\n".join(lines[match["line_start"]:body_end]).strip()

    note = [header,
            f"\n[Sezione lunga: {len(text)} caratteri, troncata. "
            f"Sotto c'e' solo il testo introduttivo della sezione.]"]
    if own:
        note.append("\n" + own[:MAX_SECTION_CHARS])
    if children:
        elenco = "\n".join(
            f"  [{c['id']}] {c['title']} (p. {c['page_start']}, {c['n_chars']} caratteri)"
            for c in children)
        note.append(f"\nSottosezioni da leggere separatamente:\n{elenco}")
    else:
        note.append("\nNessuna sottosezione: usa docs_read_pages per leggere "
                    f"le pagine {match['page_start']}-{match['page_end']} a blocchi.")
    return "\n".join(note)


@mcp.tool(name="docs_read_pages", title="Leggi pagine", annotations=READ_ONLY)
async def docs_read_pages(
    source: Annotated[str, Field(description="Nome del file.", min_length=1)],
    start_page: Annotated[int, Field(description="Prima pagina (1-indexed).", ge=1)],
    end_page: Annotated[Optional[int], Field(
        description="Ultima pagina inclusa. Se omessa, legge solo start_page.",
        ge=1)] = None,
) -> str:
    """Legge un intervallo di pagine, utile quando la struttura non basta.

    Serve soprattutto per i documenti senza titoli riconoscibili e per scendere
    dentro sezioni molto lunghe. Massimo 25 pagine per chiamata.

    Returns:
        str: testo Markdown delle pagine richieste, con intestazione di pagina.
    """
    resolved = resolve_source(source, kind="document")
    if not resolved:
        return unknown_source_msg(source, kind="document")
    data = load_doc(resolved)
    if not data:
        return f"Indice mancante per '{resolved}'. Rilancia 'python ingest.py'."

    end = end_page or start_page
    if end < start_page:
        return "Errore: end_page deve essere >= start_page."
    if end - start_page + 1 > MAX_PAGES_PER_READ:
        return (f"Errore: massimo {MAX_PAGES_PER_READ} pagine per chiamata "
                f"(richieste {end - start_page + 1}). Spezza in piu' chiamate.")

    # Ricostruisce la mappa riga -> pagina dal markdown salvato.
    md_path = BASE_DIR / "md" / f"{Path(resolved).stem}.md"
    if not md_path.exists():
        return f"Markdown mancante per '{resolved}'. Rilancia 'python ingest.py'."

    out, current, keeping = [], 0, False
    for raw in md_path.read_text().splitlines():
        m = re.match(r"^<!--page:(\d+)-->$", raw.strip())
        if m:
            current = int(m.group(1))
            keeping = start_page <= current <= end
            if keeping:
                out.append(f"\n## {resolved} — pagina {current}\n")
            continue
        if keeping:
            out.append(raw)

    text = "\n".join(out).strip()
    if not text:
        return (f"Nessun contenuto per '{resolved}' nelle pagine {start_page}-{end} "
                f"(il documento ha {data['n_pages']} pagine).")
    return text


def fts_query_string(query: str) -> Optional[str]:
    """Trasforma una query libera in un'espressione FTS5 sicura.

    I caratteri speciali di FTS5 farebbero fallire la query, e sigle come
    'LQ(bmu)' o 'CMS-TDR-015' ne sono piene. Si estraggono i termini, si
    citano singolarmente e si uniscono in OR, cosi' che il ranking BM25
    premi i documenti che ne contengono di piu'.
    """
    terms = re.findall(r"[\w\u00C0-\u024F]+", query)
    terms = [t for t in terms if len(t) > 1]
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)


def search_bm25(query: str, n: int, source: Optional[str],
                kind: str = "document") -> list[dict]:
    """Ricerca lessicale BM25. Ritorna una lista ordinata per rilevanza."""
    if not FTS_PATH.exists():
        return []
    expr = fts_query_string(query)
    if not expr:
        return []
    import sqlite3
    con = sqlite3.connect(f"file:{FTS_PATH}?mode=ro", uri=True)
    try:
        sql = ("SELECT source, section_id, section, page, text, bm25(chunks) AS score "
               "FROM chunks WHERE chunks MATCH ? AND kind = ?")
        params: list = [expr, kind]
        if source:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY score LIMIT ?"
        params.append(n)
        rows = con.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()

    return [{"source": r[0], "section_id": r[1], "section": r[2],
             "page": r[3], "text": r[4], "score": r[5]} for r in rows]


def search_vectors(query: str, n: int, source: Optional[str],
                   kind: str = "document") -> list[dict]:
    """Ricerca semantica. Lista vuota se l'indice non e' disponibile."""
    collection = get_collection()
    if collection is None:
        return []
    where: dict = {"kind": kind}
    if source:
        where = {"$and": [{"kind": kind}, {"source": source}]}
    try:
        res = collection.query(query_texts=[query], n_results=n, where=where)
    except Exception:
        return []
    out = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                                res["distances"][0]):
        out.append({"source": meta["source"], "section_id": meta.get("section_id", "?"),
                    "section": meta.get("section", "?"), "page": str(meta.get("page", "?")),
                    "text": text, "score": dist})
    return out


def fuse(lexical: list[dict], semantic: list[dict], n: int) -> list[dict]:
    """Reciprocal Rank Fusion dei due ranking.

    Si fondono le POSIZIONI, non i punteggi: BM25 e distanza vettoriale sono
    su scale incomparabili. La costante 60 e' il valore usuale: attenua il peso
    delle prime posizioni quel tanto che basta perche' un risultato presente in
    entrambe le liste superi uno che domina una sola lista.
    """
    K = 60
    merged: dict[tuple, dict] = {}
    for results, label in ((lexical, "BM25"), (semantic, "semantica")):
        for rank, item in enumerate(results, start=1):
            key = (item["source"], item["section_id"], item["text"][:80])
            entry = merged.setdefault(key, {**item, "rrf": 0.0, "found_by": []})
            entry["rrf"] += 1.0 / (K + rank)
            entry["found_by"].append(f"{label} #{rank}")
    ordered = sorted(merged.values(), key=lambda e: -e["rrf"])
    return ordered[:n]


@mcp.tool(name="docs_search", title="Ricerca nei documenti", annotations=READ_ONLY)
async def docs_search(
    query: Annotated[str, Field(
        description="Frase o termini da cercare.", min_length=1, max_length=1000)],
    n_results: Annotated[int, Field(
        description="Numero di estratti da restituire.", ge=1, le=20)] = 6,
    source: Annotated[Optional[str], Field(
        description="Nome del file per restringere a un solo documento.")] = None,
    mode: Annotated[str, Field(
        description=("'ibrida' (default) combina match esatto dei termini e "
                     "similarita' semantica; 'lessicale' usa solo il match "
                     "esatto, da preferire per sigle, identificativi e nomi "
                     "propri; 'semantica' usa solo gli embedding, utile per "
                     "concetti espressi con parole diverse dal testo."))] = "ibrida",
) -> str:
    """Cerca passaggi nei documenti, come alternativa alla navigazione per struttura.

    Preferisci quando possibile il percorso mappa -> docs_outline ->
    docs_read_section, che e' piu' affidabile. Questa ricerca serve quando non
    si sa in quale documento o sezione guardare.

    Combina due motori: BM25 (match esatto dei termini, forte su sigle e
    identificativi come 'CMS-TDR-015' o 'PbWO4') e ricerca vettoriale (forte
    sui concetti riformulati). I due ranking vengono fusi; un risultato trovato
    da entrambi sale in alto.

    Cerca SOLO nei documenti: le presentazioni hanno slides_search.

    Returns:
        str: estratti con documento, sezione di provenienza, pagina e quale
             motore li ha trovati. Per il contesto completo conviene poi aprire
             la sezione con docs_read_section.
    """
    resolved = resolve_source(source) if source else None
    if source and not resolved:
        return unknown_source_msg(source)

    mode = mode.strip().lower()
    if mode not in {"ibrida", "lessicale", "semantica"}:
        return ("Errore: mode deve essere 'ibrida', 'lessicale' o 'semantica'.")

    pool = max(n_results * 3, 12)
    lexical = [] if mode == "semantica" else search_bm25(query, pool, resolved, "document")
    semantic = [] if mode == "lessicale" else search_vectors(query, pool, resolved, "document")

    if not lexical and not semantic:
        hints = []
        if not FTS_PATH.exists():
            hints.append("l'indice BM25 non esiste (rilancia 'python ingest.py')")
        if get_collection() is None:
            hints.append("l'indice vettoriale non e' disponibile")
        extra = f" Nota: {'; '.join(hints)}." if hints else ""
        return ("Nessun risultato per questa query." + extra +
                " Prova con la mappa e docs_outline per individuare la sezione.")

    results = fuse(lexical, semantic, n_results)

    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"### Estratto {i} — {r['source']}, p. {r['page']}\n"
            f"Sezione: {r['section']} [{r['section_id']}]\n"
            f"Trovato da: {', '.join(r['found_by'])}\n\n{r['text']}")

    footer = "\n\n(Per il contesto completo apri la sezione con docs_read_section.)"
    if mode == "ibrida" and not semantic:
        footer += ("\nNota: l'indice vettoriale non era disponibile, "
                   "questi risultati vengono dal solo match lessicale.")
    elif mode == "ibrida" and not lexical:
        footer += ("\nNota: nessun match lessicale esatto; "
                   "questi risultati vengono dalla sola ricerca semantica.")
    return "\n\n---\n\n".join(blocks) + footer


@mcp.tool(name="slides_search", title="Cerca nelle presentazioni", annotations=READ_ONLY)
async def slides_search(
    query: Annotated[str, Field(
        description="Frase o termini da cercare nelle presentazioni.",
        min_length=1, max_length=1000)],
    n_results: Annotated[int, Field(
        description="Numero di slide da restituire.", ge=1, le=20)] = 6,
    source: Annotated[Optional[str], Field(
        description="Nome del file per restringere a una sola presentazione.")] = None,
    mode: Annotated[str, Field(
        description="'ibrida' (default), 'lessicale' o 'semantica'.")] = "ibrida",
) -> str:
    """Cerca nel testo delle presentazioni, separatamente dai documenti.

    Utile per domande su chi ha presentato cosa e dove, per ritrovare un numero
    mostrato a una conferenza, o per capire quale documento approfondisce un
    certo argomento. Il testo delle slide e' telegrafico e il contenuto grafico
    non e' disponibile, quindi i risultati vanno trattati come indizi: per
    un'affermazione tecnica, risali al documento con docs_search.

    Returns:
        str: slide pertinenti con file, numero di slide e titolo.
    """
    resolved = resolve_source(source, kind="slides") if source else None
    if source and not resolved:
        return unknown_source_msg(source, kind="slides")

    mode = mode.strip().lower()
    if mode not in {"ibrida", "lessicale", "semantica"}:
        return "Errore: mode deve essere 'ibrida', 'lessicale' o 'semantica'."

    _, decks = split_corpus(load_map())
    if not decks:
        return ("Nessuna presentazione indicizzata. L'utente puo' metterle in "
                "./slides ed eseguire 'python ingest.py'.")

    pool = max(n_results * 3, 12)
    lexical = [] if mode == "semantica" else search_bm25(query, pool, resolved, "slides")
    semantic = [] if mode == "lessicale" else search_vectors(query, pool, resolved, "slides")

    if not lexical and not semantic:
        return ("Nessun risultato nelle presentazioni. "
                "Prova docs_search sui documenti.")

    blocks = []
    for i, r in enumerate(fuse(lexical, semantic, n_results), start=1):
        blocks.append(
            f"### {r['source']} — slide {r['page']}\n"
            f"{r['section']}\n"
            f"Trovato da: {', '.join(r['found_by'])}\n\n{r['text']}")
    return ("\n\n---\n\n".join(blocks)
            + "\n\n(Materiale secondario: per le affermazioni tecniche verifica "
              "sui documenti con docs_search.)")


@mcp.tool(name="slides_read", title="Leggi slide", annotations=READ_ONLY)
async def slides_read(
    source: Annotated[str, Field(
        description="Nome della presentazione, es. 'ICHEP2026_ECAL.pdf'.",
        min_length=1)],
    from_slide: Annotated[Optional[int], Field(
        description="Prima slide da leggere (1-indexed). Se omessa parte dalla 1.",
        ge=1)] = None,
    to_slide: Annotated[Optional[int], Field(
        description="Ultima slide inclusa. Se omessa legge solo from_slide, "
                    "o l'intera presentazione se anche from_slide manca.",
        ge=1)] = None,
) -> str:
    """Legge il testo di un intervallo di slide, o dell'intera presentazione.

    Le presentazioni sono in genere corte, quindi leggerle per intero e'
    spesso la cosa piu' sensata: senza argomenti restituisce tutto il deck, se
    sta nel limite. Di ogni slide c'e' il solo testo, non le figure.

    Returns:
        str: testo delle slide, ciascuna con numero e titolo.
    """
    resolved = resolve_source(source, kind="slides")
    if not resolved:
        return unknown_source_msg(source, kind="slides")
    data = load_doc(resolved)
    if not data:
        return f"Indice mancante per '{resolved}'. Rilancia 'python ingest.py'."

    sections = data["sections"]
    n = len(sections)
    start = from_slide or 1
    end = to_slide or (start if from_slide else n)
    if end < start:
        return "Errore: to_slide deve essere >= from_slide."
    if start > n:
        return f"'{resolved}' ha {n} slide: la {start} non esiste."
    end = min(end, n)

    chosen = sections[start - 1:end]
    total = sum(s["n_chars"] for s in chosen)
    if total > MAX_SECTION_CHARS:
        elenco = "\n".join(f"  slide {s['page_start']}: {s['title']}" for s in sections)
        return (f"{resolved}: {end - start + 1} slide sono troppe da leggere "
                f"in una volta ({total} caratteri). Titoli delle slide:\n{elenco}\n"
                "Richiama slides_read su un intervallo piu' stretto.")

    lines = data["lines"]
    out = [f"# {resolved} — slide {start}-{end} di {n}\n"]
    for sec in chosen:
        out.append("\n".join(lines[sec["line_start"]:sec["line_end"]]).strip())
    return "\n\n".join(out)


@mcp.tool(name="docs_list_figures", title="Elenca le figure", annotations=READ_ONLY)
async def docs_list_figures(
    source: Annotated[str, Field(
        description="Nome del file (documento o presentazione).", min_length=1)],
    page: Annotated[Optional[int], Field(
        description="Limita a una pagina o slide.", ge=1)] = None,
) -> str:
    """Elenca le figure rilevate, con didascalia e testo interno al grafico.

    Il testo interno sono le etichette degli assi, la legenda e le annotazioni:
    spesso contiene i numeri del risultato. La curva in se' non e' leggibile da
    qui: per quella si usa docs_read_figure, che restituisce l'immagine.

    Returns:
        str: una voce per figura con id, pagina, didascalia e testo estratto.
    """
    resolved = resolve_source(source)
    if not resolved:
        return unknown_source_msg(source)
    data = load_doc(resolved)
    if not data:
        return f"Indice mancante per '{resolved}'. Rilancia 'python ingest.py'."

    figures = data.get("figures", [])
    if page is not None:
        figures = [f for f in figures if f["page"] == page]
    if not figures:
        where = f" a p. {page}" if page is not None else ""
        return (f"Nessuna figura rilevata in '{resolved}'{where}. "
                "Il rilevamento cerca grafica vettoriale e immagini "
                "incorporate; una figura molto semplice puo' sfuggirgli.")

    out = [f"{resolved} — {len(figures)} figure\n"]
    for f in figures:
        out.append(f"[{f['id']}] p. {f['page']}")
        if f.get("caption"):
            out.append(f"  Didascalia: {f['caption']}")
        if f.get("text"):
            out.append(f"  Testo nel grafico: {f['text'][:400]}")
        out.append("")
    out.append("Per vedere una figura: docs_read_figure con il suo id.")
    return "\n".join(out)


@mcp.tool(name="docs_read_figure", title="Guarda una figura", annotations=READ_ONLY)
async def docs_read_figure(
    source: Annotated[str, Field(
        description="Nome del file (documento o presentazione).", min_length=1)],
    figure: Annotated[Optional[str], Field(
        description="Id della figura ('f3'), da docs_list_figures. In "
                    "alternativa usa 'page' per l'intera pagina.")] = None,
    page: Annotated[Optional[int], Field(
        description="Rende l'intera pagina o slide come immagine. Utile "
                    "quando il rilevamento automatico non ha isolato la "
                    "figura, o per vederla insieme al testo attorno.",
        ge=1)] = None,
) -> Any:
    """Restituisce l'immagine di una figura, da guardare direttamente.

    E' l'unico modo per leggere il CONTENUTO di un grafico: andamento delle
    curve, punti sperimentali, bande di incertezza, valori sugli assi. Il testo
    indicizzato contiene solo didascalia ed etichette.

    L'immagine viene generata al momento dal PDF originale, che deve quindi
    restare in ./docs o ./slides.

    Quando riporti quello che vedi in un grafico, dillo esplicitamente
    ("dalla figura si legge...") e distinguilo dai valori scritti nel testo:
    una lettura a occhio da un plot e' un'approssimazione, non un dato citabile
    con la stessa precisione.

    Returns:
        Image: l'immagine della figura o della pagina.
    """
    resolved = resolve_source(source)
    if not resolved:
        return unknown_source_msg(source)
    data = load_doc(resolved)
    if not data:
        return f"Indice mancante per '{resolved}'. Rilancia 'python ingest.py'."

    pdf_path = find_pdf(resolved)
    if pdf_path is None:
        return (f"PDF originale '{resolved}' non trovato in ./docs o ./slides. "
                "Le figure vengono generate dal file originale, che non va "
                "spostato dopo l'indicizzazione.")

    clip = None
    target_page = page
    if figure:
        match = next((f for f in data.get("figures", [])
                      if f["id"].lower() == figure.strip().lower()), None)
        if match is None:
            return (f"Figura '{figure}' non trovata in {resolved}. "
                    "Usa docs_list_figures per gli id disponibili.")
        target_page = match["page"]
        clip = match["bbox"]
    if target_page is None:
        return "Errore: indica 'figure' oppure 'page'."

    try:
        import pymupdf
        with pymupdf.open(str(pdf_path)) as doc:
            if not 1 <= target_page <= doc.page_count:
                return (f"'{resolved}' ha {doc.page_count} pagine: "
                        f"la {target_page} non esiste.")
            pg = doc[target_page - 1]
            rect = pymupdf.Rect(clip) if clip else None
            if rect is not None:
                # Un margine attorno alla figura aiuta a includere assi e
                # unita' di misura che cadono appena fuori dal riquadro.
                rect = pymupdf.Rect(rect.x0 - 8, rect.y0 - 8,
                                    rect.x1 + 8, rect.y1 + 8) & pg.rect
            pix = pg.get_pixmap(dpi=FIGURE_DPI, clip=rect)
            png = pix.tobytes("png")
    except Exception as exc:
        return f"Errore nel rendering: {type(exc).__name__}: {exc}"

    return Image(data=png, format="png")


if __name__ == "__main__":
    mcp.run()
