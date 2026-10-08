# rag-docs

Server MCP che dà a Claude accesso alla documentazione tecnica locale (PDF di paper, TDR, schematiche, presentazioni). I PDF vengono convertiti in Markdown, indicizzati per struttura (sezioni e pagine) e per similarità, ed esposti a Claude Desktop e Claude Code tramite tool.

La mappa dell'intero corpus (titolo, abstract e sezioni di ogni documento) resta nel contesto di Claude: così naviga per struttura invece di fare ricerche alla cieca.

## Struttura

| Percorso | Contenuto |
|---|---|
| `docs/` | PDF dei documenti, anche in sottocartelle (es. `docs/schemas_081026/`) |
| `slides/` | PDF delle presentazioni (solo testo) |
| `ingest.py` | Conversione e indicizzazione, incrementale |
| `server.py` | Server MCP (stdio) |
| `setup_mcp.py` | Registra il server in Claude Desktop e Claude Code |
| `md/`, `index/`, `db/` | Output generato (ignorato da git) |

## Installazione

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Aggiungere documenti

1. Copia i PDF in `docs/` (anche in sottocartelle) o in `slides/`. I nomi dei file devono essere unici.
2. Lancia l'indicizzazione:

```bash
.venv/bin/python ingest.py --fast
```

L'indicizzazione è incrementale: vengono elaborati solo i file nuovi o modificati, quelli rimossi vengono ripuliti.

| Opzione | Effetto |
|---|---|
| `--fast` | usa pymupdf4llm: pochi secondi a documento, ma senza formule in LaTeX |
| *(nessuna)* | usa Marker: formule in LaTeX, lento, scarica alcuni GB di modelli al primo avvio |
| `--rebuild` | azzera tutto e riconverte |
| `--no-vectors` | salta l'indice vettoriale; resta la navigazione per struttura |
| `--refresh-figures` | ricalcola le figure senza riconvertire il testo |

3. Riavvia il server MCP (o la sessione) per aggiornare la mappa in contesto.

### Schematiche e immagini

- PDF **scansionati** o **PNG**: serve prima l'OCR. Converti i PNG con `img2pdf` e usa `ocrmypdf --skip-text` sui PDF (non `--force-ocr` se hanno già testo: lo sostituirebbe).
- PDF **senza alcun testo**: vengono scartati da `ingest.py`.
- Il contenuto grafico non è leggibile come testo: con `docs_read_figure` Claude vede la pagina o la figura renderizzata dal PDF originale.

## Registrazione in Claude

```bash
.venv/bin/python setup_mcp.py
```

Registra il server `docs` in modo globale in Claude Desktop (`claude_desktop_config.json`) e in Claude Code (`~/.claude.json`).

## Tool esposti

| Tool | Uso |
|---|---|
| `docs_outline` | albero delle sezioni di un documento |
| `docs_read_section` | legge una sezione |
| `docs_read_pages` | legge un intervallo di pagine |
| `docs_search` | ricerca lessicale e semantica |
| `docs_list_figures` | elenca le figure con didascalia |
| `docs_read_figure` | mostra una figura o una pagina come immagine |
| `slides_search`, `slides_read` | ricerca e lettura delle presentazioni |

Le risposte citano la fonte come `[nome_file, p. N]`.
