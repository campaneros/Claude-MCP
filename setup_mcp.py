#!/usr/bin/env python3
"""
Registra il server MCP "docs" (rag-docs) in modo GLOBALE:
  - Claude Desktop  -> ~/Library/Application Support/Claude/claude_desktop_config.json
  - Claude Code     -> ~/.claude.json  (scope utente: vale in qualsiasi cartella)

Fa il merge senza distruggere le configurazioni esistenti e crea un backup .bak
di ogni file toccato. Idempotente: rilanciarlo non duplica nulla.

Uso:
    ~/Documents/rag-docs/.venv/bin/python ~/Documents/rag-docs/setup_mcp.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
PYTHON = PROJECT / ".venv" / "bin" / "python"
SERVER = PROJECT / "server.py"
NAME = "docs"

ENTRY = {"command": str(PYTHON), "args": [str(SERVER)]}

TARGETS = [
    Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
    Path.home() / ".claude.json",
]


def smoke_test() -> bool:
    """Avvia il server per ~6s: se muore subito, c'e' un errore da mostrare."""
    print("-> smoke test di server.py ...")
    p = subprocess.Popen(
        [str(PYTHON), str(SERVER)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(PROJECT), text=True,
    )
    try:
        _, err = p.communicate(timeout=6)
    except subprocess.TimeoutExpired:
        p.kill()
        print("   OK: il server resta in ascolto sullo stdio.")
        return True
    print(f"   ERRORE: il server e' uscito con codice {p.returncode}")
    if err.strip():
        print("   --- stderr ---")
        for line in err.strip().splitlines()[-25:]:
            print("   " + line)
    return False


def patch(path: Path) -> None:
    data = {}
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        try:
            data = json.loads(path.read_text() or "{}")
        except json.JSONDecodeError as e:
            print(f"!! {path} non e' JSON valido ({e}); lo salto.")
            return
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    servers = data.setdefault("mcpServers", {})
    already = servers.get(NAME) == ENTRY
    servers[NAME] = ENTRY
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"{'=  gia aggiornato' if already else '-> scritto'}: {path}")


def main() -> int:
    for f in (PYTHON, SERVER):
        if not f.exists():
            print(f"!! manca {f}")
            return 1

    ok = smoke_test()
    for t in TARGETS:
        patch(t)

    print("\nFatto. Riavvia Claude Desktop con Cmd+Q e riaprilo.")
    print("Per Claude Code: `claude mcp list` da qualsiasi cartella.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
