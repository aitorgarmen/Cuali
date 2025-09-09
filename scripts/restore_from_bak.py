#!/usr/bin/env python3
"""
restore_from_bak.py

Restaura archivos desde copias .bak creadas por scripts previos.

Uso:
  - Vista previa (qué restauraría):
      python scripts/restore_from_bak.py --dry
  - Aplicar restauración (sobrescribe originales con el .bak):
      python scripts/restore_from_bak.py --apply

Por defecto recorre el repo, ignorando venv/dist/build/.git/__pycache__.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


EXCLUDE_DIRS = {"venv", "dist", "build", ".git", "__pycache__"}


def iter_baks(root: Path):
    for p in root.rglob("*.bak"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser(description="Restaura archivos desde .bak")
    ap.add_argument("--root", default=".")
    ap.add_argument("--apply", action="store_true", help="Aplicar restauración")
    ap.add_argument("--keep-bak", action="store_true", help="No borrar .bak tras restaurar")
    ap.add_argument("--dry", action="store_true", help="Vista previa (no escribe)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    count = 0
    for bak in iter_baks(root):
        dest = bak.with_suffix("")
        print(f"RESTORE {'WOULD ' if args.dry and not args.apply else ''}{bak} -> {dest}")
        if args.apply and not args.dry:
            shutil.copy2(bak, dest)
            if not args.keep_bak:
                try:
                    bak.unlink()
                except Exception:
                    pass
            count += 1

    if args.apply and not args.dry:
        print(f"Restaurados: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

