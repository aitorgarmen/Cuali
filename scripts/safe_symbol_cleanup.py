#!/usr/bin/env python3
"""
safe_symbol_cleanup.py

Limpia SOLO símbolos raros de forma conservadora, sin re-decodificar
archivos ni aplicar heurísticas agresivas. Cambia una lista blanca de
caracteres/secuencias problemáticas a alternativas legibles.

Qué corrige:
  - Elimina BOM (U+FEFF) al inicio o en cualquier punto del archivo.
  - Corrige mojibake frecuentes: °C/µs, comillas/guiones tipo "'/-".
  - Sustituye NBSP mojibake (" ") por espacio normal.

Seguridad:
  - No usa ftfy ni recodifica bytes.
  - Solo sustituye patrones explícitos.
  - Mantiene el resto del contenido intacto.

Uso:
  - Vista previa:  python scripts/safe_symbol_cleanup.py --dry
  - Aplicar:       python scripts/safe_symbol_cleanup.py
  - Limitar ruta:  python scripts/safe_symbol_cleanup.py --root tabs

Extensiones por defecto: .py .sql .md .txt
Excluye: venv, dist, build, .git, __pycache__
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Iterable


# Reemplazos de secuencia (orden importa)
SEQ_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    # BOM (al inicio o disperso): eliminar
    ("\ufeff", ""),
    # Grados Celsius
    ("°C", "°C"),
    ("(°C)", "(°C)"),
    ("°C", "°C"),
    ("(°C)", "(°C)"),
    ("°C", "°C"),
    ("(°C)", "(°C)"),
    # Microsegundos
    ("µs", "µs"),
    ("(µs)", "(µs)"),
    ("µs", "µs"),
    ("(µs)", "(µs)"),
    ("µs", "µs"),
    ("(µs)", "(µs)"),
    # Mojibake de comillas/guiones comunes
    ("'", "'"),
    ("'", "'"),
    (""", '"'),
    (""", '"'),
    ("-", "-"),
    ("-", "-"),
    ("*", "*"),
    ("...", "..."),
    # NBSP mojibake
    (" ", " "),
)

# Reemplazo de caracteres sueltos
CHAR_MAP = {
    # comillas / guiones tipográficos
    "\u2018": "'", "\u2019": "'",
    "\u201C": '"', "\u201D": '"',
    "\u00AB": '"', "\u00BB": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2026": "...",
    # espacios raros
    "\u00A0": " ", "\u202F": " ", "\u2007": " ", "\u2009": " ",
    # replacement char: eliminar
    "\uFFFD": "",
}


def iter_files(root: Path, *, include_ext: Sequence[str], exclude_dirs: Sequence[str]) -> Iterable[Path]:
    inc = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in include_ext}
    exd = set(exclude_dirs)
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if any(part in exd for part in path.parts):
            continue
        if path.suffix.lower() in inc:
            yield path


def clean_text_line(line: str) -> str:
    # primero secuencias
    for a, b in SEQ_REPLACEMENTS:
        if a in line:
            line = line.replace(a, b)
    # luego chars individuales
    if any(ord(ch) > 127 for ch in line):
        for a, b in CHAR_MAP.items():
            if a in line:
                line = line.replace(a, b)
    return line


def process_path(p: Path) -> bool:
    try:
        text = p.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return False
    # Strip BOM al inicio si quedara tras read_text
    if text.startswith('\ufeff'):
        text = text.lstrip('\ufeff')
    new_text = ''.join(clean_text_line(ln) for ln in text.splitlines(keepends=True))
    if new_text != text:
        p.write_text(new_text, encoding='utf-8', newline='\n')
        return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Limpieza conservadora de símbolos raros")
    ap.add_argument('--root', default='.')
    ap.add_argument('--ext', nargs='*', default=['.py', '.sql', '.md', '.txt'])
    ap.add_argument('--exclude', nargs='*', default=['.venv', 'venv', 'dist', 'build', '.git', '__pycache__'])
    ap.add_argument('--dry', action='store_true', help='Vista previa: no escribe cambios')
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    changed = 0
    scanned = 0
    for p in iter_files(root, include_ext=args.ext, exclude_dirs=args.exclude):
        scanned += 1
        try:
            text = p.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        new_text = ''.join(clean_text_line(ln) for ln in text.splitlines(keepends=True))
        if new_text != text:
            if args.dry:
                print(f"WOULD-CHANGE {p}")
            else:
                p.write_text(new_text, encoding='utf-8', newline='\n')
                print(f"CHANGE {p}")
            changed += 1

    print(f"\nEscaneados: {scanned}")
    if args.dry:
        print(f"Cambiarían: {changed}")
    else:
        print(f"Modificados: {changed}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
