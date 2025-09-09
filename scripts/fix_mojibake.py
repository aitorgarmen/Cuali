#!/usr/bin/env python3
"""
fix_mojibake.py

Limpia texto con mojibake y caracteres raros en archivos del repo.

- Repara lo que puede con ftfy (si está instalado).
- Normaliza comillas y guiones tipográficos a equivalentes simples.
- Corrige algunos símbolos raros observados (p. ej., ǧ→ú, ǭ→á, Ǹ→é).
- Opcionalmente translitera a ASCII (sin tildes) para máxima legibilidad.

Uso típico:
  - Vista previa (sin escribir cambios):
      python scripts/fix_mojibake.py --dry
  - Aplicar cambios con copia de seguridad:
      python scripts/fix_mojibake.py
  - Forzar transliteración a ASCII:
      python scripts/fix_mojibake.py --mode ascii

Por defecto ignora carpetas: venv, dist, build, .git, __pycache__.
Extensiones por defecto: .py .sql .md .txt
"""

from __future__ import annotations

import argparse
import shutil
import sys
import unicodedata as ud
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


def _maybe_fix_with_ftfy(text: str) -> str:
    try:
        # ftfy is optional; only used if available
        from ftfy import fix_text  # type: ignore

        # Use NFC normalization; ftfy can fix mojibake like 'cafÃ©' -> 'café'
        return fix_text(text, normalization="NFC")
    except Exception:
        return text


# Basic replacements that improve readability without changing meaning
PUNCT_MAP = {
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201C": '"',  # left double quote
    "\u201D": '"',  # right double quote
    "\u00AB": '"',  # "
    "\u00BB": '"',  # "
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2212": "-",  # minus sign
    "\u2026": "...",  # ellipsis
    "\u00A0": " ",  # NBSP → space
    "\u202F": " ",  # NNBSP
    "\u2007": " ",  # figure space
    "\u2009": " ",  # thin space
}

# Weird glyphs seen in repo due to mixed encodings
WEIRD_MAP_KEEP = {
    "ǧ": "ú",
    "ǭ": "á",
    "Ǹ": "é",
    "ǹ": "ì",  # rarely seen; best effort
    "": "",   # U+FFFD replacement char → drop
}


def _normalize_punct(text: str) -> str:
    for k, v in PUNCT_MAP.items():
        text = text.replace(k, v)
    return text


def _apply_weird_map(text: str) -> str:
    for k, v in WEIRD_MAP_KEEP.items():
        text = text.replace(k, v)
    # Handle some multi-char artifacts commonly produced by bad decoders
    text = text.replace("", '"')   # broken double-quote
    text = text.replace("?", '?')    # broken question mark
    return text


def _strip_controls(text: str) -> str:
    # Keep only printable or common whitespace (tab/newline/carriage return)
    return ''.join(c if (c in "\n\r\t" or ord(c) >= 32) else ' ' for c in text)


def _to_ascii(text: str) -> str:
    # Remove accents/diacritics; map to ASCII equivalents
    nfkd = ud.normalize('NFKD', text)
    without_marks = ''.join(ch for ch in nfkd if not ud.combining(ch))
    # Force ASCII, dropping any remaining non-ASCII
    return without_marks.encode('ascii', 'ignore').decode('ascii')


def clean_text(original: str, *, mode: str = 'keep') -> str:
    text = original
    text = _maybe_fix_with_ftfy(text)
    text = _normalize_punct(text)
    text = _apply_weird_map(text)
    text = _strip_controls(text)
    if mode == 'ascii':
        text = _to_ascii(text)
    return text


@dataclass
class RunConfig:
    root: Path
    include_ext: Sequence[str]
    exclude_dirs: Sequence[str]
    mode: str  # 'keep' | 'ascii'
    dry: bool
    backup: bool


def iter_files(root: Path, *, include_ext: Sequence[str], exclude_dirs: Sequence[str]) -> Iterable[Path]:
    inc = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in include_ext}
    exd = set(exclude_dirs)
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        # Skip excluded directories by checking parts
        if any(part in exd for part in path.parts):
            continue
        if path.suffix.lower() in inc:
            yield path


def process_file(path: Path, cfg: RunConfig) -> tuple[bool, int, int]:
    # Returns: (changed, old_len, new_len)
    data = path.read_bytes()
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        # Fallback to cp1252; last resort replace
        try:
            text = data.decode('cp1252')
        except UnicodeDecodeError:
            text = data.decode('utf-8', errors='replace')

    fixed = clean_text(text, mode=cfg.mode)
    if fixed != text:
        if not cfg.dry:
            if cfg.backup:
                bak = path.with_suffix(path.suffix + '.bak')
                if not bak.exists():
                    shutil.copy2(path, bak)
            path.write_text(fixed, encoding='utf-8', newline='\n')
        return True, len(text), len(fixed)
    return False, len(text), len(text)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Limpia mojibake y caracteres raros en archivos de texto.")
    ap.add_argument('--root', default='.', help='Carpeta raíz (default: .)')
    ap.add_argument('--ext', nargs='*', default=['.py', '.sql', '.md', '.txt'],
                    help='Extensiones a incluir (default: .py .sql .md .txt)')
    ap.add_argument('--exclude', nargs='*', default=['venv', 'dist', 'build', '.git', '__pycache__'],
                    help='Carpetas a excluir')
    ap.add_argument('--mode', choices=['keep', 'ascii'], default='keep',
                    help="Modo de salida: 'keep' conserva tildes si es posible; 'ascii' translitera")
    ap.add_argument('--dry', action='store_true', help='Vista previa: no escribe cambios')
    ap.add_argument('--no-backup', action='store_true', help='No crear .bak antes de escribir')

    args = ap.parse_args(argv)
    cfg = RunConfig(
        root=Path(args.root).resolve(),
        include_ext=args.ext,
        exclude_dirs=args.exclude,
        mode=args.mode,
        dry=args.dry,
        backup=not args.no_backup,
    )

    changed = 0
    scanned = 0
    delta = 0
    for f in iter_files(cfg.root, include_ext=cfg.include_ext, exclude_dirs=cfg.exclude_dirs):
        scanned += 1
        was_changed, old_len, new_len = process_file(f, cfg)
        if was_changed:
            changed += 1
            delta += (new_len - old_len)
            action = 'CHANGE' if not cfg.dry else 'WOULD-CHANGE'
            print(f"[{action}] {f}")

    print(f"\nArchivos escaneados: {scanned}")
    if cfg.dry:
        print(f"Archivos que cambiarían: {changed}")
    else:
        print(f"Archivos modificados: {changed}")
    if changed:
        print(f"Δ tamaño total (bytes): {delta:+d}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

