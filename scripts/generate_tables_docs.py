#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Dict, List

import pandas as pd

# -------------------------
# Settings
# -------------------------
MAX_ROWS_PREVIEW = int(os.getenv("DOCS_MAX_ROWS", "50"))  # set -1 for all rows (careful!)
MAX_COL_WIDTH = int(os.getenv("DOCS_MAX_COL_WIDTH", "40"))
TABLE_GLOB = "**/*.csv"  # recurse under quma/Tables
SECTIONS_EXCLUDE = {"__pycache__", ".ipynb_checkpoints"}

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "quma" / "Tables"
DOCS_ROOT = PROJECT_ROOT / "docs" / "source" / "Tables"
DOCS_ROOT.mkdir(parents=True, exist_ok=True)

TOP_PAGE = PROJECT_ROOT / "docs" / "source" / "tables.rst"


def safe_text(x) -> str:
    """Stringify and trim cell content for RST tables."""
    s = "" if pd.isna(x) else str(x)
    s = s.replace("|", "\\|")  # don't break RST table cells
    if len(s) > MAX_COL_WIDTH:
        s = s[: MAX_COL_WIDTH - 1] + "…"
    return s


def df_to_rst_grid_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as an RST grid table."""
    if df.empty:
        return "\n*(empty table)*\n"

    df = df.copy()
    df.columns = [safe_text(c) for c in df.columns]
    df = df.applymap(safe_text)

    widths = []
    for col in df.columns:
        w = max(len(col), *(len(v) for v in df[col].astype(str).tolist()))
        widths.append(w)

    def horiz(sep: str = "-") -> str:
        return "+" + "+".join(sep * (w + 2) for w in widths) + "+\n"

    def row(cells: Iterable[str]) -> str:
        cells = list(cells)
        padded = [f" {c}{' ' * (w - len(c))} " for c, w in zip(cells, widths)]
        return "|" + "|".join(padded) + "|\n"

    out = []
    out.append(horiz("-"))
    out.append(row(df.columns.tolist()))
    out.append(horiz("="))
    for _, r in df.iterrows():
        out.append(row(r.tolist()))
        out.append(horiz("-"))
    return "".join(out)


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)
    except Exception as e:
        print(f"⚠️  Failed to read {path}: {e}", file=sys.stderr)
        return pd.DataFrame()


def rel_from_page(page_dir: Path, target_path: Path) -> str:
    """
    Relative path from the directory containing the RST page to the target file.
    Always return POSIX-style separators for Sphinx.
    """
    rel = Path(os.path.relpath(target_path, start=page_dir))
    return rel.as_posix()


def collect_csvs() -> Dict[str, List[Path]]:
    """
    Return dict: section_name -> list of CSV paths.
    Section is the immediate subdir under quma/Tables (or 'root' for top-level files).
    """
    files = list(SRC_ROOT.glob(TABLE_GLOB))
    sections: Dict[str, List[Path]] = {}
    for f in files:
        if not f.is_file():
            continue
        try:
            parent = f.relative_to(SRC_ROOT).parts[0]
        except Exception:
            parent = "root"
        section = parent if parent not in SECTIONS_EXCLUDE else "root"
        sections.setdefault(section, []).append(f)
    for k in sections:
        sections[k].sort()
    return sections


def write_section(section: str, csvs: List[Path]):
    """
    Create docs for a section:
      docs/source/Tables/<section>/index.rst
      docs/source/Tables/<section>/<file>.rst for each CSV
      Also include README.md if present in the source section.
    """
    sec_dir = DOCS_ROOT / section
    sec_dir.mkdir(parents=True, exist_ok=True)

    # Section title
    title = section.replace("_", " ").title() if section != "root" else "Tables (Root)"
    underline = "=" * len(title)
    idx_lines = [title, underline, ""]

    # Include section-level README.md (MyST)
    src_readme = (SRC_ROOT / section / "README.md") if section != "root" else (SRC_ROOT / "README.md")
    if src_readme.exists():
        include_rel = rel_from_page(sec_dir, src_readme)
        idx_lines += [
            f".. include:: {include_rel}",
            "   :parser: myst_parser.sphinx_",
            "",
        ]

    # ToC for CSV pages
    idx_lines += [".. toctree::", "   :maxdepth: 1", ""]

    # Per-file pages
    for csv_path in csvs:
        df = read_csv_safe(csv_path)
        nrows, ncols = df.shape
        preview = df if MAX_ROWS_PREVIEW < 0 else df.head(MAX_ROWS_PREVIEW)

        page_name = csv_path.stem
        page_path = sec_dir / f"{page_name}.rst"
        page_title = page_name.replace("_", " ").title()
        page_underline = "=" * len(page_title)

        # paths relative to the page location
        download_rel = rel_from_page(page_path.parent, csv_path)
        sidecar_md = csv_path.with_suffix(".md")
        sidecar_rel = rel_from_page(page_path.parent, sidecar_md) if sidecar_md.exists() else None

        with open(page_path, "w", encoding="utf-8") as f:
            f.write(f"{page_title}\n{page_underline}\n\n")
            f.write(f"**Source:** :download:`{csv_path.name} <{download_rel}>`\n\n")
            f.write(f"- **Rows:** {nrows}\n")
            f.write(f"- **Columns:** {ncols}\n\n")

            # Optional per-CSV Markdown notes (MyST)
            if sidecar_rel is not None:
                f.write(".. include:: " + sidecar_rel + "\n")
                f.write("   :parser: myst_parser.sphinx_\n\n")

            # Table preview
            if not preview.empty:
                f.write(df_to_rst_grid_table(preview))
                f.write("\n")
            else:
                f.write("*(No preview available — file could not be read or is empty.)*\n\n")

        idx_lines.append(f"   {page_name}")

    # Write section index
    with open(sec_dir / "index.rst", "w", encoding="utf-8") as f:
        f.write("\n".join(idx_lines) + "\n")


def write_top(sections: Iterable[str]):
    title = "Tables"
    underline = "=" * len(title)
    lines = [title, underline, ""]

    # Top-level README.md (if present) above the global ToC
    top_readme = SRC_ROOT / "README.md"
    if top_readme.exists():
        include_rel = rel_from_page(TOP_PAGE.parent, top_readme)
        lines += [
            f".. include:: {include_rel}",
            "   :parser: myst_parser.sphinx_",
            "",
        ]

    lines += [".. toctree::", "   :maxdepth: 2", ""]
    for sec in sorted(sections):
        lines.append(f"   Tables/{sec}/index")

    with open(TOP_PAGE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    if not SRC_ROOT.exists():
        print(f"⚠️  {SRC_ROOT} not found; skipping", file=sys.stderr)
        return
    sections = collect_csvs()
    for sec, paths in sections.items():
        if not paths:
            continue
        print(f"📁 Section: {sec} ({len(paths)} csv)")
        write_section(sec, paths)
    write_top(sections.keys())
    print("✅ Generated docs for CSV tables.")


if __name__ == "__main__":
    main()
