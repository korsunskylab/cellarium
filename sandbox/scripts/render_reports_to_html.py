"""Render reports/*.md to self-contained HTML files with embedded PNGs (base64).

For each markdown file in reports/, produce reports/<name>.html with:
  - GitHub-flavored markdown rendered via the `markdown` package
  - All image references resolved relative to the .md file
  - PNGs read from disk and embedded as base64 data URIs (single-file, no broken
    images even when emailed/uploaded)
"""
from __future__ import annotations
import base64, re, sys
from pathlib import Path

try:
    import markdown
except ImportError:
    print("install markdown: pip install markdown"); sys.exit(1)

REPORTS = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium/reports")
IMG_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 980px; margin: 2em auto; padding: 0 1em; line-height: 1.55; color: #24292e; }
h1 { border-bottom: 2px solid #eaecef; padding-bottom: .3em; }
h2 { border-bottom: 1px solid #eaecef; padding-bottom: .3em; margin-top: 1.5em; }
img { max-width: 100%; height: auto; display: block; margin: 1em 0; border: 1px solid #eee; }
code { background: #f6f8fa; padding: 0.1em 0.4em; border-radius: 3px; font-size: 0.9em; }
pre code { background: none; padding: 0; }
pre { background: #f6f8fa; padding: 1em; border-radius: 6px; overflow-x: auto; }
table { border-collapse: collapse; margin: 1em 0; }
th, td { border: 1px solid #dfe2e5; padding: 6px 13px; }
th { background: #f6f8fa; }
em { color: #586069; }
"""


def embed_images(md_text: str, base_dir: Path) -> str:
    def replace(m):
        alt, path = m.group(1), m.group(2)
        full = (base_dir / path).resolve()
        if full.suffix.lower() == ".png" and full.exists():
            data = base64.b64encode(full.read_bytes()).decode()
            return f'![{alt}](data:image/png;base64,{data})'
        return m.group(0)
    return IMG_RE.sub(replace, md_text)


def render(md_path: Path):
    md_text = md_path.read_text()
    md_text = embed_images(md_text, md_path.parent)
    html = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    full = f'<!doctype html><html><head><meta charset="utf-8"><title>{md_path.stem}</title>' \
           f'<style>{CSS}</style></head><body>{html}</body></html>'
    out = md_path.with_suffix(".html")
    out.write_text(full)
    return out


def main():
    if not REPORTS.exists():
        print(f"no reports dir at {REPORTS}"); return
    for md in sorted(REPORTS.rglob("*.md")):
        out = render(md)
        rel = md.relative_to(REPORTS)
        print(f"  {rel} → {out.name} ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
