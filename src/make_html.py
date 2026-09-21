"""Render the report to a single self-contained HTML file with figures embedded."""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

from common import RESULTS, ROOT

CSS = """
:root{color-scheme:light dark;--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--grid:#e1e0d9;--rule:#c3c2b7;--accent:#2a78d6;--code:#f2f1ed}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--rule:#383835;--accent:#3987e5;--code:#232322}}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:52rem;margin:0 auto;padding:3rem 1.25rem 6rem}
h1{font-size:1.9rem;line-height:1.25;margin:0 0 1.5rem;letter-spacing:-.01em}
h2{font-size:1.35rem;margin:2.75rem 0 .75rem;padding-top:1.25rem;border-top:1px solid var(--grid)}
h3{font-size:1.08rem;margin:1.75rem 0 .5rem;color:var(--ink)}
p,li{color:var(--ink2)}
a{color:var(--accent)}
code{background:var(--code);padding:.12em .35em;border-radius:4px;font-size:.9em;
font-family:ui-monospace,"Cascadia Code",Consolas,monospace}
pre{background:var(--code);padding:1rem;border-radius:8px;overflow-x:auto;font-size:.85rem}
pre code{background:none;padding:0}
table{border-collapse:collapse;width:100%;margin:1.25rem 0;font-size:.92rem;display:block;
overflow-x:auto}
th{text-align:left;color:var(--muted);font-weight:600;border-bottom:1px solid var(--rule);
padding:.45rem .6rem;white-space:nowrap}
td{padding:.45rem .6rem;border-bottom:1px solid var(--grid);color:var(--ink2)}
tr:last-child td{border-bottom:none}
blockquote{margin:1.25rem 0;padding:.5rem 0 .5rem 1rem;border-left:3px solid var(--rule);
color:var(--muted)}
figure{margin:1.75rem 0}
img{max-width:100%;height:auto;border-radius:8px;background:var(--surface)}
figcaption{color:var(--muted);font-size:.85rem;margin-top:.5rem}
hr{border:none;border-top:1px solid var(--grid);margin:2.5rem 0}
"""


def embed_images(html: str, base: Path) -> str:
    def repl(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2)
        p = (base / src).resolve()
        if not p.exists():
            return m.group(0)
        data = base64.b64encode(p.read_bytes()).decode()
        mime = "image/svg+xml" if p.suffix.lower() == ".svg" else "image/png"
        return (
            f'<figure><img alt="{alt}" src="data:{mime};base64,{data}">'
            f"<figcaption>{alt}</figcaption></figure>"
        )

    return re.sub(r'<img alt="([^"]*)" src="([^"]+)"\s*/?>', repl, html)


def main(args) -> None:
    import markdown

    src = ROOT / args.src
    text = src.read_text(encoding="utf-8")
    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"]
    )
    body = embed_images(body, src.parent)
    title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("# ")), args.src)
    out = ROOT / args.out
    # The language attribute follows the source file: REPORT.ru.md -> ru, everything else -> en.
    lang = "ru" if Path(args.src).name.endswith(".ru.md") else "en"
    out.write_text(
        f"<!doctype html><html lang={lang}><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>",
        encoding="utf-8",
    )
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="REPORT.md")
    ap.add_argument("--out", default="report.html")
    main(ap.parse_args())
