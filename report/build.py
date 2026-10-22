"""Build the project report PDF from report/report.md (charts come from docs/figures/).

Run from the orbit-agent folder (needs Google Chrome for the PDF step):
    python report/build.py

Writes report/report.html and report/orbit-agent_report.pdf (the PDF is not kept in git).
In report.md, [[FIGURE name.png | caption]] puts a chart from docs/figures/ in the page.
"""
import base64
import re
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "docs" / "figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 16mm 15mm 17mm 15mm;
        @bottom-center { content: counter(page); font: 8.5pt -apple-system, Arial, sans-serif; color: #888; } }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 10.2pt; line-height: 1.45; color: #1c1c1c; }
h1 { font-size: 21pt; margin: 0 0 4px; }
h1 + p { color: #555; font-size: 9.5pt; margin-top: 0; }
h2 { font-size: 14pt; margin: 20px 0 6px; padding-bottom: 3px; border-bottom: 1px solid #ddd; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14px 0 4px; break-after: avoid; }
h2.newpage { break-before: page; }
p, li { margin: 4px 0; }
ul, ol { padding-left: 20px; }
blockquote { margin: 8px 0; padding: 7px 12px; border-left: 3px solid #c98500; background: #faf6ec; font-size: 10.6pt; }
blockquote p { margin: 3px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.1pt; break-inside: avoid; }
th, td { border: 1px solid #d6d6d6; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #f2f2f2; }
figure { margin: 10px 0; break-inside: avoid; }
figure img { width: 90%; display: block; margin: 0 auto; border-radius: 4px; }
figcaption { font-size: 8.8pt; color: #555; margin-top: 4px; }
a { color: #1a5fb4; text-decoration: none; }
code { font-size: 8.8pt; background: #f2f2f2; padding: 0 3px; border-radius: 3px; }
"""


def figure(match: re.Match) -> str:
    name, caption = match.group(1).strip(), match.group(2).strip()
    data = base64.b64encode((FIGURES / name).read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{caption}">'
            f"<figcaption>{caption}</figcaption></figure>")


def main() -> None:
    text = (HERE / "report.md").read_text()
    text = re.sub(r"\[\[FIGURE (.+?) \| (.+?)\]\]", figure, text)
    body = markdown.markdown(text, extensions=["tables"])
    body = body.replace("<h2>[newpage] ", '<h2 class="newpage">')
    html = HERE / "report.html"
    html.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>orbit-agent report</title>'
                    f"<style>{CSS}</style></head><body>{body}</body></html>")
    pdf = HERE / "orbit-agent_report.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=10000", f"--print-to-pdf={pdf}", html.as_uri()],
                   check=True, capture_output=True)
    print(f"Wrote {pdf.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
