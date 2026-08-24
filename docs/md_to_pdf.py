#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_docs_to_pdf.py
Converts DOCS.md into a high-quality academic PDF with MathJax equations,
figures, tables, and typography via Playwright headless Chromium.
"""

import os
import re
import markdown
from playwright.sync_api import sync_playwright

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DOCS_DIR)
DOCS_MD_PATH = os.path.join(DOCS_DIR, "DOCS.md")
TEMP_HTML_PATH = os.path.join(DOCS_DIR, "temp_docs.html")
OUTPUT_PDF_PATH = os.path.join(DOCS_DIR, "DOCS.pdf")


def build_html():
    with open(DOCS_MD_PATH, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Convert relative screenshot paths to file:/// URIs so Playwright loads them reliably
    screenshots_dir = os.path.join(DOCS_DIR, "screenshots").replace("\\", "/")
    if not os.path.exists(os.path.join(DOCS_DIR, "screenshots")):
        screenshots_dir = os.path.join(BASE_DIR, "screenshots").replace("\\", "/")

    md_content = re.sub(
        r'!\[(.*?)\]\((?:docs/)?screenshots/(.*?)\)',
        rf'![\1](file:///{screenshots_dir}/\2)',
        md_content
    )

    # Convert markdown to HTML with tables, code blocks, and math extensions
    html_body = markdown.markdown(
        md_content,
        extensions=[
            "extra",
            "tables",
            "fenced_code",
            "codehilite",
            "toc",
            "pymdownx.arithmatex",
        ],
        extension_configs={
            "pymdownx.arithmatex": {
                "generic": True,
            }
        }
    )

    # Ensure em-dashes in prose have elegant typographic thin spaces (&thinsp;) so they do not touch adjacent glyphs in EB Garamond
    def fix_em_dashes(html):
        parts = re.split(r'(<(?:pre|code|script|span\s+class="arithmatex")[^>]*>.*?</(?:pre|code|script|span)>)', html, flags=re.DOTALL)
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(r'\s*—\s*', '&thinsp;—&thinsp;', parts[i])
        return ''.join(parts)

    html_body = fix_em_dashes(html_body)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Semantic Manifold Vector Visualizer — Thesis Documentation</title>
<!-- MathJax 3 Configuration -->
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  options: {{
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }},
  startup: {{
    pageReady: () => {{
      return MathJax.startup.defaultPageReady().then(() => {{
        window.__mathJaxDone = true;
      }});
    }}
  }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=Fira+Code:wght@400;500&display=swap');

* {{
  box-sizing: border-box;
}}

@page {{
  size: A4;
  margin: 22mm 20mm 25mm 20mm;
}}

body {{
  font-family: 'EB Garamond', Georgia, serif;
  font-size: 11.5pt;
  line-height: 1.55;
  color: #1a1a1a;
  background: #ffffff;
  margin: 0;
  padding: 0;
}}

h1 {{
  font-family: 'Cinzel', 'EB Garamond', serif;
  font-size: 20pt;
  font-weight: 700;
  line-height: 1.25;
  text-align: center;
  color: #111111;
  margin-top: 10pt;
  margin-bottom: 18pt;
  border-bottom: 2px solid #111111;
  padding-bottom: 12pt;
}}

h2 {{
  font-family: 'Cinzel', 'EB Garamond', serif;
  font-size: 14pt;
  font-weight: 700;
  color: #111111;
  margin-top: 24pt;
  margin-bottom: 8pt;
  border-bottom: 1px solid #c5c5c5;
  padding-bottom: 4pt;
  page-break-after: avoid;
}}

h3 {{
  font-size: 12pt;
  font-weight: 700;
  color: #222222;
  margin-top: 16pt;
  margin-bottom: 6pt;
  page-break-after: avoid;
}}

h4 {{
  font-size: 11.5pt;
  font-weight: 600;
  color: #333333;
  margin-top: 12pt;
  margin-bottom: 4pt;
  page-break-after: avoid;
}}

p {{
  margin-top: 0;
  margin-bottom: 8pt;
  text-align: justify;
  text-justify: inter-word;
}}

blockquote {{
  margin: 12pt 0;
  padding: 8pt 14pt;
  background: #f8f9fa;
  border-left: 3.5px solid #2b5797;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #333333;
  border-radius: 0;
}}

blockquote p {{
  margin-bottom: 4pt;
}}
blockquote p:last-child {{
  margin-bottom: 0;
}}

hr {{
  border: 0;
  height: 1px;
  background: #d0d0d0;
  margin: 18pt 0;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  margin: 14pt 0;
  font-size: 10pt;
  border-radius: 0 !important;
  page-break-inside: avoid;
}}

th, td {{
  border: 1px solid #d0d0d0;
  padding: 6pt 8pt;
  text-align: left;
  border-radius: 0 !important;
}}

th {{
  background: #f1f3f5;
  font-weight: 700;
  color: #111111;
}}

tr:nth-child(even) {{
  background: #fafbfc;
}}

pre, code {{
  font-family: 'Fira Code', 'Courier New', monospace;
  border-radius: 0 !important;
}}

code {{
  font-size: 9.5pt;
  background: #f1f3f5;
  padding: 1.5pt 4pt;
  border-radius: 0 !important;
  color: #901633;
}}

pre {{
  background: #f8f9fa;
  border: 1px solid #d0d0d0;
  border-radius: 0 !important;
  padding: 10pt 12pt;
  font-size: 9pt;
  line-height: 1.4;
  overflow-x: auto;
  margin: 12pt 0;
  page-break-inside: avoid;
  box-sizing: border-box;
}}

pre code {{
  background: transparent;
  padding: 0;
  color: #24292e;
  font-size: 9pt;
}}

/* Syntax highlighting */
.codehilite {{ background: #f8f9fa; }}
.codehilite .k, .codehilite .kd, .codehilite .kn, .codehilite .kr {{ color: #d73a49; font-weight: 600; }}
.codehilite .s, .codehilite .s1, .codehilite .s2, .codehilite .sb, .codehilite .sc {{ color: #032f62; }}
.codehilite .c, .codehilite .c1, .codehilite .cm {{ color: #6a737d; font-style: italic; }}
.codehilite .nf, .codehilite .nc {{ color: #6f42c1; font-weight: 600; }}
.codehilite .mi, .codehilite .mf, .codehilite .mh, .codehilite .m {{ color: #005cc5; }}
.codehilite .na, .codehilite .nt {{ color: #22863a; }}
.codehilite .o, .codehilite .ow {{ color: #d73a49; }}
.codehilite .nb, .codehilite .bp {{ color: #005cc5; }}
.codehilite .nx, .codehilite .p {{ color: #24292e; }}

img {{
  display: block;
  width: calc(100% - 4px);
  max-width: calc(100% - 4px);
  height: auto;
  margin: 14pt 0 4pt 0;
  border: 1.5px solid #111111;
  border-radius: 0 !important;
  box-shadow: none !important;
  box-sizing: border-box !important;
  page-break-inside: avoid;
}}

p:has(> img) {{
  overflow: visible;
}}

em {{
  display: block;
  text-align: left;
  font-size: 10pt;
  color: #111111;
  margin-top: 4pt;
  margin-bottom: 14pt;
  font-style: italic;
  line-height: 1.4;
}}


.mjx-chtml {{
  font-size: 108% !important;
}}

ul, ol {{
  margin-top: 4pt;
  margin-bottom: 10pt;
  padding-left: 20pt;
}}

li {{
  margin-bottom: 3pt;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    with open(TEMP_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"[OK] Generated temporary HTML at: {TEMP_HTML_PATH}")


def convert_to_pdf():
    print("[*] Starting Playwright PDF generation...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the HTML document
        file_uri = f"file:///{TEMP_HTML_PATH.replace('\\', '/')}"
        page.goto(file_uri, wait_until="networkidle")

        # Wait for MathJax to finish typesetting equations
        try:
            page.wait_for_function("window.__mathJaxDone === true", timeout=15000)
            print("[OK] MathJax typesetting completed.")
        except Exception:
            print("[WARN] MathJax timeout, proceeding with render.")

        # Brief pause to ensure all high-res images and web fonts are fully rendered
        page.wait_for_timeout(2000)

        # Generate PDF with A4 margins and header/footer page numbers
        page.pdf(
            path=OUTPUT_PDF_PATH,
            format="A4",
            print_background=True,
            margin={
                "top": "22mm",
                "bottom": "25mm",
                "left": "20mm",
                "right": "20mm"
            },
            display_header_footer=True,
            header_template="<div></div>",
            footer_template="""
            <div style="width: 100%; font-family: 'EB Garamond', Georgia, serif; font-size: 9pt; color: #777; text-align: right; padding-right: 20mm;">
                <span class="pageNumber"></span> / <span class="totalPages"></span>
            </div>
            """
        )

        browser.close()

    # Clean up temporary HTML
    if os.path.exists(TEMP_HTML_PATH):
        os.remove(TEMP_HTML_PATH)

    print(f"[SUCCESS] PDF successfully created: {OUTPUT_PDF_PATH}")


if __name__ == "__main__":
    build_html()
    convert_to_pdf()
