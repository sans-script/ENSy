#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
md_to_pdf.py
Converts DOCS.md into a high-quality academic PDF.

Pipeline:
  1. Pre-processor  — plots (matplotlib SVG), Mermaid, theorems, booktabs
  2. HTML builder   — Markdown → HTML + CSS + MathJax (AMS) + Mermaid.js
  3. Playwright PDF — waits for MathJax + Mermaid, exports A4 PDF

Usage:
  python md_to_pdf.py [--input FILE] [--output FILE] [--no-cleanup]
"""

import argparse
import os
import re
import subprocess
import sys

def ensure_environment():
    try:
        import markdown
        import playwright
        return
    except ImportError:
        pass

    # Check for existing virtualenvs in workspace root that have required dependencies
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base_dir, ".venv-tests", "Scripts", "python.exe"),
        os.path.join(base_dir, ".venv", "Scripts", "python.exe"),
        os.path.join(base_dir, ".venv-tests", "bin", "python"),
        os.path.join(base_dir, ".venv", "bin", "python"),
    ]
    for cand in candidates:
        if os.path.exists(cand) and os.path.abspath(cand) != os.path.abspath(sys.executable):
            ret = subprocess.call([cand, os.path.abspath(__file__), *sys.argv[1:]])
            sys.exit(ret)

    # If no virtualenv is available, install missing dependencies in current python
    required_packages = [
        ("markdown", "markdown"),
        ("pymdownx", "pymdown-extensions"),
        ("playwright", "playwright"),
        ("pypdf", "pypdf"),
        ("matplotlib", "matplotlib"),
    ]
    missing = []
    for mod_name, pkg_name in required_packages:
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(pkg_name)
    if missing:
        print(f"[*] Installing missing documentation dependencies: {missing}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing, "--default-timeout", "100"])
            if "playwright" in missing:
                subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
            print("[OK] Dependencies installed successfully.")
        except Exception as exc:
            print(f"[WARN] Error auto-installing dependencies: {exc}")

ensure_environment()

import markdown
from playwright.sync_api import sync_playwright

# Attempt to import the optional plot renderer
try:
    from plot_renderer import preprocess_plots
    _PLOTS_AVAILABLE = True
except ImportError:
    _PLOTS_AVAILABLE = False

DOCS_DIR       = os.path.dirname(os.path.abspath(__file__))
BASE_DIR       = os.path.dirname(DOCS_DIR)
DOCS_MD_PATH   = os.path.join(DOCS_DIR, "DOCS.md")
TEMP_HTML_PATH = os.path.join(DOCS_DIR, "temp_docs.html")
OUTPUT_PDF_PATH = os.path.join(DOCS_DIR, "DOCS.pdf")


# ---------------------------------------------------------------------------
# Pre-processors
# ---------------------------------------------------------------------------

def preprocess_mermaid(md_content: str) -> str:
    """Replace ```mermaid blocks with <div class="mermaid"> elements."""
    return re.sub(
        r'```mermaid\s*\n(.*?)\n```',
        lambda m: f'<div class="mermaid">\n{m.group(1)}\n</div>',
        md_content,
        flags=re.DOTALL,
    )


def preprocess_theorems(md_content: str) -> str:
    """
    Replace :::type[Title]{#anchor} ... ::: blocks with styled HTML.

    Supported environments: definition, theorem, lemma, corollary,
                            proposition, remark, example, proof
    """
    # Counters per environment type (proof is never numbered)
    counters: dict[str, int] = {}
    proof_label_map = {
        "definition":   "Definition",
        "theorem":      "Theorem",
        "lemma":        "Lemma",
        "corollary":    "Corollary",
        "proposition":  "Proposition",
        "remark":       "Remark",
        "example":      "Example",
        "proof":        "Proof",
    }
    env_pattern = re.compile(
        r':::(?P<env>definition|theorem|lemma|corollary|proposition|remark|example|proof)'
        r'(?:\[(?P<title>[^\]]*)\])?'
        r'(?:\{#(?P<anchor>[^}]*)\})?'
        r'\s*\n(?P<body>.*?)\n:::',
        re.DOTALL | re.IGNORECASE,
    )

    def replace_env(m: re.Match) -> str:
        env   = m.group("env").lower()
        title = m.group("title") or ""
        anchor = m.group("anchor") or ""
        body  = m.group("body").strip()

        label_word = proof_label_map.get(env, env.capitalize())

        if env == "proof":
            label_html = f'<span class="theorem-label">{label_word}.</span>'
            qed = '<span class="qed">&#9632;</span>'  # ■
            number_str = ""
        else:
            counters[env] = counters.get(env, 0) + 1
            number_str = f" {counters[env]}"
            label_html = f'<span class="theorem-label">{label_word}{number_str}</span>'
            qed = ""

        title_html = (
            f' <span class="theorem-title">({title})</span>' if title else ""
        )
        anchor_attr = f' id="{anchor}"' if anchor else ""

        # Convert body Markdown inline (bold, italic, math) — keep it simple
        return (
            f'<div class="theorem-block theorem-{env}"{anchor_attr}>'
            f'<div class="theorem-header">{label_html}{title_html}</div>'
            f'<div class="theorem-body">{body}{qed}</div>'
            f'</div>'
        )

    return env_pattern.sub(replace_env, md_content)


def preprocess_booktabs(md_content: str) -> str:
    """
    Wrap :::booktabs ... ::: blocks with a marker div that uses markdown='1'
    so that the Python-Markdown md_in_html extension processes the inner table.
    """
    pattern = re.compile(r':::booktabs\s*\n(.*?)\n:::', re.DOTALL)

    def replace_bt(m: re.Match) -> str:
        inner = m.group(1).strip()
        return f'<div class="booktabs-wrapper" markdown="1">\n\n{inner}\n\n</div>'

    return pattern.sub(replace_bt, md_content)


def preprocess_pagebreaks(md_content: str) -> str:
    """
    Replace :::pagebreak, :::page-break, or ---pagebreak--- directives
    with clean <div class="page-break"></div> elements.
    """
    return re.sub(
        r'(?::::+page-?break|---page-?break---)',
        '<div class="page-break"></div>',
        md_content,
        flags=re.IGNORECASE,
    )


# Unique tokens that Python-Markdown will wrap in <p> tags, preserving
# surrounding Markdown (headings, tables, images) completely intact.
_COL_OPEN_TOKEN  = 'XXCOLUMNOPENXX'
_COL_CLOSE_TOKEN = 'XXCOLUMNCLOSEXX'


def preprocess_columns(md_content: str) -> str:
    """
    Stage 1 (pre-Markdown): replace ---columns{...}--- / ---/columns--- with
    plain-text tokens on their own blank-line-separated paragraphs.
    Python-Markdown treats them as ordinary <p> text and leaves all surrounding
    Markdown (headings, tables, images) completely untouched.
    postprocess_columns() replaces the <p>TOKEN</p> tags with real divs.
    """
    open_pat  = re.compile(r'---columns\{([^}]*)\}---')
    close_pat = re.compile(r'---/columns---')

    md_content = open_pat.sub(
        lambda m: f'\n\n{_COL_OPEN_TOKEN} {m.group(1)} {_COL_OPEN_TOKEN}\n\n',
        md_content,
    )
    md_content = close_pat.sub(
        f'\n\n{_COL_CLOSE_TOKEN}\n\n',
        md_content,
    )
    return md_content


def postprocess_columns(html: str) -> str:
    """
    Stage 2 (post-Markdown): replace the <p>TOKEN</p> paragraphs that
    Python-Markdown produced with the real column-layout divs.
    """
    open_pat  = re.compile(
        r'<p>\s*' + re.escape(_COL_OPEN_TOKEN) + r'\s*(.*?)\s*' + re.escape(_COL_OPEN_TOKEN) + r'\s*</p>',
        re.DOTALL,
    )
    close_pat = re.compile(
        r'<p>\s*' + re.escape(_COL_CLOSE_TOKEN) + r'\s*</p>'
    )

    def parse_attrs(attr_str: str) -> dict:
        attrs = {}
        for m in re.finditer(r'(\w+)=([^\s]+)', attr_str):
            attrs[m.group(1)] = m.group(2)
        return attrs

    def replace_open(m: re.Match) -> str:
        attrs  = parse_attrs(m.group(1))
        count  = attrs.get('count', '2')
        space  = attrs.get('space', '8')          # mm
        sep    = attrs.get('separate', 'false').lower() == 'true'
        crule  = 'column-rule: 0.5px solid #cccccc;' if sep else ''
        return (
            f'<div class="columns-layout" '
            f'style="column-count:{count}; column-gap:{space}mm; {crule}">'
        )

    html = open_pat.sub(replace_open, html)
    html = close_pat.sub('</div>', html)
    return html


def preprocess_figures(md_content: str) -> str:
    """
    Find Markdown images followed by italic captions:
      ![Alt text](path)
      _Figure X: Caption text..._
    and transform them into structured IEEE/ACM <figure> elements.
    """
    pattern = re.compile(
        r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)\s*\n+_(?:Figure|Fig\.?)\s*(?P<num>\d+)\s*:\s*(?P<caption>[^\n_]+)_',
        re.MULTILINE
    )

    def replace_fig(m: re.Match) -> str:
        alt = m.group("alt")
        src = m.group("src")
        num = m.group("num")
        cap = m.group("caption").strip()
        return (
            f'<figure class="academic-figure">\n'
            f'  <img src="{src}" alt="{alt}" />\n'
            f'  <figcaption><strong class="caption-label">Fig. {num}.</strong> {cap}</figcaption>\n'
            f'</figure>'
        )

    return pattern.sub(replace_fig, md_content)


def preprocess_images(md_content: str, docs_dir: str, base_dir: str) -> str:
    """Convert relative screenshot paths to file:/// URIs."""
    screenshots_dir = os.path.join(docs_dir, "screenshots").replace("\\", "/")
    if not os.path.exists(os.path.join(docs_dir, "screenshots")):
        screenshots_dir = os.path.join(base_dir, "screenshots").replace("\\", "/")
    md_content = re.sub(
        r'!\[(.*?)\]\((?:docs/)?screenshots/(.*?)\)',
        rf'![\1](file:///{screenshots_dir}/\2)',
        md_content,
    )
    md_content = preprocess_figures(md_content)
    return md_content


def fix_em_dashes(html: str) -> str:
    """Add typographic thin-spaces around em-dashes, skipping code/math spans."""
    parts = re.split(
        r'(<(?:pre|code|script|span\s+class="arithmatex")[^>]*>.*?</(?:pre|code|script|span)>)',
        html,
        flags=re.DOTALL,
    )
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'\s*—\s*', '&thinsp;—&thinsp;', parts[i])
    return ''.join(parts)


def fix_inline_code_breaks(html: str) -> str:
    """
    Insert <wbr> break opportunities inside inline <code> tags (not in <pre>)
    after delimiters like /, ., _, -, :, {, }, (, ), etc. and at camelCase boundaries
    so long identifiers and endpoints wrap cleanly without causing white rivers.
    """
    def inject_wbr(m: re.Match) -> str:
        code_text = m.group(1)
        # Add <wbr> after symbols: / . _ - : { } ( ) , = ? &
        code_text = re.sub(r'([/._\-:{}(),?=&])', r'\1<wbr>', code_text)
        # Add <wbr> at camelCase transitions
        code_text = re.sub(r'([a-z0-9])([A-Z])', r'\1<wbr>\2', code_text)
        return f'<code>{code_text}</code>'

    parts = re.split(r'(<pre[^>]*>.*?</pre>)', html, flags=re.DOTALL)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'<code>(.*?)</code>', inject_wbr, parts[i], flags=re.DOTALL)
    return ''.join(parts)


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(input_path: str = DOCS_MD_PATH,
               output_path: str = TEMP_HTML_PATH) -> None:

    with open(input_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    docs_dir = os.path.dirname(os.path.abspath(input_path))
    base_dir = os.path.dirname(docs_dir)

    # --- Pre-processing pipeline (order matters) ---
    if _PLOTS_AVAILABLE:
        md_content = preprocess_plots(md_content)
        print("[OK] Plot blocks rendered.")
    else:
        print("[WARN] matplotlib not available — plot blocks skipped.")

    md_content = preprocess_columns(md_content)
    md_content = preprocess_mermaid(md_content)
    md_content = preprocess_theorems(md_content)
    md_content = preprocess_booktabs(md_content)
    md_content = preprocess_pagebreaks(md_content)
    md_content = preprocess_images(md_content, docs_dir, base_dir)

    # --- Markdown → HTML ---
    html_body = markdown.markdown(
        md_content,
        extensions=[
            "extra",
            "tables",
            "fenced_code",
            "codehilite",
            "toc",
            "pymdownx.arithmatex",
            "md_in_html",          # enables Markdown inside HTML blocks
        ],
        extension_configs={
            "pymdownx.arithmatex": {"generic": True}
        },
    )

    html_body = fix_em_dashes(html_body)
    html_body = fix_inline_code_breaks(html_body)

    # Post-process: inject column divs now that Markdown is already HTML
    html_body = postprocess_columns(html_body)

    # Post-process: give booktabs-wrapper tables the right CSS class
    html_body = re.sub(
        r'<div class="booktabs-wrapper">\s*(<table)',
        r'<div class="booktabs-wrapper">\1 class="booktabs-table"',
        html_body,
    )

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ENSy — Embedding Navigation System — Technical Documentation</title>

<!-- MathJax 3 — AMS numbering + eqref support -->
<script>
window.MathJax = {{
  tex: {{
    tags: 'ams',
    tagSide: 'right',
    tagIndent: '0.8em',
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true,
    processEnvironments: true,
  }},
  chtml: {{
    matchFontHeight: false,
    scale: 1,
  }},
  options: {{
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }},
  startup: {{
    pageReady: () => {{
      return MathJax.startup.defaultPageReady().then(() => {{
        // Dynamic calculation: measure equation widths and auto-scale if wider than column/container
        const autoscaleEquations = () => {{
          document.querySelectorAll('mjx-container[jax="CHTML"][display="true"]').forEach(container => {{
            const parent = container.parentElement;
            if (!parent) return;
            const parentWidth = parent.clientWidth || parent.getBoundingClientRect().width;
            if (!parentWidth || parentWidth <= 0) return;

            container.style.fontSize = '100%';
            const mathEl = container.querySelector('mjx-math') || container;
            const mathWidth = mathEl.scrollWidth || mathEl.getBoundingClientRect().width;

            if (mathWidth > parentWidth - 4) {{
              const scale = Math.min(1.0, (parentWidth - 6) / mathWidth);
              container.style.setProperty('font-size', (scale * 100).toFixed(1) + '%', 'important');
              container.style.maxWidth = '100%';
            }}
          }});
        }};

        autoscaleEquations();
        window.__mathJaxDone = true;
      }});
    }}
  }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>

<!-- Mermaid diagram renderer -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({{
    startOnLoad: true,
    theme: 'base',
    themeVariables: {{
      fontFamily: "'EB Garamond', Georgia, serif",
      fontSize: '11px',
      primaryColor: '#e8f0fb',
      primaryBorderColor: '#2b5797',
      primaryTextColor: '#111111',
      lineColor: '#555555',
      edgeLabelBackground: '#ffffff',
    }},
    flowchart: {{ curve: 'linear' }},
  }});
  document.addEventListener('DOMContentLoaded', () => {{
    mermaid.init(undefined, '.mermaid').then(() => {{
      window.__mermaidDone = true;
    }}).catch(() => {{
      window.__mermaidDone = true;
    }});
  }});
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=Fira+Code:wght@400;500&display=swap');

@page {{
  size: 210mm 297mm;
  margin: 22mm 20mm;
}}

* {{
  box-sizing: border-box;
}}

html {{
  margin: 0;
  padding: 0;
  width: 100%;
}}

body {{
  font-family: 'EB Garamond', Georgia, serif;
  font-size: 10.5pt;
  line-height: 1.5;
  color: #1a1a1a;
  background: #ffffff;
  margin: 0;
  padding: 0;
  width: 100%;
  box-sizing: border-box;
  hyphens: auto;
  -webkit-hyphens: auto;
  text-rendering: optimizeLegibility;
}}

/* ── Headings ───────────────────────────────────────────── */
h1 {{
  font-family: 'Cinzel', 'EB Garamond', serif;
  font-size: 19pt;
  font-weight: 700;
  line-height: 1.25;
  text-align: center;
  color: #111111;
  margin-top: 10pt;
  margin-bottom: 16pt;
  border-bottom: 1.5px solid #111111;
  padding-bottom: 8pt;
  hyphens: none;
}}
h2 {{
  font-family: 'Cinzel', 'EB Garamond', serif;
  font-size: 13pt;
  font-weight: 700;
  color: #111111;
  text-align: left !important;
  text-wrap: balance;
  margin-top: 20pt;
  margin-bottom: 6pt;
  border-bottom: none;
  padding-bottom: 0;
  page-break-after: avoid;
  break-after: avoid;
  hyphens: none;
}}
h3 {{
  font-size: 11.5pt;
  font-weight: 700;
  color: #222222;
  text-align: left !important;
  text-wrap: balance;
  margin-top: 14pt;
  margin-bottom: 5pt;
  page-break-after: avoid;
  break-after: avoid;
  hyphens: none;
}}
h4 {{
  font-size: 11pt;
  font-weight: 600;
  color: #333333;
  text-align: left !important;
  text-wrap: balance;
  margin-top: 10pt;
  margin-bottom: 4pt;
  page-break-after: avoid;
  break-after: avoid;
  hyphens: none;
}}

/* ── Body text ──────────────────────────────────────────── */
p {{
  margin-top: 0;
  margin-bottom: 7pt;
  text-align: justify;
  hyphens: auto;
  -webkit-hyphens: auto;
  word-break: normal;
  overflow-wrap: break-word;
  text-wrap: pretty;
}}

blockquote {{
  margin: 12pt 0;
  padding: 8pt 14pt;
  background: #f8f9fa;
  border-left: 3.5px solid #2b5797;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #333333;
}}
blockquote p {{ margin-bottom: 4pt; }}
blockquote p:last-child {{ margin-bottom: 0; }}

hr {{
  border: 0;
  height: 1px;
  background: #d0d0d0;
  margin: 18pt 0;
}}

ul, ol {{
  margin-top: 4pt;
  margin-bottom: 10pt;
  padding-left: 20pt;
}}
li {{ margin-bottom: 3pt; }}

/* ── Standard tables ────────────────────────────────────── */
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 14pt 0;
  font-size: 10pt;
  page-break-inside: avoid;
}}
th, td {{
  border: 1px solid #d0d0d0;
  padding: 6pt 8pt;
  text-align: left;
}}
th {{
  background: #f1f3f5;
  font-weight: 700;
  color: #111111;
}}
tr:nth-child(even) {{ background: #fafbfc; }}

/* ── Booktabs tables ────────────────────────────────────── */
.booktabs-wrapper table {{
  width: 100%;
  border-collapse: collapse;
  margin: 14pt 0;
  font-size: 10pt;
  font-variant-numeric: tabular-nums;
  page-break-inside: avoid;
  border-top: 2px solid #111111;
  border-bottom: 2px solid #111111;
}}
.booktabs-wrapper th,
.booktabs-wrapper td {{
  border: none;
  border-bottom: none;
  padding: 5pt 10pt;
  text-align: left;
  background: transparent;
}}
.booktabs-wrapper th {{
  font-weight: 600;
  border-bottom: 1px solid #111111;
  background: transparent;
}}
.booktabs-wrapper tr:nth-child(even) {{ background: transparent; }}

/* ── Code ───────────────────────────────────────────────── */
pre, code {{
  font-family: 'Fira Code', 'Courier New', monospace;
  font-variant-ligatures: none !important;
  font-feature-settings: "liga" 0, "calt" 0 !important;
  border-radius: 0 !important;
}}
code {{
  font-size: 8pt;
  background: transparent;
  padding: 0.5pt 1.5pt;
  color: #87132d;
  white-space: normal !important;
  word-break: break-word !important;
  overflow-wrap: anywhere !important;
}}
code:not(pre code) {{
  font-size: 8pt;
  background: transparent !important;
  padding: 0;
  color: #87132d;
  white-space: normal !important;
  word-break: break-word !important;
  overflow-wrap: anywhere !important;
  line-break: anywhere !important;
}}
pre {{
  background: #ffffff;
  border: none !important;
  padding: 6pt 0;
  font-size: 7.8pt;
  line-height: 1.35;
  white-space: pre-wrap !important;
  word-break: break-word !important;
  overflow-wrap: break-word !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
  margin: 8pt 0;
  page-break-inside: avoid;
}}
pre code {{
  background: transparent;
  padding: 0;
  color: #24292e;
  font-size: 7.8pt;
  white-space: pre-wrap !important;
  word-break: break-word !important;
  overflow-wrap: break-word !important;
}}

/* Syntax highlighting */
.codehilite {{ background: #ffffff; }}
.codehilite .k,  .codehilite .kd, .codehilite .kn, .codehilite .kr {{ color: #d73a49; font-weight: 600; }}
.codehilite .s,  .codehilite .s1, .codehilite .s2, .codehilite .sb, .codehilite .sc {{ color: #032f62; }}
.codehilite .c,  .codehilite .c1, .codehilite .cm {{ color: #6a737d; font-style: italic; }}
.codehilite .nf, .codehilite .nc {{ color: #6f42c1; font-weight: 600; }}
.codehilite .mi, .codehilite .mf, .codehilite .mh, .codehilite .m {{ color: #005cc5; }}
.codehilite .na, .codehilite .nt {{ color: #22863a; }}
.codehilite .o,  .codehilite .ow {{ color: #d73a49; }}
.codehilite .nb, .codehilite .bp {{ color: #005cc5; }}
.codehilite .nx, .codehilite .p  {{ color: #24292e; }}

/* ── Figures & Captions (IEEE/ACM Standard) ─────────────── */
.academic-figure {{
  margin: 10pt 0 14pt 0;
  padding: 0;
  text-align: center;
  page-break-inside: avoid;
  break-inside: avoid;
}}
.academic-figure img {{
  display: block;
  width: 100%;
  max-width: 100%;
  height: auto;
  margin: 0 auto 5pt auto;
  border: 0.5px solid #333333;
  box-sizing: border-box;
}}
.academic-figure figcaption {{
  font-family: 'EB Garamond', Georgia, serif;
  font-size: 9pt;
  line-height: 1.35;
  color: #222222;
  text-align: justify;
  text-justify: inter-word;
  padding: 0 1pt;
  margin: 0;
}}
.caption-label {{
  font-weight: 700;
  color: #111111;
  margin-right: 2pt;
}}

img {{
  display: block;
  width: 100%;
  max-width: 100%;
  height: auto;
  margin: 8pt 0 4pt 0;
  border: 0.5px solid #333333;
  box-sizing: border-box;
  page-break-inside: avoid;
}}

/* ── MathJax ────────────────────────────────────────────── */
.mjx-chtml {{
  font-size: 100% !important;
  max-width: 100% !important;
}}
mjx-container {{
  font-size: 100% !important;
}}
mjx-container:not([display="true"]) {{
  font-size: 100% !important;
  vertical-align: -0.05em;
  margin: 0 0.5pt !important;
  overflow-wrap: anywhere;
}}
mjx-container[jax="CHTML"][display="true"] {{
  margin: 8pt 0 !important;
  overflow-x: auto;
  overflow-y: hidden;
  max-width: 100%;
  font-size: 100% !important;
}}

/* ── Mermaid diagrams ───────────────────────────────────── */
.mermaid {{
  text-align: center;
  margin: 16pt 0;
  page-break-inside: avoid;
  font-family: 'EB Garamond', Georgia, serif !important;
}}
.mermaid svg {{
  max-width: 100%;
  height: auto;
}}

/* ── Theorem / Definition / Proof environments ──────────── */
.theorem-block {{
  margin: 14pt 0;
  padding: 10pt 14pt;
  page-break-inside: avoid;
  border-radius: 0;
}}
.theorem-header {{
  margin-bottom: 5pt;
}}
.theorem-label {{
  font-weight: 700;
  color: #111111;
}}
.theorem-title {{
  font-style: italic;
  color: #333333;
}}
.theorem-body {{
  font-style: italic;
  color: #1a1a1a;
  line-height: 1.55;
  position: relative;
}}

/* environment-specific borders and backgrounds */
.theorem-definition {{
  background: #f0f4ff;
  border-left: 3.5px solid #2b5797;
}}
.theorem-theorem,
.theorem-proposition {{
  background: #fff8f0;
  border-left: 3.5px solid #e36209;
}}
.theorem-lemma,
.theorem-corollary {{
  background: #f2fff4;
  border-left: 3.5px solid #22863a;
}}
.theorem-remark,
.theorem-example {{
  background: #f8f9fa;
  border-left: 3.5px solid #888888;
  font-style: normal;
}}
.theorem-proof {{
  background: transparent;
  border: none;
  padding-left: 0;
  font-style: normal;
  color: #333333;
}}
.theorem-proof .theorem-label {{
  font-style: italic;
  font-weight: 400;
}}
.theorem-body .qed {{
  display: block;
  text-align: right;
  font-style: normal;
  font-size: 11pt;
  color: #111111;
  margin-top: 4pt;
}}

/* ── Plot figures ───────────────────────────────────────── */
.plot-figure {{
  margin: 14pt 0;
  text-align: center;
  page-break-inside: avoid;
  border: none;
  padding: 0;
}}
.plot-figure svg {{
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0 auto;
}}
.plot-caption {{
  display: block;
  text-align: center;
  font-size: 10pt;
  color: #111111;
  margin-top: 5pt;
  font-style: italic;
  line-height: 1.4;
}}
.plot-error {{
  padding: 8pt 12pt;
  background: #fff0f0;
  border-left: 3px solid #d73a49;
  font-family: 'Fira Code', monospace;
  font-size: 9pt;
  color: #901633;
  margin: 10pt 0;
}}

/* ── Multi-column layout ────────────────────────────────── */
.columns-layout {{
  column-gap: 8mm;
  text-align: justify;
  hyphens: auto;
  -webkit-hyphens: auto;
  word-break: normal;
  overflow-wrap: break-word;
  text-wrap: pretty;
  orphans: 3;
  widows: 3;
}}
.columns-layout h2,
.columns-layout h3,
.columns-layout h4 {{
  text-align: left !important;
  text-wrap: balance;
  column-span: none;
  break-after: avoid;
  page-break-after: avoid;
}}
.columns-layout figure,
.columns-layout .plot-figure,
.columns-layout .mermaid,
.columns-layout table,
.columns-layout .theorem-block {{
  break-inside: avoid;
  page-break-inside: avoid;
}}

/* ── Page Break Utilities ───────────────────────────────── */
.page-break, .page-break-before {{
  page-break-before: always !important;
  break-before: page !important;
  display: block;
  height: 0;
  margin: 0;
  padding: 0;
}}
.page-break-after {{
  page-break-after: always !important;
  break-after: page !important;
  display: block;
  height: 0;
  margin: 0;
  padding: 0;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"[OK] HTML generated: {output_path}")


# ---------------------------------------------------------------------------
# Font embedding for Playwright header/footer
# ---------------------------------------------------------------------------

def _build_header_font_style() -> str:
    """
    Playwright header/footer templates run in a sandboxed document with no
    access to Google Fonts or the main page's CSS. The only way to use a
    custom font there is to embed it as a base64 data URI.

    This function fetches EB Garamond woff2 from Google Fonts using a modern
    browser User-Agent, caches it locally in .font_cache/, and embeds it as base64.
    """
    import base64
    import urllib.request

    cache_dir = os.path.join(DOCS_DIR, ".font_cache")
    os.makedirs(cache_dir, exist_ok=True)

    css_url = "https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap"
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    face_rules = []
    try:
        req = urllib.request.Request(css_url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=10) as r:
            css = r.read().decode("utf-8")

        # Parse each @font-face block that covers the latin unicode range
        blocks = re.findall(r"@font-face\s*\{([^}]+)\}", css)
        for b in blocks:
            if "U+0000-00FF" in b or "latin" in b:
                style_m = re.search(r"font-style:\s*([^;]+);", b)
                weight_m = re.search(r"font-weight:\s*([^;]+);", b)
                url_m = re.search(r"url\((https://fonts\.gstatic\.com/[^)]+\.woff2)\)", b)

                if style_m and weight_m and url_m:
                    f_style = style_m.group(1).strip()
                    f_weight = weight_m.group(1).strip()
                    f_url = url_m.group(1).strip()
                    cache_name = f"eb-garamond-{f_style}-{f_weight}.woff2"
                    cache_path = os.path.join(cache_dir, cache_name)

                    if not os.path.exists(cache_path):
                        print(f"[*] Downloading EB Garamond ({f_style} {f_weight}): {f_url}")
                        f_req = urllib.request.Request(f_url, headers={"User-Agent": ua})
                        with urllib.request.urlopen(f_req, timeout=15) as resp:
                            f_data = resp.read()
                        with open(cache_path, "wb") as fh:
                            fh.write(f_data)
                        print(f"[OK] Font cached: {cache_path}")

                    try:
                        with open(cache_path, "rb") as fh:
                            b64 = base64.b64encode(fh.read()).decode("ascii")
                        face_rules.append(
                            f"@font-face {{ font-family: 'EB Garamond'; font-style: {f_style}; "
                            f"font-weight: {f_weight}; "
                            f"src: url('data:font/woff2;base64,{b64}') format('woff2'); }}"
                        )
                    except OSError:
                        pass
    except Exception as exc:
        print(f"[WARN] Error setting up header font embedding: {exc}")

    if not face_rules:
        return ""
    return f"<style>{''.join(face_rules)}</style>"


# ---------------------------------------------------------------------------
# Playwright PDF renderer
# ---------------------------------------------------------------------------

# Build once at module load — download is cached after the first run
_HEADER_FONT_STYLE: str = _build_header_font_style()

def convert_to_pdf(html_path: str = TEMP_HTML_PATH,
                   pdf_path: str = OUTPUT_PDF_PATH,
                   cleanup: bool = True) -> None:

    print("[*] Starting Playwright PDF generation...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 794, "height": 1123},
            device_scale_factor=2,
        )

        file_uri = "file:///" + os.path.abspath(html_path).replace("\\", "/")
        page.goto(file_uri, wait_until="networkidle")

        # Wait for MathJax
        try:
            page.wait_for_function("window.__mathJaxDone === true", timeout=20000)
            print("[OK] MathJax typesetting complete.")
        except Exception:
            print("[WARN] MathJax timeout — proceeding.")

        # Wait for Mermaid
        try:
            page.wait_for_function("window.__mermaidDone === true", timeout=15000)
            print("[OK] Mermaid diagrams rendered.")
        except Exception:
            print("[WARN] Mermaid timeout — proceeding.")

        # Extra settle time for fonts and SVG layout
        page.wait_for_timeout(1500)

        # Dynamic verification: scale down any equations that exceed column width
        page.evaluate("""() => {
            document.querySelectorAll('mjx-container[jax="CHTML"][display="true"]').forEach(container => {
                const parent = container.parentElement;
                if (!parent) return;
                const parentWidth = parent.clientWidth || parent.getBoundingClientRect().width;
                if (!parentWidth || parentWidth <= 0) return;

                container.style.fontSize = '100%';
                const mathEl = container.querySelector('mjx-math') || container;
                const mathWidth = mathEl.scrollWidth || mathEl.getBoundingClientRect().width;

                if (mathWidth > parentWidth - 4) {
                    const scale = Math.min(1.0, (parentWidth - 6) / mathWidth);
                    container.style.setProperty('font-size', (scale * 100).toFixed(1) + '%', 'important');
                    container.style.maxWidth = '100%';
                }
            });
        }""")

        page.pdf(
            path=pdf_path,
            width="210mm",
            height="297mm",
            landscape=False,
            prefer_css_page_size=False,
            print_background=True,
            margin={
                "top":    "22mm",
                "bottom": "22mm",
                "left":   "20mm",
                "right":  "20mm",
            },
            display_header_footer=True,
            header_template=f"""
            {_HEADER_FONT_STYLE}
            <div style="width:100%; font-family:'EB Garamond',Georgia,serif;
                        font-size:8pt; color:#777; display:flex;
                        justify-content:space-between; padding:0 20mm; margin:0;
                        box-sizing:border-box;">
              <span>ENSy &#8212; Embedding Navigation System</span>
              <span>Alexandre Santos &#8212; UFMA</span>
            </div>""",
            footer_template=f"""
            {_HEADER_FONT_STYLE}
            <div style="width:100%; font-family:'EB Garamond',Georgia,serif;
                        font-size:9pt; color:#777; text-align:center; padding:0 20mm; margin:0;
                        box-sizing:border-box;">
              <span class="pageNumber"></span>
              <span style="color:#bbb;"> / </span>
              <span class="totalPages"></span>
            </div>""",
        )

        browser.close()

    # Post-process with pypdf to guarantee 100% portrait orientation across all PDF viewers
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        writer = pypdf.PdfWriter()
        modified = False
        for pg in reader.pages:
            mb = pg.mediabox
            w, h = float(mb.width), float(mb.height)
            rot = pg.get("/Rotate", 0)
            if rot != 0:
                pg.rotate(-rot)
                modified = True
            if w > h:
                pg.rotate(-90)
                modified = True
            writer.add_page(pg)
        if modified:
            with open(pdf_path, "wb") as f:
                writer.write(f)
            print("[OK] PDF orientation post-processed and normalized to portrait.")
    except Exception as e:
        print(f"[WARN] pypdf post-processing skipped: {e}")

    if cleanup and os.path.exists(html_path):
        os.remove(html_path)

    print(f"[SUCCESS] PDF created: {pdf_path}")


def watch_mode(input_path: str, output_path: str, cleanup: bool = True) -> None:
    """
    Watch for file modifications in input_path, docs directory, scripts,
    and screenshots, automatically regenerating the PDF on changes.
    """
    import time
    from datetime import datetime

    input_dir = os.path.dirname(os.path.abspath(input_path))
    temp_html = os.path.join(input_dir, "temp_docs.html")

    def get_watch_files() -> dict[str, float]:
        """Collect last modification timestamps of watched files."""
        watched = {}
        if os.path.exists(input_path):
            watched[input_path] = os.path.getmtime(input_path)
        this_script = os.path.abspath(__file__)
        if os.path.exists(this_script):
            watched[this_script] = os.path.getmtime(this_script)
        plot_script = os.path.join(input_dir, "plot_renderer.py")
        if os.path.exists(plot_script):
            watched[plot_script] = os.path.getmtime(plot_script)
        screens_dir = os.path.join(input_dir, "screenshots")
        if os.path.exists(screens_dir):
            for root, _, files in os.walk(screens_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        watched[fp] = os.path.getmtime(fp)
                    except OSError:
                        pass
        return watched

    print(f"\n[WATCH] Starting watch mode for: {input_path}")
    print("[WATCH] Monitoring changes in markdown, scripts, and screenshots...")
    print("[WATCH] Press Ctrl+C to stop.\n")

    # Initial build
    try:
        build_html(input_path=input_path, output_path=temp_html)
        convert_to_pdf(html_path=temp_html, pdf_path=output_path, cleanup=cleanup)
    except Exception as exc:
        print(f"[ERROR] Initial compilation failed: {exc}")

    last_snapshot = get_watch_files()

    try:
        while True:
            time.sleep(1.0)
            current_snapshot = get_watch_files()
            changed_files = [
                fp for fp, mtime in current_snapshot.items()
                if fp not in last_snapshot or mtime > last_snapshot[fp]
            ]
            if changed_files:
                now_str = datetime.now().strftime("%H:%M:%S")
                changed_names = ", ".join([os.path.basename(f) for f in changed_files[:3]])
                print(f"\n[{now_str}] [WATCH] Change detected in {changed_names} — Rebuilding PDF...")
                time.sleep(0.3)  # Debounce
                try:
                    build_html(input_path=input_path, output_path=temp_html)
                    convert_to_pdf(html_path=temp_html, pdf_path=output_path, cleanup=cleanup)
                    print(f"[{now_str}] [SUCCESS] PDF updated: {output_path}")
                except Exception as exc:
                    print(f"[{now_str}] [ERROR] Rebuild failed: {exc}")
                last_snapshot = get_watch_files()
    except KeyboardInterrupt:
        print("\n[WATCH] Stopped watch mode.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an academic Markdown document to a high-quality PDF."
    )
    parser.add_argument(
        "--input", "-i",
        default=DOCS_MD_PATH,
        help="Path to the input Markdown file (default: DOCS.md)",
    )
    parser.add_argument(
        "--output", "-o",
        default=OUTPUT_PDF_PATH,
        help="Path for the output PDF file (default: DOCS.pdf)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep the intermediate HTML file for debugging",
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Watch input file and assets for changes and auto-regenerate PDF",
    )
    args = parser.parse_args()

    if args.watch:
        watch_mode(
            input_path=args.input,
            output_path=args.output,
            cleanup=not args.no_cleanup,
        )
    else:
        # Derive a temporary HTML path next to the input file
        input_dir  = os.path.dirname(os.path.abspath(args.input))
        temp_html  = os.path.join(input_dir, "temp_docs.html")

        build_html(input_path=args.input, output_path=temp_html)
        convert_to_pdf(
            html_path=temp_html,
            pdf_path=args.output,
            cleanup=not args.no_cleanup,
        )


if __name__ == "__main__":
    main()
