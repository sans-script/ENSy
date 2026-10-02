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
    required_packages = [
        ("markdown", "markdown"),
        ("pymdownx", "pymdown-extensions"),
        ("playwright", "playwright"),
        ("pypdf", "pypdf"),
        ("matplotlib", "matplotlib"),
        ("pyphen", "pyphen"),
    ]
    all_present = True
    for mod_name, _ in required_packages:
        try:
            __import__(mod_name)
        except ImportError:
            all_present = False
            break
    if all_present:
        return

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

        return (
            f'<div class="theorem-block theorem-{env}"{anchor_attr} markdown="1">\n\n'
            f'<div class="theorem-header">{label_html}{title_html}</div>\n\n'
            f'<div class="theorem-body" markdown="1">\n\n{body}{qed}\n\n</div>\n\n'
            f'</div>'
        )

    return env_pattern.sub(replace_env, md_content)


def preprocess_booktabs(md_content: str) -> str:
    """
    Wrap :::booktabs ... ::: blocks with a marker div that uses markdown='1'.
    Supports :::booktabs[wide] for full-width spanning tables.
    """
    pattern = re.compile(r':::booktabs(?:\[(?P<mode>[^\]]*)\])?\s*\n(?P<body>.*?)\n:::', re.DOTALL)

    def replace_bt(m: re.Match) -> str:
        mode = (m.group("mode") or "").strip().lower()
        inner = m.group("body").strip()
        cls = "booktabs-wrapper booktabs-wide" if mode == "wide" else "booktabs-wrapper"
        return f'<div class="{cls}" markdown="1">\n\n{inner}\n\n</div>'

    return pattern.sub(replace_bt, md_content)


def preprocess_pagebreaks(md_content: str) -> str:
    """
    Replace pagebreak and columnbreak directives with clean HTML elements.
    Supports:
      :::pagebreak, :::page-break, ---pagebreak---, <!-- pagebreak -->
      :::columnbreak, :::column-break, ---columnbreak---, <!-- columnbreak -->
    """
    # Column break directive
    md_content = re.sub(
        r'(?::::+col(?:umn)?-?break|---col(?:umn)?-?break---|<!--\s*col(?:umn)?-?break\s*-->)',
        '\n\n<div class="column-break"></div>\n\n',
        md_content,
        flags=re.IGNORECASE,
    )
    # Page break directive
    md_content = re.sub(
        r'(?::::+page-?break|---page-?break---|<!--\s*page-?break\s*-->)',
        '\n\n<div class="page-break"></div>\n\n',
        md_content,
        flags=re.IGNORECASE,
    )
    return md_content


def preprocess_references(md_content: str) -> str:
    """
    Wrap the ## References section (and everything after it) in a
    <div class="references-section" markdown="1"> block so that:
      - a page break is forced before it
      - list items are rendered without bullet points
      - text is justified with hanging indent (academic style)
    """
    marker = '\n## References'
    idx = md_content.rfind(marker)
    if idx == -1:
        # try at the very start of the string
        if md_content.startswith('## References'):
            idx = 0
            before = ''
            refs_block = md_content
        else:
            return md_content
    else:
        before = md_content[:idx]
        refs_block = md_content[idx:].lstrip('\n')

    wrapped = (
        '\n<div class="references-section" markdown="1">\n\n'
        + refs_block
        + '\n\n</div>\n'
    )
    return before + wrapped


# Unique tokens that Python-Markdown will wrap in <p> tags, preserving
# surrounding Markdown (headings, tables, images) completely intact.
_COL_OPEN_TOKEN  = 'XXCOLUMNOPENXX'
_COL_CLOSE_TOKEN = 'XXCOLUMNCLOSEXX'


def link_citations(html: str) -> str:
    """
    Turn IEEE numbered citations into links to the reference list.

    Reference entries are paragraphs after the "References" heading that start
    with "[n]"; each gets id="ref-n". In-text citations "[n]" and "[n, p. 12]"
    become <a href="#ref-n">; intervals such as "[0, 1]" are left alone, as are
    code, math and the entries themselves.
    """
    m = re.search(r'<h2[^>]*>\s*References\s*</h2>', html)
    if not m:
        return html
    body, refs = html[:m.start()], html[m.start():]
    refs = re.sub(r'<p>\[(\d+)\]', r'<p class="reference" id="ref-\1">[\1]', refs)
    known = set(re.findall(r'id="ref-(\d+)"', refs))
    cite = re.compile(r'\[(\d+)((?:,\s*[^\]\d\s][^\]<]*)?)\]')

    def link(mm: re.Match) -> str:
        n = mm.group(1)
        if n not in known:
            return mm.group(0)
        return f'<a class="cite" href="#ref-{n}">[{n}{mm.group(2)}]</a>'

    protected = re.compile(
        r'(<pre.*?</pre>|<code.*?</code>|<span class="arithmatex">.*?</span>'
        r'|<div class="arithmatex">.*?</div>|<[^>]+>)', re.S)
    parts = protected.split(body)
    for i in range(0, len(parts), 2):
        parts[i] = cite.sub(link, parts[i])
    return "".join(parts) + refs


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

    def column_div(attrs: dict) -> str:
        count  = attrs.get('count', '2')
        space  = attrs.get('space', '8')          # mm
        sep    = attrs.get('separate', 'false').lower() == 'true'
        crule  = 'column-rule: 0.5px solid #cccccc;' if sep else ''
        return (
            f'<div class="columns-layout" '
            f'style="column-count:{count}; column-gap:{space}mm; {crule}">'
        )

    # Walk the markers in order so each close knows whether its open was mode=auto;
    # auto regions are left as comments and laid out at render time (layout_auto_columns).
    stack: list[bool] = []

    def replace(m: re.Match) -> str:
        if m.group('open') is not None:
            attrs = parse_attrs(m.group('open'))
            auto = attrs.get('mode', '').lower() == 'auto'
            stack.append(auto)
            if auto:
                return f'<!--AUTOCOL-OPEN {column_div(attrs)}-->'
            return column_div(attrs)
        auto = stack.pop() if stack else False
        return _AUTO_CLOSE if auto else '</div>'

    token_pat = re.compile(
        r'(?:' + open_pat.pattern.replace(r'(.*?)', r'(?P<open>.*?)') + r')|(?:' + close_pat.pattern + r')',
        re.DOTALL,
    )
    return token_pat.sub(replace, html)


# ── Automatic column layout ────────────────────────────────────────────────
# Inside ---columns{count=2 mode=auto}--- ... ---/columns--- the document is split
# at every h2/h3. A run of consecutive sections without figures, tables or code
# is set in columns when it is long enough; everything else uses the full text
# width. The rendered PDF is then measured, and a run whose right column is left
# empty on some page, or which ends in a short balanced tail followed by blank
# space, is set full width instead. This repeats until the layout is stable.
_AUTO_OPEN_RE = re.compile(r'<!--AUTOCOL-OPEN (.*?)-->(.*?)<!--AUTOCOL-CLOSE-->', re.DOTALL)
_AUTO_CLOSE = '<!--AUTOCOL-CLOSE-->'
AUTO_MIN_WORDS = 600            # shorter text runs stay full width
AUTO_MAX_PASSES = 6
_AUTO_SPANNER = re.compile(
    r'<figure|<img|<table|<pre|<svg|class="[^"]*\b(?:academic-figure|plot-figure|mermaid|'
    r'booktabs-wrapper|pagebreak|page-break-before|page-break-after)\b')
_PROBE_STYLE = ('position:absolute;font-size:4px;line-height:1;color:#fefefe;'
                'white-space:nowrap;pointer-events:none')


def _auto_label(key) -> str:
    return f'ACK{key}Q'


def _with_probe(chunk: str, label: str, probe: bool) -> str:
    """Attach an invisible, layout-neutral text marker to the start of a chunk."""
    if not probe:
        return chunk
    span = f'<span style="{_PROBE_STYLE}">{label}</span>'
    m = re.match(r'\s*<(?:h[1-6]|p|div|ul|ol|blockquote)\b[^>]*>', chunk)
    if m:
        return chunk[:m.end()] + span + chunk[m.end():]
    return f'<div style="height:0;margin:0;padding:0">{span}</div>' + chunk


def _words(fragment: str) -> int:
    return len(re.sub(r'<[^>]+>', ' ', fragment).split())


def layout_auto_columns(html: str, demoted: frozenset = frozenset(), probe: bool = False):
    """
    Lay out every auto region. Returns (html, groups), where each group is
    (chunk_ids, next_label): the chunk ids set in columns and the label of the
    first marker after them (used by the measurement step).
    """
    groups: list[tuple[list[int], str]] = []
    next_id = 0
    region = 0

    def lay(m: re.Match) -> str:
        nonlocal next_id, region
        div_open, body = m.group(1), m.group(2)
        chunks = [c for c in re.split(r'(?=<h[23][\s>])', body) if c.strip()]
        ids = list(range(next_id, next_id + len(chunks)))
        next_id += len(chunks)
        end_label = _auto_label(f'E{region}')
        region += 1

        cand = [i not in demoted and not _AUTO_SPANNER.search(c) for i, c in zip(ids, chunks)]
        two = [False] * len(chunks)
        k = 0
        while k < len(chunks):
            if not cand[k]:
                k += 1
                continue
            j = k
            while j < len(chunks) and cand[j]:
                j += 1
            if sum(_words(c) for c in chunks[k:j]) >= AUTO_MIN_WORDS:
                two[k:j] = [True] * (j - k)
            k = j

        out = []
        k = 0
        while k < len(chunks):
            if not two[k]:
                out.append(_with_probe(chunks[k], _auto_label(ids[k]), probe))
                k += 1
                continue
            j = k
            while j < len(chunks) and two[j]:
                j += 1
            nxt = _auto_label(ids[j]) if j < len(chunks) else end_label
            groups.append((ids[k:j], nxt))
            out.append(div_open)
            out.extend(_with_probe(chunks[t], _auto_label(ids[t]), probe) for t in range(k, j))
            out.append('</div>')
            k = j
        if probe:
            out.append(f'<div style="height:0;margin:0;padding:0"><span style="{_PROBE_STYLE}">{end_label}</span></div>')
        return ''.join(out)

    return _AUTO_OPEN_RE.sub(lay, html), groups


def _measure_auto_columns(pdf_path: str, groups) -> set:
    """
    Return the chunk ids to set full width: those of runs whose right column is
    (nearly) empty on a page, and short runs ending in a balanced tail with blank
    space below it.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    pages = []
    for i in range(len(pdf)):
        pg = pdf[i]
        tp = pg.get_textpage()
        w, h = pg.get_size()
        rects = [tp.get_rect(k) for k in range(tp.count_rects())]
        pages.append((tp, w, h, [r for r in rects if r[3] - r[1] > 4]))
    margin = 22 / 25.4 * 72                         # top/bottom page margin, pt

    def find(label: str):
        for p, (tp, _, _, _) in enumerate(pages):
            hit = tp.search(label, match_case=True).get_next()
            if hit:
                return p, tp.get_charbox(hit[0])[3]
        return None

    demote: set = set()
    for ids, nxt in groups:
        start, end = find(_auto_label(ids[0])), find(nxt)
        if not start or not end:
            continue
        (p0, y0), (p1, y1) = start, end
        for p in range(p0, p1 + 1):
            _, w, h, rects = pages[p]
            top = y0 + 2 if p == p0 else h - margin
            bottom = y1 if p == p1 else margin
            if top - bottom < 30:
                continue
            band = [r for r in rects if bottom < (r[1] + r[3]) / 2 < top]
            mid = w / 2
            left = sum(r[2] < mid - 6 for r in band)
            right = sum(r[0] > mid + 6 for r in band)
            if left >= 6 and right < 0.3 * left:
                # this page and everything after it, plus the section flowing into it
                starts = {i: find(_auto_label(i)) for i in ids}
                before = [i for i in ids if starts[i] and starts[i][0] < p]
                keep = set(before[:-1])
                demote |= set(ids) - keep
                print(f"[AUTO] p. {p + 1}: right column empty — sections from #{ids[0] + len(keep)} set full width")
                break
        else:
            # short run whose last page holds only a balanced tail and blank space
            _, w, h, rects = pages[p1]
            tail = (h - margin) - y1 if p1 > p0 else y0 - y1
            below = [r for r in rects if margin + 4 < (r[1] + r[3]) / 2 < y1 - 6]
            if p1 - p0 <= 1 and p1 > p0 and tail < 0.4 * (h - 2 * margin) and not below:
                demote |= set(ids)
                print(f"[AUTO] p. {p1 + 1}: short two-column tail above blank space — run set full width")
    for tp, *_ in pages:   # release the file (Windows keeps it locked otherwise)
        tp.close()
    pdf.close()
    return demote


def preprocess_figures(md_content: str) -> str:
    """
    Find Markdown images followed by italic captions:
      ![Alt text](path)
      _Figure X: Caption text..._
    and transform them into structured IEEE/ACM <figure> elements.
    Supports captions containing internal underscores or variable names.
    """
    pattern = re.compile(
        r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)\s*\n+_(?:Figure|Fig\.?)\s*(?P<num>\d+)\s*:\s*(?P<caption>.+?)_\s*(?=\n\n|\n[#!\-\*\d]|\Z)',
        re.MULTILINE | re.DOTALL
    )

    def replace_fig(m: re.Match) -> str:
        alt = m.group("alt")
        src = m.group("src")
        num = m.group("num")
        cap = m.group("caption").strip().replace('\n', ' ')
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
    """Use standard flexible spacing around em-dashes to avoid non-breaking justification rivers."""
    parts = re.split(
        r'(<(?:pre|code|script|span\s+class="arithmatex")[^>]*>.*?</(?:pre|code|script|span)>)',
        html,
        flags=re.DOTALL,
    )
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'\s*—\s*', ' — ', parts[i])
    return ''.join(parts)


def fix_inline_code_breaks(html: str) -> str:
    """
    Insert <wbr> break opportunities inside inline <code> tags (not in <pre>)
    only for long identifiers (> 14 chars) like method chains or URLs,
    so short terms like `min_dist` or `n_neighbors` never break mid-word.
    Protects HTML entities (e.g. &amp;, &lt;, &#123;) from being broken by <wbr>.
    """
    def inject_wbr(m: re.Match) -> str:
        code_text = m.group(1)
        if len(code_text) > 14:
            # Add <wbr> after symbols: / . _ - : { } ( ) , = ? (and & only when not part of an HTML entity)
            code_text = re.sub(r'([/._\-:{}(),?=]|&(?!#?[a-zA-Z0-9]+;))', r'\1<wbr>', code_text)
            # Add <wbr> at camelCase transitions
            code_text = re.sub(r'([a-z0-9])([A-Z])', r'\1<wbr>\2', code_text)
        return f'<code>{code_text}</code>'

    parts = re.split(r'(<pre[^>]*>.*?</pre>)', html, flags=re.DOTALL)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'<code>(.*?)</code>', inject_wbr, parts[i], flags=re.DOTALL)
    return ''.join(parts)


def postprocess_heading_hyphens(html: str) -> str:
    """
    Editorial standard: headings (h1..h6) do NOT undergo hyphenation (syllable breaking).
    Protects digit and prefix compounds (e.g. '6-DOF', '3-D', 't-SNE', 't-Distributed')
    with non-breaking hyphens (&#8209;) so hyphenated prefixes are never orphaned.
    """
    def fix_hyphens_in_heading(m: re.Match) -> str:
        tag     = m.group(1)   # h1..h6
        attrs   = m.group(2)   # e.g. ' id="..."'
        content = m.group(3)
        segments = re.split(
            r'(<(?:code|span[^>]*)>.*?</(?:code|span)>|<mjx-container[^>]*>.*?</mjx-container>|<[^>]+>)',
            content, flags=re.DOTALL
        )
        fixed = []
        for j, seg in enumerate(segments):
            if j % 2 == 1:
                fixed.append(seg)
            else:
                # Protect digit-dash-letter (e.g. 6-DOF, 3-D)
                seg = re.sub(r'(\d)-([A-Za-z])', r'\1&#8209;\2', seg)
                # Protect single-letter-dash (e.g. t-SNE, t-Distributed)
                seg = re.sub(r'\b([A-Za-z])-([A-Za-z])', r'\1&#8209;\2', seg)
                fixed.append(seg)
        return f'<{tag}{attrs}>{"" .join(fixed)}</{tag}>'

    return re.sub(
        r'<(h[1-6])([^>]*)>(.*?)</\1>',
        fix_hyphens_in_heading,
        html,
        flags=re.DOTALL,
    )

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
    md_content = preprocess_references(md_content)
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
    html_body = postprocess_heading_hyphens(html_body)

    # Post-process: inject column divs now that Markdown is already HTML
    html_body = postprocess_columns(html_body)

    # Post-process: clickable [n] citations
    html_body = link_citations(html_body)

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
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
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
              const scale = Math.min(1.0, Math.max(0.70, (parentWidth - 6) / mathWidth));
              container.style.setProperty('font-size', (scale * 100).toFixed(1) + '%', 'important');
              container.style.maxWidth = '100%';
              container.style.overflowX = 'hidden';
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
  text-wrap: pretty;
  margin-top: 20pt;
  margin-bottom: 6pt;
  border-bottom: none;
  padding-bottom: 0;
  page-break-after: avoid;
  break-after: avoid;
  hyphens: none !important;
  -webkit-hyphens: none !important;
}}
h3 {{
  font-size: 11.5pt;
  font-weight: 700;
  color: #222222;
  text-align: left !important;
  text-wrap: pretty;
  margin-top: 14pt;
  margin-bottom: 5pt;
  page-break-after: avoid;
  break-after: avoid;
  hyphens: none !important;
  -webkit-hyphens: none !important;
}}
h4 {{
  font-size: 11pt;
  font-weight: 600;
  color: #333333;
  text-align: left !important;
  text-wrap: pretty;
  margin-top: 10pt;
  margin-bottom: 4pt;
  page-break-after: avoid;
  break-after: avoid;
  hyphens: none !important;
  -webkit-hyphens: none !important;
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
/* ── Academic Booktabs tables (Applied to all tables) ─────── */
table,
.booktabs-wrapper table {{
  width: 100%;
  border-collapse: collapse;
  margin: 10pt 0;
  font-size: 8.5pt;
  font-variant-numeric: tabular-nums;
  page-break-inside: avoid;
  border-top: 1.8px solid #111111;
  border-bottom: 1.8px solid #111111;
}}
th, td,
.booktabs-wrapper th,
.booktabs-wrapper td {{
  border: none !important;
  border-bottom: none !important;
  padding: 4pt 6pt;
  text-align: left;
  background: transparent !important;
}}
th,
.booktabs-wrapper th {{
  font-weight: 600;
  border-bottom: 1px solid #111111 !important;
  background: transparent !important;
  color: #111111;
}}
tr:nth-child(even),
.booktabs-wrapper tr:nth-child(even) {{ background: transparent !important; }}

/* ── Code ──────────────────────────────────────────── */
pre, code {{
  font-family: 'Fira Code', 'Courier New', monospace;
  font-variant-ligatures: none !important;
  font-feature-settings: "liga" 0, "calt" 0 !important;
  font-weight: 400 !important;
}}

/* ONLY inline code — zero vertical expansion to maintain exact font baseline rhythm */
code:not(pre code),
p code,
li code,
td code,
th code,
span code,
figcaption code,
a code {{
  font-family: 'Fira Code', 'Courier New', monospace;
  font-size: 8pt !important;
  font-weight: 400 !important;
  line-height: inherit !important;
  vertical-align: baseline !important;
  display: inline !important;
  padding: 0 2.5pt !important;
  margin: 0 !important;
  border: none !important;
  border-radius: 2px !important;
  background: #f0f0f2 !important;
  color: #2e3440 !important;
  white-space: nowrap !important;
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
}}

/* Block code containers and everything inside pre: completely transparent and borderless */
pre,
pre code,
pre span,
pre * {{
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  color: #24292e !important;
  font-size: 7.8pt !important;
  line-height: 1.35 !important;
  font-weight: 400 !important;
  white-space: pre-wrap !important;
  word-break: break-word !important;
  overflow-wrap: break-word !important;
}}

pre {{
  margin: 6pt 0 !important;
  padding: 4pt 0 !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
  page-break-inside: avoid;
}}

/* Syntax highlighting — transparent bg to inherit from pre */
.codehilite {{ background: transparent; }}
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
a.cite {{
  color: inherit;
  text-decoration: none;
}}
p.reference {{
  scroll-margin-top: 10mm;
}}
/* ── Multi-column layout ────────────────────────────────── */
.columns-layout {{
  column-gap: 8mm;
  column-fill: balance !important;
  text-align: justify;
  hyphens: auto;
  -webkit-hyphens: auto;
  word-break: normal;
  overflow-wrap: break-word;
  text-wrap: pretty;
}}

/* Top alignment: eliminate top margin on first element to align both columns flush to the top */
.columns-layout > *:first-child,
.columns-layout > h2:first-child,
.columns-layout > h3:first-child,
.columns-layout > p:first-child {{
  margin-top: 0 !important;
  padding-top: 0 !important;
}}

/* White-river mitigation & exact baseline alignment */
.columns-layout p,
.columns-layout li {{
  line-height: 14pt !important;
  margin-top: 0 !important;
  margin-bottom: 6pt !important;
  word-spacing: -0.03em;
  hyphenate-limit-chars: 5 2 3;
  -webkit-hyphenate-limit-before: 2;
  -webkit-hyphenate-limit-after: 3;
  orphans: 2 !important;
  widows: 2 !important;
}}

.columns-layout h1,
.columns-layout h2,
.columns-layout h3,
.columns-layout h4,
.columns-layout h5,
.columns-layout h6 {{
  text-align: left !important;
  text-wrap: pretty !important;
  hyphens: none !important;
  -webkit-hyphens: none !important;
  word-break: normal !important;
  overflow-wrap: normal !important;
  column-span: none;
  break-before: auto !important;
  page-break-before: auto !important;
  break-after: avoid;
  page-break-after: avoid;
  margin-top: 8pt !important;
  margin-bottom: 3pt !important;
  line-height: 1.25 !important;
}}

/* Headings or paragraphs immediately following a full-width spanning block */
.columns-layout .booktabs-wide + h2,
.columns-layout .booktabs-wide + h3,
.columns-layout .booktabs-wide + h4,
.columns-layout .booktabs-wide + p,
.columns-layout pre + h2,
.columns-layout pre + h3,
.columns-layout pre + h4,
.columns-layout .academic-figure + h2,
.columns-layout .academic-figure + h3,
.columns-layout .academic-figure + h4,
.columns-layout .mermaid + h2,
.columns-layout .mermaid + h3,
.columns-layout .mermaid + h4 {{
  margin-top: 0 !important;
  padding-top: 0 !important;
}}

/* Standard small tables and code snippets stay in-column */
.columns-layout table,
.columns-layout .booktabs-wrapper,
.columns-layout .booktabs-wrapper table,
.columns-layout pre,
.columns-layout .codehilite {{
  column-span: none !important;
  width: 100% !important;
  max-width: 100% !important;
  break-inside: avoid;
  page-break-inside: avoid;
  margin-top: 6pt;
  margin-bottom: 6pt;
}}

/* --- Elements that must span both columns (academic full-width) --- */
.columns-layout .booktabs-wide,
.columns-layout .booktabs-wide table,
.columns-layout table.wide-table,
.columns-layout .academic-figure,
.columns-layout figure,
.columns-layout .plot-figure,
.columns-layout .mermaid {{
  /* !important: wide tables also carry .booktabs-wrapper, whose in-column rule
     above sets column-span: none !important */
  column-span: all !important;
  break-inside: avoid;
  page-break-inside: avoid;
  margin-top: 12pt;
  margin-bottom: 12pt;
}}

/* Theorem blocks stay within one column */
.columns-layout .theorem-block {{
  break-inside: avoid;
  page-break-inside: avoid;
}}

/* Display equations in narrow columns: prevent overflow clipping and scale smoothly */
.columns-layout mjx-container[jax="CHTML"][display="true"] {{
  max-width: 100% !important;
  overflow-x: hidden !important;
  overflow-y: visible !important;
  font-size: 92% !important;
  margin: 5pt 0 !important;
  display: block !important;
  clear: both !important;
}}

/* Engine-level handling: when equations, code, or tables directly follow a heading */
.columns-layout h2 + .arithmatex,
.columns-layout h3 + .arithmatex,
.columns-layout h4 + .arithmatex,
.columns-layout h2 + mjx-container[display="true"],
.columns-layout h3 + mjx-container[display="true"],
.columns-layout h4 + mjx-container[display="true"],
.columns-layout h2 + div.arithmatex,
.columns-layout h3 + div.arithmatex,
.columns-layout h4 + div.arithmatex {{
  display: block !important;
  clear: both !important;
  margin-top: 8pt !important;
  margin-bottom: 6pt !important;
}}

.columns-layout h2 + pre,
.columns-layout h3 + pre,
.columns-layout h4 + pre,
.columns-layout h2 + table,
.columns-layout h3 + table,
.columns-layout h4 + table,
.columns-layout h2 + ul,
.columns-layout h3 + ul,
.columns-layout h4 + ul {{
  margin-top: 6pt !important;
}}

/* ── Column & Page Break Directives ─────────────────────── */
.column-break,
.columnbreak,
.col-break {{
  break-before: column !important;
  -webkit-column-break-before: always !important;
  display: block;
  height: 0;
  margin: 0;
  padding: 0;
}}

.page-break,
.pagebreak,
.page-break-before {{
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

/* ── References section ──────────────────────────────────────── */
.references-section {{
  page-break-before: always !important;
  break-before: page !important;
  column-count: 2;
  column-gap: 8mm;
  text-align: justify;
  text-justify: inter-word;
  hyphens: auto;
  -webkit-hyphens: auto;
}}
.references-section h2 {{
  column-span: all;
  text-align: left !important;
  margin-top: 0 !important;
  margin-bottom: 12pt !important;
  page-break-after: avoid;
  break-after: avoid;
}}
.references-section ul {{
  list-style: none !important;
  list-style-type: none !important;
  padding: 0 !important;
  margin: 0 !important;
}}
.references-section p,
.references-section li {{
  text-align: justify !important;
  text-justify: inter-word !important;
  hyphens: auto !important;
  -webkit-hyphens: auto !important;
  word-break: normal;
  overflow-wrap: break-word;
  text-wrap: pretty;
  margin-top: 0 !important;
  margin-bottom: 7pt !important;
  padding-left: 0 !important;
  text-indent: 0 !important;
  font-size: 9.5pt !important;
  line-height: 1.45 !important;
  break-inside: avoid;
  page-break-inside: avoid;
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

_HEADER_FONT_STYLE_CACHE: str | None = None

def get_header_font_style() -> str:
    """Lazy evaluation and caching of EB Garamond header/footer font embedding."""
    global _HEADER_FONT_STYLE_CACHE
    if _HEADER_FONT_STYLE_CACHE is None:
        _HEADER_FONT_STYLE_CACHE = _build_header_font_style()
    return _HEADER_FONT_STYLE_CACHE


def convert_to_pdf(html_path: str = TEMP_HTML_PATH,
                   pdf_path: str = OUTPUT_PDF_PATH,
                   cleanup: bool = True) -> None:

    print("[*] Starting Playwright PDF generation...")
    header_font_style = get_header_font_style()
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:
            # Bundled Chromium not installed: fall back to an installed Edge/Chrome.
            for channel in ("msedge", "chrome"):
                try:
                    browser = p.chromium.launch(channel=channel)
                    break
                except Exception:
                    continue
            else:
                raise
        page = browser.new_page(
            viewport={"width": 794, "height": 1123},
            device_scale_factor=2,
        )

        file_uri = "file:///" + os.path.abspath(html_path).replace("\\", "/")
        page.goto(file_uri, wait_until="domcontentloaded", timeout=60000)

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
            {header_font_style}
            <div style="width:100%; font-family:'EB Garamond',Georgia,serif;
                        font-size:8pt; color:#777; display:flex;
                        justify-content:space-between; padding:0 20mm; margin:0;
                        box-sizing:border-box;">
              <span>ENSy &#8212; Embedding Navigation System</span>
              <span>Alexandre Santos</span>
            </div>""",
            footer_template=f"""
            {header_font_style}
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


def render_pdf(html_path: str = TEMP_HTML_PATH,
               pdf_path: str = OUTPUT_PDF_PATH,
               cleanup: bool = True) -> None:
    """
    convert_to_pdf() plus the automatic column layout: draft renders with
    invisible markers are measured until no column run needs to be set full
    width, then the final PDF is rendered without markers.
    """
    with open(html_path, encoding="utf-8") as f:
        source = f.read()
    if not _AUTO_OPEN_RE.search(source):
        convert_to_pdf(html_path, pdf_path, cleanup)
        return

    demoted: frozenset = frozenset()
    try:
        import pypdfium2  # noqa: F401  (measurement needs it)
        measurable = True
    except ImportError:
        print("[WARN] pypdfium2 not installed — automatic columns are not verified.")
        measurable = False

    probe_html = html_path + ".probe.html"
    probe_pdf = pdf_path + ".probe.pdf"
    for n in range(AUTO_MAX_PASSES if measurable else 0):
        html, groups = layout_auto_columns(source, demoted, probe=True)
        with open(probe_html, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[AUTO] layout pass {n + 1}: {len(groups)} two-column run(s)")
        convert_to_pdf(probe_html, probe_pdf, cleanup=True)
        more = _measure_auto_columns(probe_pdf, groups)
        if not more - demoted:
            break
        demoted = demoted | frozenset(more)
    if os.path.exists(probe_pdf):
        os.remove(probe_pdf)

    html, groups = layout_auto_columns(source, demoted, probe=False)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[AUTO] final layout: {len(groups)} two-column run(s)")
    convert_to_pdf(html_path, pdf_path, cleanup)


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
        render_pdf(html_path=temp_html, pdf_path=output_path, cleanup=cleanup)
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
                    render_pdf(html_path=temp_html, pdf_path=output_path, cleanup=cleanup)
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
        render_pdf(
            html_path=temp_html,
            pdf_path=args.output,
            cleanup=not args.no_cleanup,
        )


if __name__ == "__main__":
    main()
