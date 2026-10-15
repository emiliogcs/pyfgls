#!/usr/bin/env python3
"""Build the pyfgls page of emiliogcs.com (https://emiliogcs.com/code/).

The page is assembled from this repository's README.md and docs/*.md, so the
documentation keeps one source. Markdown is rendered with markdown-it-py; math is
left for MathJax as \\( \\) and \\[ \\]; code is highlighted by Pygments with CSS
classes (the site's stylesheet colours them); every heading is an anchor; the
table of contents is generated. The Quickstart examples are run and their real
output replaces the output blocks, and the API reference is generated from the
docstrings with griffe (the parser behind mkdocstrings), following docs/api.md.

    python scripts/build_site_page.py --site ../emiliogcs.github.io  # update the site's code/
    python scripts/build_site_page.py --out build/site               # write the fragment only
    python scripts/build_site_page.py --update-readme                # README outputs from a real run

Needs the ``site`` and ``formula`` extras (``pip install -e ".[site,formula]"``).
The site's code/index.html holds the page shell (head, menu, footer); the
generated part goes between ``<!-- pyfgls:start -->`` and ``<!-- pyfgls:end -->``.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.dollarmath import dollarmath_plugin
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/emiliogcs/pyfgls"
BLOB = f"{REPO}/blob/main/"
RAW = "https://raw.githubusercontent.com/emiliogcs/pyfgls/main/"
PYPI = "https://pypi.org/project/pyfgls/"
START, END = "<!-- pyfgls:start -->", "<!-- pyfgls:end -->"

# Labels shown above code blocks, and the Pygments lexer for each fence language.
LANGS = {
    "python": ("python", "python"),
    "py": ("python", "python"),
    "pycon": ("python", "pycon"),
    "bash": ("shell", "bash"),
    "sh": ("shell", "bash"),
    "text": ("output", None),
    "bibtex": ("bibtex", "bibtex"),
}


# ----------------------------------------------------------------------------- sources
def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def sections(text: str, level: int) -> dict[str, str]:
    """Split Markdown into {heading: body} at the given heading level (fences respected)."""
    out: dict[str, str] = {}
    name, buf, fence = None, [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("```"):
            fence = not fence
        m = None if fence else re.match(rf"^{'#' * level} (.+?)\s*$", line)
        if m or (not fence and re.match(rf"^#{{1,{level - 1}}} ", line)):
            if name is not None:
                out[name] = "".join(buf)
            name, buf = (m.group(1) if m else None), []
            continue
        if name is not None:
            buf.append(line)
    if name is not None:
        out[name] = "".join(buf)
    return out


def pick(secs: dict[str, str], start: str) -> tuple[str, str]:
    """The section whose heading starts with ``start``: (heading, body)."""
    hits = [(k, v) for k, v in secs.items() if k.startswith(start)]
    if len(hits) != 1:
        raise SystemExit(f"Expected one section starting with {start!r}, found {len(hits)}.")
    return hits[0]


def tidy(body: str) -> str:
    """Drop rules between sections and the snippet markers mkdocs uses."""
    body = re.sub(r"^<!-- --8<-- .*?-->\n", "", body, flags=re.M)
    return re.sub(r"(\n---\s*)+\s*$", "\n", body.strip("\n")) + "\n"


# --------------------------------------------------------------------------- examples
FENCE_PAIR = re.compile(
    r"(?P<head>```python\n)(?P<code>.*?)(?P<mid>\n```\n\s*\n```text\n)(?P<out>.*?)(?P<tail>```)",
    re.S,
)


def run_example(code: str) -> str:
    """Run one example in a fresh interpreter; return what it prints."""
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0", "PYTHONWARNINGS": "ignore"}
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env, check=False
    )
    if proc.returncode:
        raise SystemExit(f"Example failed:\n{code}\n\n{proc.stderr}")
    lines = [line.rstrip() for line in proc.stdout.rstrip("\n").splitlines()]
    return "\n".join(lines) + "\n"


def run_examples(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Replace every ```text block that follows a ```python block with the real output.

    Returns the new text and (code, old output, new output) per example.
    """
    runs: list[tuple[str, str, str]] = []

    def sub(m: re.Match[str]) -> str:
        new = run_example(m.group("code"))
        runs.append((m.group("code"), m.group("out"), new))
        return m.group("head") + m.group("code") + m.group("mid") + new + m.group("tail")

    return FENCE_PAIR.sub(sub, text), runs


# ---------------------------------------------------------------------------- slugs
def slug(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE)
    s = re.sub(r"[\s_]+", "-", s.strip())
    return re.sub(r"-{2,}", "-", s).strip("-") or "section"


def github_slug(text: str) -> str:
    """The anchor GitHub (and mkdocs' toc) gives a heading."""
    s = re.sub(r"[^\w\- ]", "", text.strip().lower(), flags=re.UNICODE)
    return s.replace(" ", "-")


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


# ----------------------------------------------------------------------------- page
@dataclass
class Chunk:
    """A piece of Markdown placed on the page."""

    source: str  # path relative to the repository, for links and anchors
    text: str
    top: int  # level of its first heading on the page
    title: str | None = None  # replaces the first heading's text
    drop_title: bool = False  # the page section heading stands in for it
    id_prefix: str = ""
    anchor: str = ""  # page id the source file maps to (set while parsing)
    tokens: list[Token] = field(default_factory=list)


@dataclass
class Section:
    id: str
    title: str
    chunks: list[Chunk] = field(default_factory=list)


class Page:
    def __init__(self) -> None:
        self.md = make_md(self)
        self.ids: set[str] = set()
        self.anchors: dict[str, str] = {}  # "README.md#slug" or "docs/x.md" -> page id
        self.toc: list[tuple[int, str, str]] = []  # (level, id, inner HTML)
        self.assets: dict[str, Path] = {}  # published name -> source file

    def new_id(self, text: str, prefix: str = "", section: str = "") -> str:
        base = (prefix + "-" if prefix else "") + slug(text)
        if base[0].isdigit():  # ids begin with a letter: "quickstart-1-serial-correlation..."
            base = f"{section or 'h'}-{base}"
        cand, n = base, 2
        while cand in self.ids:
            cand, n = f"{base}-{n}", n + 1
        self.ids.add(cand)
        return cand

    # --- pass 1: parse, place headings, register anchors
    def parse(self, chunk: Chunk) -> None:
        tokens = self.md.parse(chunk.text, {"page": self})
        levels = [int(t.tag[1]) for t in tokens if t.type == "heading_open"]
        # the first heading lands on `top`; a dropped title is the page's section heading, one up
        shift = chunk.top - (1 if chunk.drop_title else 0) - min(levels) if levels else 0
        first = True
        out: list[Token] = []
        skip = 0
        for i, t in enumerate(tokens):
            if skip:
                skip -= 1
                continue
            if t.type == "heading_open":
                inline = tokens[i + 1]
                original = plain(inline)
                if first and chunk.drop_title:
                    self.anchors[chunk.source] = chunk.anchor
                    self.anchors[f"{chunk.source}#{github_slug(original)}"] = chunk.anchor
                    first = False
                    skip = 2
                    continue
                if first and chunk.title:
                    inline.children = self.md.parseInline(chunk.title, {})[0].children
                level = min(6, int(t.tag[1]) + shift)
                t.tag = tokens[i + 2].tag = f"h{level}"
                hid = self.new_id(plain(inline), chunk.id_prefix, chunk.anchor)
                t.attrSet("id", hid)
                self.anchors[f"{chunk.source}#{github_slug(original)}"] = hid
                if first:
                    self.anchors.setdefault(chunk.source, hid)
                    first = False
                # no links inside a heading: the heading itself is the link
                inline.children = [
                    c for c in inline.children or [] if c.type not in ("link_open", "link_close")
                ]
            out.append(t)
        chunk.tokens = out

    # --- pass 2: links, then HTML
    def render(self, chunk: Chunk) -> str:
        for i, t in enumerate(chunk.tokens):
            for c in [t, *(t.children or [])]:
                if c.type == "link_open":
                    c.attrSet("href", self.href(str(c.attrGet("href")), chunk.source))
                elif c.type == "image":
                    c.attrSet("src", self.asset(str(c.attrGet("src")), chunk.source))
            if t.type == "heading_open" and int(t.tag[1]) <= 3:
                inner = self.md.renderer.render(
                    chunk.tokens[i + 1].children or [], self.md.options, {}
                )
                self.toc.append((int(t.tag[1]), str(t.attrGet("id")), inner))
        return self.md.renderer.render(chunk.tokens, self.md.options, {"page": self})

    def href(self, href: str, source: str) -> str:
        # the README links to its own files with absolute URLs (they must work on PyPI)
        own = href.startswith((BLOB, RAW))
        if own:
            href = href.removeprefix(BLOB).removeprefix(RAW)
        elif re.match(r"^[a-z]+:", href) or href.startswith("//"):
            return href
        path, _, frag = href.partition("#")
        if not path:  # in-file anchor
            return "#" + self.anchors.get(f"{source}#{frag}", frag)
        target = os.path.normpath(path if own else os.path.join(os.path.dirname(source), path))
        if target.startswith("docs/") and target != "docs/assets" and (ROOT / target).is_dir():
            target = target.rstrip("/") + "/index.md"
        if frag and f"{target}#{frag}" in self.anchors:
            return "#" + self.anchors[f"{target}#{frag}"]
        if target in self.anchors:
            return "#" + self.anchors[target]
        if target == "docs/changelog.md":
            target = "CHANGELOG.md"
        return BLOB + target + (f"#{frag}" if frag else "")

    def asset(self, src: str, source: str) -> str:
        if src.startswith(RAW):  # served from the site, not fetched from GitHub
            path = (ROOT / src.removeprefix(RAW)).resolve()
        elif re.match(r"^[a-z]+:", src):
            return src
        else:
            path = (ROOT / os.path.dirname(source) / src).resolve()
        self.assets[path.name] = path
        return path.name

    def heading(self, level: int, hid: str, inner: str, cls: str = "") -> str:
        c = f' class="{cls}"' if cls else ""
        return f'<h{level} id="{hid}"{c}><a href="#{hid}">{inner}</a></h{level}>\n'


def plain(inline: Token) -> str:
    return "".join(c.content for c in inline.children or [] if c.type in ("text", "code_inline"))


# --------------------------------------------------------------------------- markdown
def make_md(page: Page) -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    md.use(dollarmath_plugin, allow_space=True, allow_digits=False, double_inline=True)
    esc = html.escape

    def math_inline(self: Any, tokens: list[Token], i: int, options: Any, env: Any) -> str:
        return f'<span class="math">\\({esc(tokens[i].content)}\\)</span>'

    def math_block(self: Any, tokens: list[Token], i: int, options: Any, env: Any) -> str:
        return f'<div class="math">\\[{esc(tokens[i].content.strip())}\\]</div>\n'

    def fence(self: Any, tokens: list[Token], i: int, options: Any, env: Any) -> str:
        t = tokens[i]
        lang = (t.info or "text").split()[0]
        return code_block(t.content, lang)

    def heading_open(self: Any, tokens: list[Token], i: int, options: Any, env: Any) -> str:
        t = tokens[i]
        hid = t.attrGet("id")
        return f'<{t.tag} id="{hid}"><a href="#{hid}">' if hid else f"<{t.tag}>"

    def heading_close(self: Any, tokens: list[Token], i: int, options: Any, env: Any) -> str:
        t = tokens[i]
        opened = tokens[i - 2]
        return f"</a></{t.tag}>\n" if opened.attrGet("id") else f"</{t.tag}>\n"

    def html_block(self: Any, tokens: list[Token], i: int, options: Any, env: Any) -> str:
        content = tokens[i].content
        return "" if content.lstrip().startswith("<!--") else content

    def image(self: Any, tokens: list[Token], i: int, options: Any, env: Any) -> str:
        t = tokens[i]
        src = str(t.attrGet("src"))
        alt = esc(plain(t)) if t.children else ""
        size = ""
        src_path = page.assets.get(src)
        if src_path and src_path.suffix == ".png":
            w, h = png_size(src_path)
            size = f' width="{w}" height="{h}"'
        return f'<img src="{esc(src)}" alt="{alt}"{size} loading="lazy" decoding="async">'

    md.add_render_rule("math_inline", math_inline)
    md.add_render_rule("math_inline_double", math_block)
    md.add_render_rule("math_block", math_block)
    md.add_render_rule("math_block_label", math_block)
    md.add_render_rule("fence", fence)
    md.add_render_rule(
        "code_block", lambda self, tokens, i, o, e: code_block(tokens[i].content, "text")
    )
    md.add_render_rule("heading_open", heading_open)
    md.add_render_rule("heading_close", heading_close)
    md.add_render_rule("html_block", html_block)
    md.add_render_rule("image", image)
    md.add_render_rule(
        "table_open", lambda self, tokens, i, o, e: '<div class="table-wrap"><table>\n'
    )
    md.add_render_rule("table_close", lambda self, tokens, i, o, e: "</table></div>\n")
    return md


def code_block(code: str, lang: str) -> str:
    label, lexer = LANGS.get(lang, (lang or "text", None))
    if lexer:
        body = highlight(code, get_lexer_by_name(lexer), HtmlFormatter(nowrap=True))
    else:
        body = html.escape(code)
    return (
        f'<div class="code" data-lang="{label}"><div class="code-head"><span class="code-lang">{label}</span></div>'
        f"<pre><code>{body.rstrip()}</code></pre></div>\n"
    )


# ------------------------------------------------------------------------------- API
class Api:
    """The API reference from the docstrings, in the order of docs/api.md."""

    def __init__(self, page: Page) -> None:
        import logging

        import griffe

        logging.getLogger("griffe").setLevel(logging.ERROR)
        self.page = page
        self.pkg = griffe.load("pyfgls", search_paths=[str(ROOT / "src")], docstring_parser="numpy")
        self.ids: dict[str, str] = {}  # dotted path (public or canonical) -> page id
        self.groups: list[tuple[str, list[tuple[str, Any]]]] = []
        group: list[tuple[str, Any]] = []
        for line in read("docs/api.md").splitlines():
            if line.startswith("## "):
                group = []
                self.groups.append((line[3:].strip(), group))
            elif line.startswith("::: "):
                path = line[4:].strip()
                group.append((path, self.get(path)))
        for _, objs in self.groups:
            for path, obj in objs:
                self.register(obj, path)

    def get(self, path: str) -> Any:
        obj = self.pkg[path.removeprefix("pyfgls.")]
        return obj.final_target if obj.is_alias else obj

    def members(self, obj: Any) -> list[tuple[str, Any]]:
        """Public members with a docstring, in source order (as mkdocstrings shows them)."""
        out = []
        exports = None
        if obj.kind.value == "module" and obj.exports is not None:
            exports = {str(e) for e in obj.exports}
        for name, m in obj.members.items():
            if name.startswith("_"):
                continue
            if exports is not None and name not in exports:
                continue
            if m.is_alias:
                if exports is None:
                    continue
                m = m.final_target
            if m.docstring is None:
                continue
            out.append((name, m))
        return out

    def register(self, obj: Any, path: str) -> None:
        hid = path.replace(".", "-")  # pyfgls-FGLS-fit: letters, digits, dashes, underscores
        self.page.ids.add(hid)
        self.ids[path] = self.ids[obj.path] = hid
        for name, m in self.members(obj):
            self.register(m, f"{path}.{name}")

    def link(self, target: str, context: Any) -> str | None:
        target = target.lstrip("~")
        mod = context.module.path if hasattr(context, "module") else "pyfgls"
        for cand in (target, f"{mod}.{target}", f"pyfgls.{target}", f"{context.path}.{target}"):
            if cand in self.ids:
                return self.ids[cand]
        return None

    def text(self, md_text: str, context: Any) -> str:
        """Docstring Markdown to HTML: Sphinx roles become code (linked when on the page)."""

        def role(m: re.Match[str]) -> str:
            target = m.group(2)
            shown = target.lstrip("~").rsplit(".", 1)[-1] if target.startswith("~") else target
            if m.group(1) in ("func", "meth"):
                shown += "()"
            hid = self.link(target, context)
            return f"[`{shown}`](#{hid})" if hid else f"`{shown}`"

        md_text = re.sub(
            r":(?:py:)?(func|class|meth|mod|data|attr|obj|exc):`([~\w.]+)`", role, md_text
        )
        md_text = re.sub(r"::\n", ":\n", md_text)
        md_text = re.sub(r"(?<=[\w)`\"'])--(?=[\w(`\"'])", "–", md_text)  # ranges: an en dash
        return self.page.md.render(md_text, {"page": self.page})

    def signature(self, obj: Any, name: str) -> str:
        func = obj
        if obj.kind.value == "class":
            func = obj.members.get("__init__")
            if func is None or func.is_alias:
                return name
        params = []
        star = False
        seen_pos_only = False
        for p in func.parameters:
            if p.name in ("self", "cls"):
                continue
            kind = p.kind.value
            if kind == "positional-only":
                seen_pos_only = True
            elif seen_pos_only:
                params.append("/")
                seen_pos_only = False
            if kind == "keyword-only" and not star:
                params.append("*")
                star = True
            if kind == "variadic positional":
                params.append(f"*{p.name}")
                star = True
            elif kind == "variadic keyword":
                params.append(f"**{p.name}")
            else:
                params.append(p.name if p.default is None else f"{p.name}={p.default}")
        if seen_pos_only:
            params.append("/")
        one = f"{name}({', '.join(params)})"
        if len(one) <= 72:
            return one
        return f"{name}(\n" + "".join(f"    {p},\n" for p in params) + ")"

    def sections(self, obj: Any) -> str:
        out = []
        for sec in obj.docstring.parsed:
            kind = sec.kind.value
            if kind == "text":
                out.append(self.text(sec.value, obj))
            elif kind == "examples":
                out.append('<p class="api-label">examples</p>\n')
                for sub_kind, value in sec.value:
                    if sub_kind.value == "examples":
                        out.append(code_block(value + "\n", "pycon"))
                    else:
                        out.append(self.text(value, obj))
            elif isinstance(sec.value, list):
                out.append(f'<p class="api-label">{html.escape(sec.title or kind)}</p>\n')
                out.append(self.items(sec.value, obj))
            else:
                out.append(f'<p class="api-label">{html.escape(sec.title or kind)}</p>\n')
                out.append(self.text(str(getattr(sec.value, "description", sec.value)), obj))
        return "".join(out)

    def items(self, items: list[Any], obj: Any) -> str:
        """Parameters, returns, attributes... as a definition list; repeated
        descriptions (``a, b, c : ...`` in the docstring) are merged."""
        merged: list[tuple[list[str], str, str]] = []
        for it in items:
            name = getattr(it, "name", "") or ""
            ann = str(it.annotation) if getattr(it, "annotation", None) else ""
            desc = getattr(it, "description", "") or ""
            if merged and merged[-1][2] == desc and merged[-1][1] == ann and desc:
                merged[-1][0].append(name)
            else:
                merged.append(([name], ann, desc))
        rows = []
        for names, ann, desc in merged:
            if len(names) > 1:  # a shared description: the types differ, so none is shown
                ann = ""
            shown = ", ".join(f"<code>{html.escape(n)}</code>" for n in names if n)
            t = f' <span class="api-type">{html.escape(ann)}</span>' if ann else ""
            dt = (
                (shown + t)
                if shown
                else (f'<span class="api-type">{html.escape(ann)}</span>' if ann else "")
            )
            rows.append(f"<dt>{dt}</dt>\n<dd>{self.text(desc, obj) if desc else ''}</dd>\n")
        return '<dl class="api-items">\n' + "".join(rows) + "</dl>\n"

    def render_obj(self, obj: Any, path: str, level: int) -> str:
        hid = self.ids[path]
        kind = obj.kind.value
        labels = set(getattr(obj, "labels", set()))
        name = path if level <= 4 else path.rsplit(".", 1)[-1]
        if kind == "attribute" and "property" in labels:
            tag = "property"
        elif kind == "function" and "classmethod" in labels:
            tag = "classmethod"
        elif kind == "function" and level > 4 and obj.parent.kind.value == "class":
            tag = "method"
        elif kind == "class" and "dataclass" in labels:
            tag = "dataclass"
        else:
            tag = kind
        lvl = min(level, 6)
        out = [
            f'<div class="api-obj api-{tag}">\n',
            f'<h{lvl} id="{hid}" class="api-h"><a href="#{hid}"><code>{html.escape(name)}</code></a>'
            f' <span class="api-kind">{tag}</span></h{lvl}>\n',
        ]
        if kind in ("class", "function"):
            in_class = obj.parent is not None and obj.parent.kind.value == "class"
            short = path.rsplit(".", 1)[-1] if in_class else path
            sig = html.escape(self.signature(obj, short), quote=False)
            out.append(f'<div class="code sig"><pre><code>{sig}</code></pre></div>\n')
        if obj.docstring:
            out.append(self.sections(obj))
        if kind == "class":
            out.append(self.inherited(obj))
        for mname, m in self.members(obj):
            out.append(self.render_obj(m, f"{path}.{mname}", level + 1))
        out.append("</div>\n")
        return "".join(out)

    def inherited(self, obj: Any) -> str:
        try:
            bases = [b for b in obj.resolved_bases if str(b.path).startswith("pyfgls.")]
        except Exception:  # bases outside the package (scikit-learn) need not resolve
            bases = []
        names = []
        for base in bases:
            for name, m in self.members(base):
                if name not in obj.members:
                    hid = self.ids.get(m.path)
                    shown = f"{name}()" if m.kind.value == "function" else name
                    names.append(
                        f'<a href="#{hid}"><code>{shown}</code></a>'
                        if hid
                        else f"<code>{shown}</code>"
                    )
        if not names:
            return ""
        base_names = ", ".join(
            f'<a href="#{self.ids[b.path]}"><code>{b.name}</code></a>'
            if b.path in self.ids
            else f"<code>{b.name}</code>"
            for b in bases
        )
        return f'<p class="api-inherits">Inherits {", ".join(names)} from {base_names}.</p>\n'

    def html(self) -> str:
        out = []
        for title, objs in self.groups:
            gid = self.page.new_id(title, "api")
            out.append(self.page.heading(3, gid, html.escape(title)))
            self.page.toc.append((3, gid, html.escape(title)))
            for path, obj in objs:
                out.append(self.render_obj(obj, path, 4))
        return "".join(out)


# ---------------------------------------------------------------------------- build
def build(update_readme: bool = False) -> tuple[str, dict[str, Path], list[tuple[str, str, str]]]:
    readme = read("README.md")
    readme_h2 = sections(readme, 2)
    _, quick = pick(readme_h2, "Quick examples")
    quick_run, runs = run_examples(quick)
    if update_readme:
        (ROOT / "README.md").write_text(readme.replace(quick, quick_run), encoding="utf-8")
    full_h3 = sections(pick(readme_h2, "Full functionality and usage")[1], 3)

    def readme_sub(start: str, title: str | None = None) -> Chunk:
        head, body = pick(full_h3, start)
        return Chunk("README.md", f"### {head}\n\n{tidy(body)}", 3, title)

    def doc(rel: str, top: int = 3, **kw: Any) -> Chunk:
        return Chunk(rel, read(rel), top, **kw)

    quick_md = read("docs/quickstart.md")
    inference = pick(sections(quick_md, 2), "Inference options")[1]

    page = Page()
    plan = [
        Section("install", "Install", [Chunk("README.md", tidy(pick(readme_h2, "Install")[1]), 3)]),
        Section("quickstart", "Quickstart", [Chunk("README.md", tidy(quick_run), 3)]),
        Section(
            "choosing",
            "When to use which estimator",
            [
                Chunk("README.md", tidy(pick(readme_h2, "Which estimator")[1]), 3),
                Chunk("README.md", tidy(pick(readme_h2, "Guidance for practitioners")[1]), 3),
            ],
        ),
        Section(
            "guide",
            "User guide",
            [
                readme_sub("Input data", "Data requirements"),
                readme_sub("`SerialFGLS`"),
                readme_sub("`LassoFGLS`"),
                readme_sub("`FGLS`"),
                Chunk(
                    "docs/quickstart.md",
                    f"### Inference options (`cov_type`)\n\n{tidy(inference)}",
                    3,
                ),
                doc("docs/howto/reading-summary.md"),
                doc("docs/howto/nonexogeneity.md"),
                doc("docs/howto/lagged-dependent.md"),
                doc("docs/howto/many-covariates.md"),
                readme_sub("Results"),
                readme_sub("Diagnostics"),
                readme_sub("Benchmarks"),
                readme_sub("scikit-learn"),
                doc("docs/howto/reproducibility.md"),
                readme_sub("Performance"),
            ],
        ),
        Section(
            "theory",
            "Theory",
            [
                doc("docs/theory/serial.md"),
                doc("docs/theory/hetero.md"),
                doc("docs/theory/combined.md"),
                Chunk(
                    "README.md",
                    "### Implementation choices and open items\n\n"
                    + tidy(pick(readme_h2, "Implementation choices")[1]),
                    3,
                ),
            ],
        ),
        Section("api", "API reference"),
        Section("replication", "Replication", [doc("docs/replication.md", drop_title=True)]),
        Section("faq", "FAQ and troubleshooting", [doc("docs/faq.md", drop_title=True)]),
        Section(
            "citation", "Citation", [Chunk("README.md", tidy(pick(readme_h2, "Citing")[1]), 3)]
        ),
    ]
    for sec in plan:
        page.ids.add(sec.id)
    # where links to the other files and to README sections land
    page.anchors.update(
        {
            "docs/quickstart.md": "quickstart",
            "docs/guidance.md": "choosing",
            "docs/index.md": "pyfgls",
            "docs/api.md": "api",
            "README.md#install": "install",
            "README.md#quick-examples": "quickstart",
            "README.md#which-estimator-should-i-use": "choosing",
            "README.md#guidance-for-practitioners": "choosing",
            "README.md#full-functionality-and-usage": "guide",
            "README.md#what-the-estimators-do": "theory",
            "README.md#replication-of-the-papers": "replication",
            "README.md#citing": "citation",
        }
    )
    api = Api(page)
    for sec in plan:
        for ch in sec.chunks:
            ch.anchor = sec.id
            ch.id_prefix = "" if sec.id != "api" else "api"
            page.parse(ch)

    # header block: the README's title and introduction, the docs' one-line pitch
    title = re.search(r"^# (.+)$", readme, re.M).group(1)  # type: ignore[union-attr]
    intro = readme.split("\n", 1)[1].split("```python", 1)[0]
    intro = re.sub(r"^\[!\[.*\n", "", intro, flags=re.M)  # badges
    pitch = re.search(r"^\*\*(.+?)\*\*", read("docs/index.md"), re.M).group(1)  # type: ignore[union-attr]
    dois = re.findall(r"https://doi\.org/\S+?(?=\s|$)", pick(readme_h2, "Citing")[1])
    version = re.search(r'__version__ = "(.+?)"', read("src/pyfgls/_version.py")).group(1)  # type: ignore[union-attr]
    pyreq = re.search(r'requires-python = ">=\s*(.+?)"', read("pyproject.toml")).group(1)  # type: ignore[union-attr]
    license_ = re.search(r'^license = (?:\{ text = )?"(.+?)"', read("pyproject.toml"), re.M)
    intro_chunk = Chunk("README.md", tidy(intro), 3)
    page.parse(intro_chunk)

    newtab = '<span class="vh"> ({}, opens in a new tab)</span> <span aria-hidden="true">↗</span>'
    acts = [
        f'<a class="act" href="{REPO}" target="_blank" rel="noopener">github{newtab.format("source code")}</a>',
        f'<a class="act" href="{PYPI}" target="_blank" rel="noopener">pypi{newtab.format("the package on PyPI")}</a>',
        f'<a class="act" href="{dois[0]}" target="_blank" rel="noopener">paper 2026{newtab.format("Feasible GLS for Time Series Regression")}</a>',
        f'<a class="act" href="{dois[1]}" target="_blank" rel="noopener">paper 2024{newtab.format("Estimation in the Presence of Heteroskedasticity of Unknown Form")}</a>',
        '<a class="act" href="#citation">bibtex</a>',
    ]
    sep = '\n          <span class="sep" aria-hidden="true">·</span>\n          '
    note = f"version {version} · python ≥ {pyreq}" + (
        f" · {license_.group(1)} license" if license_ else ""
    )
    header = (
        '<section class="wrap grid intro code-intro" aria-labelledby="pyfgls">\n'
        '      <div class="body reading">\n'
        f'        <h1 class="statement" id="pyfgls">{html.escape(title)}</h1>\n'
        f'        <p class="pitch">{html.escape(pitch)}</p>\n'
        f'        <div class="lede">\n{page.render(intro_chunk)}</div>\n'
        f'        <div class="actions">\n          {sep.join(acts)}\n        </div>\n'
        f'        <p class="note">{html.escape(note)}</p>\n'
        "      </div>\n"
        "    </section>\n"
    )
    page.toc.clear()

    body = []
    for sec in plan:
        page.toc.append((2, sec.id, html.escape(sec.title)))
        parts = [page.heading(2, sec.id, html.escape(sec.title), "doc-h2")]
        parts += [page.render(ch) for ch in sec.chunks]
        if sec.id == "api":
            parts.append(
                '<p class="api-note">Generated from the docstrings of pyfgls '
                f"{html.escape(version)}.</p>\n"
            )
            parts.append(api.html())
        body.append(
            f'<section class="doc-section" aria-labelledby="{sec.id}">\n{"".join(parts)}</section>\n'
        )

    toc = toc_html(page.toc)
    fragment = (
        f"{START}\n"
        f"    {header}\n"
        '    <div class="wrap grid section docs">\n'
        f'      <nav class="toc toc-side" aria-label="Contents">\n        <p class="toc-title">contents</p>\n{toc}      </nav>\n'
        '      <div class="body doc">\n'
        '        <details class="toc-top">\n'
        '          <summary>contents <span class="sign" aria-hidden="true"></span></summary>\n'
        f'          <nav aria-label="Contents of this page">\n{toc}          </nav>\n'
        "        </details>\n"
        f"{''.join(body)}"
        "      </div>\n"
        "    </div>\n"
        f"    {END}"
    )
    return fragment, page.assets, runs


def toc_html(entries: list[tuple[int, str, str]]) -> str:
    out = ['        <ol class="toc-list">\n']
    open_sub = False
    for i, (level, hid, inner) in enumerate(entries):
        if level == 2:
            if open_sub:
                out.append("          </ol></li>\n")
                open_sub = False
            has_sub = i + 1 < len(entries) and entries[i + 1][0] == 3
            out.append(
                f'          <li><a href="#{hid}">{inner}</a>'
                + ("\n          <ol>\n" if has_sub else "</li>\n")
            )
            open_sub = has_sub
        else:
            out.append(f'            <li><a href="#{hid}">{inner}</a></li>\n')
    if open_sub:
        out.append("          </ol></li>\n")
    out.append("        </ol>\n")
    return "".join(out)


def splice(shell: str, fragment: str) -> str:
    if shell.count(START) != 1 or shell.count(END) != 1:
        raise SystemExit(f"The page shell needs exactly one {START} and one {END}.")
    head, rest = shell.split(START, 1)
    _, tail = rest.split(END, 1)
    return head + fragment + tail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--site", type=Path, help="checkout of the website: updates its code/index.html and assets"
    )
    ap.add_argument("--out", type=Path, help="write fragment.html and the assets to this folder")
    ap.add_argument(
        "--update-readme", action="store_true", help="write the real example outputs into README.md"
    )
    args = ap.parse_args()
    if not (args.site or args.out or args.update_readme):
        ap.error("pass --site, --out or --update-readme")

    fragment, assets, runs = build(update_readme=args.update_readme)
    stale = [i + 1 for i, (_, old, new) in enumerate(runs) if old.strip() != new.strip()]
    print(f"examples run: {len(runs)}", end="")
    print(
        f"; README output differs for example(s) {stale} (--update-readme refreshes it)"
        if stale and not args.update_readme
        else ""
    )

    targets = []
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "fragment.html").write_text(fragment + "\n", encoding="utf-8")
        targets.append(args.out)
    if args.site:
        page = args.site / "code" / "index.html"
        if not page.exists():
            raise SystemExit(
                f"{page} not found: the site needs a code/index.html shell with the markers."
            )
        page.write_text(splice(page.read_text(encoding="utf-8"), fragment), encoding="utf-8")
        targets.append(page.parent)
        print(f"updated {page}")
    for folder in targets:
        for name, src in assets.items():
            shutil.copyfile(src, folder / name)


if __name__ == "__main__":
    main()
