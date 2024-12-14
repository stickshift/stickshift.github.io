from contextlib import contextmanager
from datetime import date
from functools import partial
import logging
import re
import subprocess
from textwrap import dedent
from typing import Any, Iterator, Literal, Mapping, MutableMapping, Self, Sequence

import bibtexparser
from bibtexparser import Library
from bibtexparser.middlewares import SeparateCoAuthors, SplitNameParts
from bs4 import BeautifulSoup
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
from markdown_it import MarkdownIt
from markdown_it.common.utils import escapeHtml
from markdown_it.renderer import RendererHTML
from markdown_it.rules_core import StateCore
from markdown_it.token import Token
from markdown_it.tree import SyntaxTreeNode
from markdown_it.utils import EnvType, OptionsDict
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.colon_fence import colon_fence_plugin
from mdit_py_plugins.front_matter import front_matter_plugin
from mdit_py_plugins.myst_blocks import myst_block_plugin
from mdit_py_plugins.myst_role.index import myst_role
from nbclient import NotebookClient
from nbformat import NotebookNode
from nbformat import v4 as nbf
from pydantic import BaseModel, model_validator
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, get_lexer_for_mimetype
import yaml

__all__ = [
    "TOC",
    "create_parser",
    "front_matter",
    "render",
    "toc",
]

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------------


class TOC(BaseModel):
    """Table of contents."""

    level: int

    name: str

    link: str

    sections: list[Self]


class ArticleFrontMatter(BaseModel):
    title: str | None = None
    subtitle: str | None = None
    published: date | None = None
    banner: str | None = None


class DirectiveMeta(BaseModel):
    name: str
    argument: str | None
    options: Mapping[str, str]
    body: str | None


class ImageOptions(BaseModel):
    width: str | None = None
    height: str | None = None
    align: str | None = None


class ImageMeta(DirectiveMeta):
    name: Literal["image"]
    argument: str
    options: ImageOptions
    body: None


class FigureOptions(ImageOptions):
    label: str


class FigureMeta(DirectiveMeta):
    name: Literal["figure"]
    argument: str
    options: FigureOptions
    body: str


class MathMeta(DirectiveMeta):
    name: Literal["math"]
    argument: None
    body: str


class BibliographyOptions(BaseModel):
    csl: str


class BibliographyMeta(DirectiveMeta):
    name: Literal["bibliography"]
    argument: None
    options: BibliographyOptions
    body: None


class CodeCellOptions(BaseModel):
    tags: Sequence[str]

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        tags = data.get("tags", ())
        if isinstance(tags, str):
            tags = tuple(v.strip() for v in tags.split(","))
        data["tags"] = tags

        return data


class CodeCellMeta(DirectiveMeta):
    name: Literal["code-cell"]

    argument: Literal["python"]

    options: CodeCellOptions

    body: str

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Map python synonyms to 'python'
        if data["argument"] in {"python", "ipython", "python3", "ipython3"}:
            data["argument"] = "python"

        return data


# -------------------------------------------------------------------------------
# Public Functions
# -------------------------------------------------------------------------------


def create_parser(max_heading_level: int | None = None) -> MarkdownIt:
    """Parser factory."""
    # Defaults
    if max_heading_level is None:
        max_heading_level = 2

    parser = (
        MarkdownIt("commonmark")
        .enable("table")
        .use(anchors_plugin, max_level=max_heading_level)
        .use(front_matter_plugin)
        .use(myst_block_plugin)
        .use(colon_fence_plugin)
        .use(_myst_plugin)
    )

    return parser


def parse(content: str) -> list[Token]:
    """Parse MyST content into MarkdownIt token stream."""
    return create_parser().parse(content)


def render(content: str) -> str:
    """Render MyST content as HTML."""
    fm = front_matter(content)

    with _render_environment(fm) as env:
        return create_parser().render(content, env=env)


def front_matter(content: str) -> Mapping[str, Any]:
    """Parse front matter from MyST content."""
    tokens = create_parser().parse(content)
    if tokens and tokens[0].type == "front_matter":
        return yaml.safe_load(tokens[0].content)

    return {}


def toc(content: str, max_heading_level: int | None = None) -> TOC:
    """Parse MyST content into TOC."""
    md = create_parser(max_heading_level=max_heading_level)

    root_toc = TOC(level=0, name="root", link="#", sections=[])
    tocs = [root_toc]

    # Parse content
    root_node = SyntaxTreeNode(md.parse(content))

    # Parse headings
    for child_node in root_node.children:
        if child_node.type != "heading":
            continue

        # If child has no id, then it has exceeded max depth
        if "id" not in child_node.attrs:
            continue

        heading_level = int(child_node.tag[1:])

        # Pop last toc if levels are the same
        while heading_level <= tocs[-1].level:
            tocs = tocs[:-1]

        parent_toc = tocs[-1]

        # Sanity check
        if parent_toc.level != heading_level - 1:
            raise ValueError("Invalid heading hierarchy.")

        # Render name
        tokens = [gc.token for gc in child_node.children]
        name = md.renderer.render(tokens, OptionsDict({}), {})

        child_toc = TOC(level=heading_level, name=name, link=f"#{child_node.attrs['id']}", sections=[])
        parent_toc.sections.append(child_toc)
        tocs.append(child_toc)

    return root_toc


# -------------------------------------------------------------------------------
# Directive Renderers
# -------------------------------------------------------------------------------


def _admonition_directive(renderer: RendererHTML, token: Token, **kwargs) -> str:
    html = f'<aside class="admonition {token.meta["name"]}">'
    html += f'<p class="admonition-title">{token.meta["name"].capitalize()}</p>'
    html += render(token.meta["body"])
    html += "</aside>"

    return html


def _card_directive(renderer: RendererHTML, token: Token, **kwargs) -> str:
    # Validate
    meta = DirectiveMeta.model_validate(token.meta)

    html = "<div "

    classes = ["card"]
    if "align" in meta.options:
        classes.append(f"align-{meta.options['align']}")
    html += f' class="{" ".join(classes)}"'

    html += ">"
    html += render(meta.body)
    html += "</div>"

    return html


def _image_directive(renderer: RendererHTML, token: Token, meta: DirectiveMeta | None = None, **kwargs) -> str:
    # Validate
    if meta is None:
        meta = ImageMeta.model_validate(token.meta)

    html = f'<img src="{meta.argument}"'

    classes = []
    if meta.options.align:
        classes.append(f"align-{meta.options.align}")

    if classes:
        html += f' class="{" ".join(classes)}"'

    style = ""
    if meta.options.width:
        style += f"width:{meta.options.width};"
    if meta.options.height:
        style += f"height:{meta.options.height};"

    if style:
        html += f' style="{style}"'

    html += ">"

    return html


def _figure_directive(renderer: RendererHTML, token: Token, **kwargs) -> str:
    # Validate
    meta = FigureMeta.model_validate(token.meta)

    # Extend image AST
    img_html = _image_directive(renderer, token, meta=meta)
    img_tag = _parse_html(img_html).find("img")
    img_tag["alt"] = meta.body

    html = f'<figure id="{meta.options.label}">{img_tag}<figcaption>{_reference_name(token)}: {meta.body}</figcaption></figure>'

    return html


def _math_directive(renderer: RendererHTML, token: Token, **kwargs) -> str:
    # Validate
    meta = MathMeta.model_validate(token.meta)

    # If there is no label, just return body wrapped in $$s
    if "label" not in meta.options:
        return f"$$\n{meta.body}\n$$\n"

    html = "$$\n"
    html += "\\begin{equation}\n"
    html += meta.body + "\n"
    html += f"\\label{{{meta.options['label']}}}\n"
    html += "\\end{equation}\n"
    html += "$$\n"

    return html


def _code_cell_directive(renderer: RendererHTML, token: Token, env: EnvType, **kwargs) -> str:
    # Parse metadata
    meta = CodeCellMeta.model_validate(token.meta)
    language = meta.argument
    source = meta.body

    # Look up notebook client
    client = env["jupyter"]["clients"].get(language)
    if not client:
        raise ValueError(f"Kernel not found for language: {language}")

    # Add code cell to end of notebook
    cell = nbf.new_code_cell(source)
    cell_index = len(client.nb.cells)
    client.nb.cells.append(cell)

    # Execute notebook cell
    client.execute_cell(cell, cell_index)
    logger.info(f"Executed cell {cell.id}")

    # Check if we need to filter the cell completely
    if "remove-cell" in meta.options.tags:
        return ""

    # Preprocess cells
    cell.outputs = _merge_stream_outputs(cell.outputs)

    html = ""

    # Render cell input
    if "remove-input" not in meta.options.tags:
        html += f'<div class="cell cell-input">{_highlight_code(source, language=language)}</div>'

    # Render cell outputs
    if "remove-output" not in meta.options.tags:
        for output in cell.outputs:
            html += f'<div class="cell cell-output">{_render_cell_output(output)}</div>'

    return html


def _png_data(data: str) -> str:
    return f'<img src="data:image/png;base64,{escapeHtml(data)}">'


def _data_frame_data(soup: BeautifulSoup) -> str:
    """Transform pandas data frame tables."""

    html = ""

    tables = soup.find_all("table", class_="dataframe")

    for table in tables:

        html += '<table class="dataframe">'

        # Header
        html += "<thead><tr>" + "".join([str(c) for c in table.thead.tr.contents]) + "</tr></thead>"

        # Body
        html += str(table.tbody)
        
        html += "</table>"


    return html


def _html_data(data: str) -> str:

    soup = _parse_html(data)

    # Transform pandas data frame tables
    if soup.find_all("table", class_="dataframe"):
        return _data_frame_data(soup)
    
    return data


def _render_cell_output(output: dict) -> str:
    # Stream
    if output["output_type"] == "stream":
        return _highlight_code(output["text"], language="plain")

    # Execute Result / Display Data
    if output["output_type"] in {"execute_result", "display_data"}:
        mime_bundle = output["data"]

        # PNGs
        if "image/png" in mime_bundle:
            return _png_data(mime_bundle["image/png"])

        # HTML text
        if "text/html" in mime_bundle:
            return _html_data(mime_bundle["text/html"])

        # Plain text
        if "text/plain" in mime_bundle:
            return _highlight_code(mime_bundle["text/plain"], language="plain")

        raise ValueError(f"Unsupported mime bundle: {mime_bundle}")

    raise ValueError(f"Unsupported output type: {output['output_type']}")


def _bibliography_directive(renderer: RendererHTML, token: Token, env: EnvType, **kwargs) -> str:
    # Validate
    meta = BibliographyMeta.model_validate(token.meta)

    if "bibliography_path" not in env:
        raise ValueError("Missing bibliography")
    bibliography_path = env["bibliography_path"]

    # Shell out to pandoc to convert bibliography to HTML
    command = ["pandoc", bibliography_path, "--citeproc", "--csl", meta.options.csl]
    result = subprocess.run(command, capture_output=True, text=True, check=True)

    html = result.stdout

    # Strip newlines
    html = html.replace("\n", " ")

    return html


# -------------------------------------------------------------------------------
# Roles
# -------------------------------------------------------------------------------


_abbr_pattern = re.compile(r"^(.*) \((.*)\)")


def _abbr_role(_: RendererHTML, token: Token, **kwargs) -> str:
    match = re.match(_abbr_pattern, token.content)
    if not match:
        raise ValueError(f"Invalid abbr role: {token.content}")

    name = match.group(1)
    title = match.group(2)

    return f'<abbr title="{title}">{name}</abbr>'


def _math_role(_: RendererHTML, token: Token, **kwargs) -> str:
    return f"${token.content}$"


def _subscript_role(_: RendererHTML, token: Token, **kwargs) -> str:
    return f"<sub>{escapeHtml(token.content)}</sub>"


def _superscript_role(_: RendererHTML, token: Token, **kwargs) -> str:
    return f"<sup>{escapeHtml(token.content)}</sup>"


def _cite_p_role(_: RendererHTML, token: Token, env: EnvType, **kwargs) -> str:
    citation_key = token.content
    author, year = _author_year(env, citation_key)

    return f'(<a class="ref" href="#ref-{citation_key}">{author} <em>et al.</em>, {year}</a>)'


def _cite_t_role(_: RendererHTML, token: Token, env: EnvType, **kwargs) -> str:
    citation_key = token.content
    author, year = _author_year(env, citation_key)

    return f'<a class="ref" href="#ref-{citation_key}">{author} <em>et al.</em> ({year})</a>'


def _reference_name(target: Token | None, default: str | None = None) -> str | None:
    if target is None:
        return default

    # Target should have a name and a number
    assert "name" in target.meta
    assert "number" in target.meta

    return f"{target.meta['name'].capitalize()} {target.meta['number']}"


def _ref_role(_: RendererHTML, token: Token, env: EnvType, **kwargs) -> str:
    classes = ["ref"]

    target = token.meta.get("target")
    if target is None:
        classes.append("missing")

    return f'<a class="{" ".join(classes)}" href="#{token.content}">{_reference_name(target, token.content)}</a>'


# -------------------------------------------------------------------------------
# Parser Rules
# -------------------------------------------------------------------------------

_directive_pattern = re.compile(r"^\{([^}]+)\}(.*)")
_option_pattern = re.compile(r"^:([^:]+):(.*)")


def _parse_directive_attrs(token: Token) -> tuple[str | None, str | None, dict[str, str] | None, str | None]:
    """Parse attributes from directive tokens.

    ```{name} argument
    :option1: value1
    :option2: value2

    Body
    ```

    or

    ```{name} argument
    ---
    option1: value1
    option2: value2
    ---

    Body
    ```


    """
    empty_results = (None, None, None, None)

    # Check for fence blocks
    if token.type not in {"fence", "colon_fence"}:
        return empty_results

    # Parse name and argument
    match = re.match(_directive_pattern, token.info)
    if not match:
        return empty_results

    name = match.group(1)
    argument = match.group(2).strip()
    if not argument:
        argument = None

    # Parse options
    options = {}

    # Yaml options
    if token.content.startswith("---\n"):
        # Extract yaml options
        start_index = len("---\n")
        end_index = token.content.find("---\n", start_index)
        data = token.content[start_index:end_index]

        options = yaml.safe_load(data)

        # Combine remaining lines into body
        body = token.content[end_index + len("---\n") :]
        if not body:
            body = None

    # Sphinx options
    else:
        lines = token.content.splitlines()
        current_line = 0
        while current_line < len(lines):
            match = re.match(_option_pattern, lines[current_line])
            if not match:
                break

            option_name = match.group(1)
            option_value = match.group(2).strip()
            options[option_name] = option_value

            current_line += 1

        # Combine remaining lines into body
        body = None
        if current_line < len(lines):
            body = "\n".join(lines[current_line:])

    # Remove leading newline that separates options from body
    if body and body[0] == "\n":
        body = body[1:]

    return name, argument, options, body


def _myst_directive(state: StateCore) -> None:
    numbering = {}

    # Check each token for possible directives
    for token in state.tokens:
        # Check for directive tokens
        name, argument, options, body = _parse_directive_attrs(token)
        if name is None:
            continue

        if name not in numbering:
            numbering[name] = 1

        # Update token
        token.type = "myst_directive"
        token.meta = {
            "name": name,
            "argument": argument,
            "options": options,
            "body": body,
            "number": numbering[name],
        }

        # Increment numbering
        numbering[name] = numbering[name] + 1


def _tag_references(token: Token, catalog: dict[str, Token]):
    # Tag references on ref roles
    if token.type == "myst_role" and token.meta.get("name") == "ref":
        token.meta["target"] = catalog.get(token.content)

    # Recurse into inline tokens
    if token.type == "inline":
        for child in token.children:
            _tag_references(child, catalog)


def _references(state: StateCore) -> None:
    # Catalog all labeled directives
    catalog = {}
    for token in state.tokens:
        if token.type != "myst_directive":
            continue

        label = token.meta.get("options", {}).get("label")
        if not label:
            continue

        catalog[label] = token

    # Link references
    for token in state.tokens:
        _tag_references(token, catalog)


# -------------------------------------------------------------------------------
# Renderer Rules
# -------------------------------------------------------------------------------


def _render_fence(
    renderer,
    tokens: list[Token],
    idx: int,
    options: OptionsDict,
    env: MutableMapping[str, Any],
) -> Any:
    """Render markdown fence."""
    return _highlight_code(tokens[idx].content, language=tokens[idx].info)


def _render_front_matter(
    renderer,
    tokens: list[Token],
    idx: int,
    options: OptionsDict,
    env: MutableMapping[str, Any],
) -> Any:
    """Render article front matter."""
    # Skip front matter if its not first token
    if idx != 0:
        return ""

    data = yaml.safe_load(tokens[idx].content)
    fm = ArticleFrontMatter.model_validate(data)

    # If all article front matter fields are missing, bail
    if fm.title is None and fm.subtitle is None and fm.published is None:
        return ""

    html = "<header>"

    html += dedent(
        f"""
        <h1>{fm.title}</h1>
        <h2>{fm.subtitle}</h2>
        <h3>{fm.published:%B %-d, %Y}</h3>
        """
    )

    if fm.banner:
        html += f'<img src="{fm.banner}" alt="{fm.title}">'

    html += "</header>"

    return html


_directive_renderers = {
    "note": _admonition_directive,
    "important": _admonition_directive,
    "hint": _admonition_directive,
    "seealso": _admonition_directive,
    "tip": _admonition_directive,
    "attention": _admonition_directive,
    "caution": _admonition_directive,
    "warning": _admonition_directive,
    "danger": _admonition_directive,
    "error": _admonition_directive,
    "card": _card_directive,
    "image": _image_directive,
    "figure": _figure_directive,
    "math": _math_directive,
    "code-cell": _code_cell_directive,
    "bibliography": _bibliography_directive,
}


def _render_myst_directive(
    renderer,
    tokens: list[Token],
    idx: int,
    options: OptionsDict,
    env: MutableMapping[str, Any],
) -> Any:
    name = tokens[idx].meta.get("name")
    if name not in _directive_renderers:
        raise ValueError(f"Unknown directive: {name}")

    return _directive_renderers[name](renderer, tokens[idx], env=env)


_role_renderers = {
    "abbr": _abbr_role,
    "math": _math_role,
    "sub": _subscript_role,
    "sup": _superscript_role,
    "cite:p": _cite_p_role,
    "cite:t": _cite_t_role,
    "ref": _ref_role,
}


def _render_myst_role(
    renderer,
    tokens: list[Token],
    idx: int,
    options: OptionsDict,
    env: MutableMapping[str, Any],
) -> Any:
    name = tokens[idx].meta.get("name")
    if name not in _role_renderers:
        raise ValueError(f"Unknown role: {name}")

    return _role_renderers[name](renderer, tokens[idx], env=env)


# -------------------------------------------------------------------------------
# Private Functions
# -------------------------------------------------------------------------------


def _kernel_spec(kernel_name: str) -> KernelSpec:
    """Look up kernel spec."""
    return KernelSpecManager().get_kernel_spec(kernel_name)


@contextmanager
def _render_environment(front_matter: Mapping[str, Any]) -> Iterator[EnvType]:
    env = {
        "bibliography": None,
        "jupyter": {
            "clients": {},
        },
    }

    # Bibliography
    if "bibliography" in front_matter:
        # front_matter.bibliography is a list to match mystmd's approach. But we only support a single entry for now.
        if len(front_matter["bibliography"]) != 1:
            raise ValueError("We only support a single bibtex file in bibliography for now.")

        bibliography_path = front_matter["bibliography"][0]

        middle_wares = [
            SeparateCoAuthors(),
            SplitNameParts(),
        ]

        env |= {
            "bibliography_path": bibliography_path,
            "bibliography": bibtexparser.parse_file(bibliography_path, append_middleware=middle_wares),
        }

    # Kernel
    kernel_name = front_matter.get("kernelspec", {}).get("name")
    if kernel_name is None:
        yield env
        return

    notebook = nbf.new_notebook()
    client = NotebookClient(notebook, kernel_name=kernel_name)

    spec = _kernel_spec(kernel_name)
    env["jupyter"]["clients"][spec.language] = client

    with client.setup_kernel():
        logger.info(f"Launched kernel: {kernel_name}")

        yield env


def _myst_plugin(md: MarkdownIt) -> None:
    """MyST MarkdownIt plugin."""
    # Register parser rules
    md.inline.ruler.before("backticks", "myst_role", myst_role)
    md.core.ruler.after("block", "myst_directive", _myst_directive)
    md.core.ruler.after("inline", "references", _references)

    # Register renderer rules
    md.add_render_rule("fence", _render_fence)
    md.add_render_rule("front_matter", _render_front_matter)
    md.add_render_rule("myst_role", _render_myst_role)
    md.add_render_rule("myst_directive", _render_myst_directive)


def _parse_html(html: str) -> BeautifulSoup:
    """Parse html fragment."""
    return BeautifulSoup(html, features="html.parser")


def _highlight_code(code: str, *, language: str) -> str:
    """Highlight code syntax using Pygments."""
    # Create lexer based on language
    if language == "plain":
        lexer_factory = partial(get_lexer_for_mimetype, "text/plain")
    else:
        lexer_factory = partial(get_lexer_by_name, language)
    lexer = lexer_factory(stripall=True)

    # Highlight using pygments
    html = highlight(code, lexer, HtmlFormatter())

    # Parse ast
    soup = _parse_html(html)

    # Inject language as class
    soup.div["class"].append(f"language-{language}")

    return str(soup)


def _author_year(env: EnvType, citation_key: str) -> tuple[str, str]:
    if "bibliography" not in env:
        raise ValueError("Missing bibliography")

    bibliography: Library = env["bibliography"]

    # Look up entry
    entry = bibliography.entries_dict.get(citation_key)
    if not entry:
        raise ValueError(f"Key {citation_key} not found in bibliography")

    # Look up authors
    authors = entry.get("author")
    if not authors:
        raise ValueError(f"Bibliography is missing author for key {citation_key}")

    # Use last name of first author. Do better later.
    author = authors.value[0].last[0]

    # Look up year
    year = entry.get("year")
    if not year:
        raise ValueError(f"Bibliography is missing year for key {citation_key}")
    year = year.value

    return author, year


def _merge_stream_outputs(outputs: Sequence[NotebookNode]):
    """Merge consecutive outputs for the same stream."""
    merged = []

    # Initialize block
    name = None
    buffer = ""

    for output in outputs:
        # Stream output
        if output["output_type"] == "stream":
            # New block
            if name != output["name"]:
                # Accumulate previous block
                if buffer:
                    merged.append({"output_type": "stream", "name": name, "text": buffer})

                # Start new block
                name = output["name"]
                buffer = output["text"]

            # Append to existing block
            else:
                buffer += output["text"]

        # Non-stream output
        else:
            # Accumulate previous block
            if buffer:
                merged.append({"output_type": "stream", "name": name, "text": buffer})

            # Reset block
            name = None
            buffer = ""

            # Append output
            merged.append(output)

    # Accumulate final block
    if buffer:
        merged.append({"output_type": "stream", "name": name, "text": buffer})

    return merged
