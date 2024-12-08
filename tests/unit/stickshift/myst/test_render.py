import base64
from io import BytesIO
from pathlib import Path
import textwrap
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from PIL import Image
import pytest

from stickshift import myst
from stickshift.myst._myst import _merge_stream_outputs


def test_paragraphs():
    #
    # Givens
    #

    # Markdown content w/ paragraphs
    markdown = _dedent(
        """
        alpha beta

        gamma epsilon
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = _dedent(
        """
        <p>alpha beta</p>
        <p>gamma epsilon</p>
        """
    )

    assert html == _normalize_html(expected_html)


def test_headings():
    #
    # Givens
    #

    # Markdown content w/ headings
    markdown = _dedent(
        """
        # Heading1
        ## Heading2
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = _dedent(
        """
        <h1 id="heading1">Heading1</h1>
        <h2 id="heading2">Heading2</h2>
        """
    )

    assert html == _normalize_html(expected_html)


def test_lists():
    #
    # Givens
    #

    # Markdown content w/ lists
    markdown = _dedent(
        """
        * alpha
        * beta
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = _dedent(
        """
        <ul>
            <li>alpha</li>
            <li>beta</li>
        </ul>
        """
    )

    assert html == _normalize_html(expected_html)


def test_fence():
    #
    # Givens
    #

    # Markdown content w/ fence
    markdown = _dedent(
        """
        ```markdown
        This is **markdown** code.
        ```

        ```python
        print("Hello World!")
        ```
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = _dedent(
        """
        <div class="highlight language-markdown">
            <pre><span></span>This is <span class="gs">**markdown**</span> code.\n</pre>
        </div>
        <div class="highlight language-python">
            <pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">"Hello World!"</span><span class="p">)</span>\n</pre>
        </div>
        """
    )

    assert html == _normalize_html(expected_html)


_admonition_types = [
    "note",
    "important",
    "hint",
    "seealso",
    "tip",
    "attention",
    "caution",
    "warning",
    "danger",
    "error",
]


@pytest.mark.parametrize("admonition_type", _admonition_types)
def test_admonition(admonition_type: str):
    #
    # Givens
    #

    # Markdown content w/ admonitions
    markdown = _dedent(
        f"""
        :::{{{admonition_type}}}
        This is a *note*.
        :::
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = _dedent(
        f"""
        <aside class="admonition {admonition_type}">
            <p class="admonition-title">{admonition_type.capitalize()}</p>
            <p>This is a <em>note</em>.</p>
        </aside>
        """
    )

    assert html == _normalize_html(expected_html)


def test_card():
    #
    # Givens
    #

    # Markdown content w/ card
    markdown = _dedent(
        """
        :::{card}
        This is a *card*.
        :::
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = _dedent(
        """
        <div class="card">
            <p>This is a <em>card</em>.</p>
        </div>
        """
    )

    assert html == _normalize_html(expected_html)


def test_image():
    #
    # Givens
    #

    # Markdown content w/ images
    markdown = _dedent(
        """
        ```{image} x/y.png
        ```
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = '<img src="x/y.png">'

    assert html == _normalize_html(expected_html)


def test_image_options():
    #
    # Givens
    #

    # Markdown content w/ image
    markdown = _dedent(
        """
        ```{image} x/y.png
        :width: 300px
        :height: 200px
        :align: center
        ```
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = '<img src="x/y.png" class="align-center" style="width:300px;height:200px;">'

    assert html == _normalize_html(expected_html)


def test_figure():
    #
    # Givens
    #

    # Markdown content w/ figure
    markdown = _dedent(
        """
        ```{figure} x/y.png
        :label: fig1

        Alpha
        ```
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = _dedent(
        """
        <figure id="fig1">
            <img src="x/y.png" alt="Alpha">
            <figcaption>Figure 1: Alpha</figcaption>
        </figure>
        """
    )

    assert html == _normalize_html(expected_html)


def test_math():
    #
    # Givens
    #

    # Markdown content w/ math
    markdown = _dedent(
        """
        ```{math}
        x \\times x = x^2
        ```
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = _dedent(
        """
        $$
        x \\times x = x^2
        $$
        """
    )

    assert html == _normalize_html(expected_html)


def test_math_label():
    #
    # Givens
    #

    # Markdown content w/ math
    markdown = _dedent(
        """
        ```{math}
        :label: eq1
        x \\times x = x^2
        ```
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = _dedent(
        """
        $$
        \\begin{equation}
        x \\times x = x^2
        \\label{eq1}
        \\end{equation}
        $$
        """
    )

    assert html == _normalize_html(expected_html)


def test_code_cell_stdout(kernel_name: str):
    #
    # Givens
    #

    # Markdown content w/ executable code
    markdown = _dedent(
        f"""
        ---
        kernelspec:
          name: {kernel_name}
        ---
        ```{{code-cell}} python
        print("Hello, World!")
        ```
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = _dedent(
        """
        <div class="cell cell-input">
            <div class="highlight language-python">
                <pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">"Hello, World!"</span><span class="p">)</span>\n</pre>
            </div>
        </div>
        <div class="cell cell-output">
            <div class="highlight language-plain">
                <pre><span></span>Hello, World!\n</pre>
            </div>
        </div>
        """
    )

    assert html == _normalize_html(expected_html)


def test_merge_output_streams():
    #
    # Givens
    #

    # Multiple stdout and stderr outputs
    outputs = [
        {"output_type": "stream", "name": "stdout", "text": "alpha "},
        {"output_type": "stream", "name": "stderr", "text": "one "},
        {"output_type": "stream", "name": "stderr", "text": "two "},
        {"output_type": "stream", "name": "stdout", "text": "beta "},
        {"output_type": "stream", "name": "stdout", "text": "gamma "},
        {"output_type": "stream", "name": "stderr", "text": "three "},
    ]

    #
    # Whens
    #

    # I merge stream outputs
    outputs = _merge_stream_outputs(outputs)

    #
    # Thens
    #

    # There should be 4 outputs: stdout, stderr, stdout, stderr
    assert len(outputs) == 4
    assert outputs[0]["name"] == "stdout"
    assert outputs[0]["text"] == "alpha "
    assert outputs[1]["name"] == "stderr"
    assert outputs[1]["text"] == "one two "
    assert outputs[2]["name"] == "stdout"
    assert outputs[2]["text"] == "beta gamma "
    assert outputs[3]["name"] == "stderr"
    assert outputs[3]["text"] == "three "


def test_code_cell_stdout_stream(kernel_name: str):
    #
    # Givens
    #

    # Markdown content w/ code that generates output in multiple chunks
    markdown = _dedent(
        f"""
        ---
        kernelspec:
          name: {kernel_name}
        ---
        ```{{code-cell}} python
        from sys import stdout

        # Generate output in separate chunks
        for _ in range(3):
            stdout.write("Hello ")
            stdout.flush()
        ```
        """
    )

    #
    # Whens
    #

    # I render and parse content
    soup = _parse_html(myst.render(markdown))

    # I extract the cell outputs
    cell_outputs = soup.find_all(class_="cell-output")

    #
    # Thens
    #

    # All chunks should be collapsed into 1 output cell
    assert len(cell_outputs) == 1

    expected_html = _dedent(
        """
        <div class="cell cell-output">
            <div class="highlight language-plain">
                <pre><span></span>Hello Hello Hello\n</pre>
            </div>
        </div>
        """
    )

    html = cell_outputs[0].prettify().strip()

    assert html == _normalize_html(expected_html)


def test_code_cell_execute_result(kernel_name: str):
    #
    # Givens
    #

    # Markdown content w/ executable code
    markdown = _dedent(
        f"""
        ---
        kernelspec:
          name: {kernel_name}
        ---
        ```{{code-cell}} python
        1 + 2
        ```
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = _dedent(
        """
        <div class="cell cell-input">
            <div class="highlight language-python">
                <pre><span></span><span class="mi">1</span> <span class="o">+</span> <span class="mi">2</span>\n</pre>
            </div>
        </div>
        <div class="cell cell-output">
            <div class="highlight language-plain">
                <pre><span></span>3\n</pre>
            </div>
        </div>
        """
    )

    assert html == _normalize_html(expected_html)


def test_code_cell_display_data(kernel_name: str):
    #
    # Givens
    #

    # Markdown content w/ executable code
    markdown = _dedent(
        f"""
        ---
        kernelspec:
          name: {kernel_name}
        ---
        ```{{code-cell}} python
        from matplotlib import pyplot as plt
        import numpy as np

        n = 100
        x = np.random.normal(size=n)
        y = np.random.normal(size=n)
        plt.scatter(x, y)
        ```
        """
    )

    #
    # Whens
    #

    # I render content
    html = myst.render(markdown)

    # I extract the cell outputs
    cell_outputs = _parse_html(html).find_all(class_="cell-output")

    #
    # Thens
    #

    # Last cell output should be base64 encoded png image
    img_tag = cell_outputs[-1].img
    src_prefix = "data:image/png;base64,"
    assert img_tag["src"].startswith(src_prefix)

    # png data should be valid
    data = base64.b64decode(img_tag["src"][len(src_prefix) :])
    with Image.open(BytesIO(data)) as img:
        img.verify()


def test_code_cell_remove_input(kernel_name: str):
    #
    # Givens
    #

    # Markdown content w/ remove-input tag
    markdown = _dedent(
        f"""
        ---
        kernelspec:
          name: {kernel_name}
        ---
        ```{{code-cell}} python
        :tags: remove-input
        print("Hello, World!")
        ```
        """
    )

    #
    # Whens
    #

    # I render and parse content
    soup = _parse_html(myst.render(markdown))

    # I extract the cell inputs
    cell_inputs = soup.find_all(class_="cell-input")

    # I extract the cell outputs
    cell_outputs = soup.find_all(class_="cell-output")

    #
    # Thens
    #

    # Cell inputs should be empty
    assert len(cell_inputs) == 0

    # Cell outputs should not be empty
    assert len(cell_outputs) > 0


def test_code_cell_remove_output(kernel_name: str):
    #
    # Givens
    #

    # Markdown content w/ remove-output tag
    markdown = _dedent(
        f"""
        ---
        kernelspec:
          name: {kernel_name}
        ---
        ```{{code-cell}} python
        :tags: remove-output
        print("Hello, World!")
        ```
        """
    )

    #
    # Whens
    #

    # I render and parse content
    soup = _parse_html(myst.render(markdown))

    # I extract the cell inputs
    cell_inputs = soup.find_all(class_="cell-input")

    # I extract the cell outputs
    cell_outputs = soup.find_all(class_="cell-output")

    #
    # Thens
    #

    # Cell inputs should not be empty
    assert len(cell_inputs) > 0

    # Cell outputs should be empty
    assert len(cell_outputs) == 0


def test_code_cell_remove_cell(kernel_name: str):
    #
    # Givens
    #

    # Markdown content w/ remove-cell tag
    markdown = _dedent(
        f"""
        ---
        kernelspec:
          name: {kernel_name}
        ---
        ```{{code-cell}} python
        :tags: remove-cell
        print("Hello, World!")
        ```
        """
    )

    #
    # Whens
    #

    # I render and parse content
    soup = _parse_html(myst.render(markdown))

    # I extract the cell inputs
    cell_inputs = soup.find_all(class_="cell-input")

    # I extract the cell outputs
    cell_outputs = soup.find_all(class_="cell-output")

    #
    # Thens
    #

    # Cell inputs should be empty
    assert len(cell_inputs) == 0

    # Cell outputs should be empty
    assert len(cell_outputs) == 0


def test_subscript_superscript():
    #
    # Givens
    #

    # Markdown content w/ subscript and superscript
    markdown = "H{sub}`2`O and 4{sup}`th` of July"

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = "<p>H<sub>2</sub>O and 4<sup>th</sup> of July</p>"

    assert html == _normalize_html(expected_html)


def test_abbreviation():
    #
    # Givens
    #

    # Markdown content w/ abbreviation
    markdown = "Well {abbr}`MyST (Markedly Structured Text)` is cool!"

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = '<p>Well <abbr title="Markedly Structured Text">MyST</abbr> is cool!</p>'

    assert html == _normalize_html(expected_html)


def test_ref():
    #
    # Givens
    #

    # Markdown content w/ a figure and a reference
    markdown = _dedent(
        """
        ```{figure} x/y.png
        :label: figure-x
        Alpha
        ```

        {ref}`figure-x`
        """
    )

    #
    # Whens
    #

    # I render content
    html = myst.render(markdown)

    # I extract the reference link
    link_html = _parse_html(html).a.prettify().strip()

    #
    # Thens
    #

    expected_html = '<a class="ref" href="#figure-x">Figure 1</a>'

    assert link_html == _normalize_html(expected_html)


def test_ref_missing():
    #
    # Givens
    #

    # Markdown content w/ a figure and a reference
    markdown = _dedent(
        """
        {ref}`figure-x`
        """
    )

    #
    # Whens
    #

    # I render content
    html = myst.render(markdown)

    # I extract the reference link
    link_html = _parse_html(html).a.prettify().strip()

    #
    # Thens
    #

    expected_html = '<a class="ref missing" href="#figure-x">figure-x</a>'

    assert link_html == _normalize_html(expected_html)


def test_citation(tmp_path: Path):
    #
    # Givens
    #

    # Sample bibtex references
    references = _dedent(
        r"""
        @article{vaswani2017,
            author       = {Ashish Vaswani and
                            Noam Shazeer and
                            Niki Parmar and
                            Jakob Uszkoreit and
                            Llion Jones and
                            Aidan N. Gomez and
                            Lukasz Kaiser and
                            Illia Polosukhin},
            title        = {Attention Is All You Need},
            journal      = {CoRR},
            volume       = {abs/1706.03762},
            year         = {2017},
            url          = {http://arxiv.org/abs/1706.03762},
            eprinttype    = {arXiv},
            eprint       = {1706.03762},
            timestamp    = {Sat, 23 Jan 2021 01:20:40 +0100},
            biburl       = {https://dblp.org/rec/journals/corr/VaswaniSPUJGKP17.bib},
            bibsource    = {dblp computer science bibliography, https://dblp.org}
        }

        @misc{devlin2019,
            title = {{BERT}: {Pre}-training of {Deep} {Bidirectional} {Transformers} for {Language} {Understanding}},
            shorttitle = {{BERT}},
            url = {https://arxiv.org/abs/1810.04805v2},
            abstract = {We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5\% (7.7\% point absolute improvement), MultiNLI accuracy to 86.7\% (4.6\% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).},
            language = {en},
            urldate = {2024-09-07},
            journal = {arXiv.org},
            author = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
            month = may,
            year = {2019},
            file = {Full Text PDF:/Users/andrewyoung/Zotero/storage/XV8HF8WJ/Devlin et al. - 2018 - BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf:application/pdf},
        }
        """
    )
    references_path = tmp_path / "references.bib"
    references_path.write_text(references)

    # Markdown content w/ citations
    markdown = _dedent(
        f"""
        ---
        bibliography:
          - {references_path}
        ---

        {{cite:p}}`vaswani2017` / {{cite:t}}`vaswani2017`

        {{cite:p}}`devlin2019` / {{cite:t}}`devlin2019`
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = _dedent(
        """
        <p>(<a class="ref" href="#ref-vaswani2017">Vaswani <em>et al.</em>, 2017</a>) / <a class="ref" href="#ref-vaswani2017">Vaswani <em>et al.</em> (2017)</a></p>
        <p>(<a class="ref" href="#ref-devlin2019">Devlin <em>et al.</em>, 2019</a>) / <a class="ref" href="#ref-devlin2019">Devlin <em>et al.</em> (2019)</a></p>
        """
    )

    assert html == _normalize_html(expected_html)


def test_bibliography(tmp_path: Path, workspace_path: Path):
    #
    # Givens
    #

    # Sample bibtex references
    references = _dedent(
        """
        @article{vaswani2017,
            author       = {Ashish Vaswani and
                            Noam Shazeer and
                            Niki Parmar and
                            Jakob Uszkoreit and
                            Llion Jones and
                            Aidan N. Gomez and
                            Lukasz Kaiser and
                            Illia Polosukhin},
            title        = {Attention Is All You Need},
            journal      = {CoRR},
            volume       = {abs/1706.03762},
            year         = {2017},
            url          = {http://arxiv.org/abs/1706.03762},
            eprinttype    = {arXiv},
            eprint       = {1706.03762},
            timestamp    = {Sat, 23 Jan 2021 01:20:40 +0100},
            biburl       = {https://dblp.org/rec/journals/corr/VaswaniSPUJGKP17.bib},
            bibsource    = {dblp computer science bibliography, https://dblp.org}
        }
        """
    )
    references_path = tmp_path / "references.bib"
    references_path.write_text(references)

    # CSL
    csl_path = workspace_path / "tests" / "unit" / "stickshift" / "myst" / "style.csl"

    # Markdown content w/ bibliography
    markdown = _dedent(
        f"""
        ---
        bibliography:
          - {references_path}
        ---
        ```{{bibliography}}
        :csl: {csl_path}
        ```
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = _dedent(
        """
        <div id="refs" class="references csl-bib-body hanging-indent" data-entry-spacing="0" role="list">
            <div id="ref-vaswani2017" class="csl-entry" role="listitem">
                Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. <span>“Attention Is All You Need.”</span> <em>CoRR</em> abs/1706.03762. <a href="http://arxiv.org/abs/1706.03762">http://arxiv.org/abs/1706.03762</a>.
            </div>
        </div>
        """
    )

    assert html == _normalize_html(expected_html)


def test_article():
    #
    # Givens
    #

    # Markdown content w/ article content
    markdown = _dedent(
        """
        ---
        title: Alpha
        subtitle: Beta
        published: 2021-07-01
        banner: x/y.png
        ---

        # Section 1

        Lorem ipsum dolor sit amet, consectetur adipiscing elit.

        ## Section 1.1

        Integer nec odio. Praesent libero.
        """
    )

    #
    # Whens
    #

    # I render and normalize content
    html = _normalize_html(myst.render(markdown))

    #
    # Thens
    #

    expected_html = _dedent(
        """
        <header>
            <h1>Alpha</h1>
            <h2>Beta</h2>
            <h3>July 1, 2021</h3>
            <img src="x/y.png" alt="Alpha">
        </header>

        <h1 id="section-1">Section 1</h1>
        <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>

        <h2 id="section-11">Section 1.1</h2>
        <p>Integer nec odio. Praesent libero.</p>
        """
    )

    assert html == _normalize_html(expected_html)


def _parse_html(html: str) -> BeautifulSoup:
    """Parse html fragment."""
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    return BeautifulSoup(html, features="html.parser")


def _normalize_html(html: str) -> str:
    """Normalize html fragment."""
    return _parse_html(html).prettify().strip()


def _dedent(value: str) -> str:
    return textwrap.dedent(value).strip()
