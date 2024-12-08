import textwrap

import yaml

from stickshift import myst


def test_paragraphs():
    #
    # Givens
    #

    # Markdown content w/ a paragraph
    markdown = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."

    # I created a parser
    parser = myst.create_parser()

    #
    # Whens
    #

    # I parse markdown
    tokens = parser.parse(markdown)

    #
    # Thens
    #

    # tokens should contain:
    #   * paragraph_open
    #   * inline / text
    #   * paragraph_close
    #

    assert len(tokens) == 3

    assert tokens[0].type == "paragraph_open"

    assert tokens[1].type == "inline"
    assert tokens[1].children[0].type == "text"
    assert tokens[1].children[0].content == markdown

    assert tokens[2].type == "paragraph_close"


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

    # I created a parser
    parser = myst.create_parser()

    #
    # Whens
    #

    # I parse markdown
    tokens = parser.parse(markdown)

    #
    # Thens
    #

    # tokens should contain:
    #   * heading_open / h1
    #   * inline / text
    #   * heading_close
    #   * heading_open / h2
    #   * inline / text
    #   * heading_close

    assert len(tokens) == 6
    assert tokens[0].type == "heading_open"
    assert tokens[0].tag == "h1"
    assert tokens[1].type == "inline"
    assert tokens[1].children[0].type == "text"
    assert tokens[1].children[0].content == "Heading1"
    assert tokens[2].type == "heading_close"

    assert tokens[3].type == "heading_open"
    assert tokens[3].tag == "h2"
    assert tokens[4].type == "inline"
    assert tokens[4].children[0].type == "text"
    assert tokens[4].children[0].content == "Heading2"
    assert tokens[5].type == "heading_close"


def test_roles():
    #
    # Givens
    #

    # Markdown content w/ a role
    markdown = "Lorem ipsum dolor sit amet, {math}`x + x = 2x` consectetur adipiscing elit."

    # I created a parser
    parser = myst.create_parser()

    #
    # Whens
    #

    # I parse markdown
    tokens = parser.parse(markdown)

    #
    # Thens
    #

    # tokens should contain:
    #   * paragraph_open
    #   * inline
    #     * text
    #     * myst_role
    #     * text
    #   * paragraph_close
    #

    assert len(tokens) == 3

    assert tokens[0].type == "paragraph_open"

    assert tokens[1].type == "inline"
    assert tokens[1].children[0].type == "text"
    assert tokens[1].children[0].content == "Lorem ipsum dolor sit amet, "
    assert tokens[1].children[1].type == "myst_role"
    assert tokens[1].children[1].content == "x + x = 2x"
    assert tokens[1].children[1].meta == {
        "name": "math",
    }
    assert tokens[1].children[2].type == "text"
    assert tokens[1].children[2].content == " consectetur adipiscing elit."

    assert tokens[2].type == "paragraph_close"


def test_directives_minimal():
    #
    # Givens
    #

    # Markdown content w/ a directive
    markdown = _dedent(
        """
        ```{alpha}
        ```
        """
    )

    # I created a parser
    parser = myst.create_parser()

    #
    # Whens
    #

    # I parse markdown
    tokens = parser.parse(markdown)

    #
    # Thens
    #

    # tokens should contain a single token
    assert len(tokens) == 1
    token = tokens[0]

    # token should be populated
    assert token.type == "myst_directive"
    assert token.meta == {
        "name": "alpha",
        "argument": None,
        "options": {},
        "body": None,
        "number": 1,
    }


def test_directives_argument():
    #
    # Givens
    #

    # Markdown content w/ a directive
    markdown = _dedent(
        """
        ```{alpha} beta
        ```
        """
    )

    # I created a parser
    parser = myst.create_parser()

    #
    # Whens
    #

    # I parse markdown
    tokens = parser.parse(markdown)

    #
    # Thens
    #

    # tokens should contain a single token
    assert len(tokens) == 1
    token = tokens[0]

    # token should be populated
    assert token.type == "myst_directive"
    assert token.meta == {
        "name": "alpha",
        "argument": "beta",
        "options": {},
        "body": None,
        "number": 1,
    }


def test_directives_argument_options():
    #
    # Givens
    #

    # Markdown content w/ a directive
    markdown = _dedent(
        """
        ```{alpha} beta
        :option1: gamma
        ```
        """
    )

    # I created a parser
    parser = myst.create_parser()

    #
    # Whens
    #

    # I parse markdown
    tokens = parser.parse(markdown)

    #
    # Thens
    #

    # tokens should contain a single token
    assert len(tokens) == 1
    token = tokens[0]

    # token should be populated
    assert token.type == "myst_directive"
    assert token.meta == {
        "name": "alpha",
        "argument": "beta",
        "options": {
            "option1": "gamma",
        },
        "body": None,
        "number": 1,
    }


def test_directives_argument_options_yaml():
    #
    # Givens
    #

    # Markdown content w/ a directive
    markdown = _dedent(
        """
        ```{alpha} beta
        ---
        option1: gamma
        option2: [epsilon, delta]
        ---
        ```
        """
    )

    # I created a parser
    parser = myst.create_parser()

    #
    # Whens
    #

    # I parse markdown
    tokens = parser.parse(markdown)

    #
    # Thens
    #

    # tokens should contain a single token
    assert len(tokens) == 1
    token = tokens[0]

    # token should be populated
    assert token.type == "myst_directive"
    assert token.meta == {
        "name": "alpha",
        "argument": "beta",
        "options": {"option1": "gamma", "option2": ["epsilon", "delta"]},
        "body": None,
        "number": 1,
    }


def test_directives_argument_options_body():
    #
    # Givens
    #

    # Markdown content w/ a directive
    markdown = _dedent(
        """
        ```{alpha} beta
        :option1: gamma
        :option2: epsilon

        Lorem ipsum dolor.
        ```
        """
    )

    # I created a parser
    parser = myst.create_parser()

    #
    # Whens
    #

    # I parse markdown
    tokens = parser.parse(markdown)

    #
    # Thens
    #

    # tokens should contain a single token
    assert len(tokens) == 1
    token = tokens[0]

    # token should be populated
    assert token.type == "myst_directive"
    assert token.meta == {
        "name": "alpha",
        "argument": "beta",
        "options": {
            "option1": "gamma",
            "option2": "epsilon",
        },
        "body": "Lorem ipsum dolor.",
        "number": 1,
    }


def test_directives_colons_argument_options_body():
    #
    # Givens
    #

    # Markdown content w/ a directive
    markdown = _dedent(
        """
        :::{alpha} beta
        :option1: gamma
        :option2: epsilon

        Lorem ipsum dolor.
        :::
        """
    )

    # I created a parser
    parser = myst.create_parser()

    #
    # Whens
    #

    # I parse markdown
    tokens = parser.parse(markdown)

    #
    # Thens
    #

    # tokens should contain a single token
    assert len(tokens) == 1
    token = tokens[0]

    # token should be populated
    assert token.type == "myst_directive"
    assert token.meta == {
        "name": "alpha",
        "argument": "beta",
        "options": {
            "option1": "gamma",
            "option2": "epsilon",
        },
        "body": "Lorem ipsum dolor.",
        "number": 1,
    }


def test_directives_numbering():
    #
    # Givens
    #

    # Markdown content w/ multiple directives
    markdown = _dedent(
        """
        ```{alpha}
        ```

        ```{beta}
        ```

        ```{alpha}
        ```
        """
    )

    # I created a parser
    parser = myst.create_parser()

    #
    # Whens
    #

    # I parse markdown
    tokens = parser.parse(markdown)

    #
    # Thens
    #

    # First token should be Alpha 1
    assert tokens[0].meta["number"] == 1

    # Second token should be Beta 1
    assert tokens[1].meta["number"] == 1

    # Third token should be Alpha 2
    assert tokens[2].meta["number"] == 2


def test_front_matter():
    #
    # Givens
    #

    # Markdown content w/ front matter
    markdown = _dedent(
        """
        ---
        field1: alpha
        field2: ["beta", "gamma"]
        ---
        """
    )

    # I created a parser
    parser = myst.create_parser()

    #
    # Whens
    #

    # I parse markdown
    tokens = parser.parse(markdown)

    #
    # Thens
    #

    # tokens should contain a single token
    assert len(tokens) == 1
    token = tokens[0]

    # token should be populated
    assert token.type == "front_matter"
    markdown = yaml.safe_load(token.content)
    assert markdown == {
        "field1": "alpha",
        "field2": ["beta", "gamma"],
    }


def _dedent(value: str) -> str:
    return textwrap.dedent(value).strip()
