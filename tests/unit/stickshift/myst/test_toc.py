import textwrap

from stickshift import myst


def test_toc():
    #
    # Givens
    #

    # Markdown content w/ article content
    markdown = _dedent(
        """
        # Section 1

        Lorem ipsum dolor sit amet, consectetur adipiscing elit.

        ## Section 1.1

        Integer nec odio. Praesent libero.

        ### Section 1.1.1

        Integer nec odio. Praesent libero.

        # Section 2

        Lorem ipsum dolor sit amet, consectetur adipiscing elit.

        ## Section 2.1

        Integer nec odio. Praesent libero.

        ### Section 2.1.1

        Integer nec odio. Praesent libero.
        """
    )

    #
    # Whens
    #

    # I parse toc w/ max 2 levels
    toc = myst.toc(markdown, max_heading_level=2)

    #
    # Thens
    #

    # toc should be populated
    assert toc.name == "root"
    assert toc.link == "#"

    assert toc.sections[0].name == "Section 1"
    assert toc.sections[0].link == "#section-1"
    assert toc.sections[0].sections[0].name == "Section 1.1"
    assert toc.sections[0].sections[0].link == "#section-11"

    # Section 1.1.1 should not be included in toc because it exceeds max depth of 2
    assert len(toc.sections[0].sections[0].sections) == 0

    assert toc.sections[1].name == "Section 2"
    assert toc.sections[1].link == "#section-2"
    assert toc.sections[1].sections[0].name == "Section 2.1"
    assert toc.sections[1].sections[0].link == "#section-21"

    # Section 2.1.1 should not be included in toc because it exceeds max depth of 2
    assert len(toc.sections[1].sections[0].sections) == 0


def test_toc_deep():
    #
    # Givens
    #

    # Markdown content w/ 4 levels of headings
    markdown = _dedent(
        """
        # Section 1

        Lorem ipsum dolor sit amet.

        ## Section 1.1

        Lorem ipsum dolor sit amet.

        ### Section 1.1.1

        Lorem ipsum dolor sit amet.

        #### Section 1.1.1.1

        Lorem ipsum dolor sit amet.
        """
    )

    #
    # Whens
    #

    # I parse toc w/ max 4 levels
    toc = myst.toc(markdown, max_heading_level=4)

    #
    # Thens
    #

    # toc have 4 levels
    assert toc.sections[0].sections[0].sections[0].sections[0].name == "Section 1.1.1.1"


def _dedent(value: str) -> str:
    return textwrap.dedent(value).strip()
