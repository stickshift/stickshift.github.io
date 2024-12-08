import textwrap

from stickshift import myst


def test_front_matter():
    #
    # Givens
    #

    # Markdown content w/ front matter
    markdown = _dedent(
        """
        ---
        field1: value1
        field2:
            - value2
        ---

        alpha beta

        gamma epsilon
        """
    )

    #
    # Whens
    #

    # I parse front matter
    front_matter = myst.front_matter(markdown)

    #
    # Thens
    #

    # front matter should be populated
    assert front_matter == {"field1": "value1", "field2": ["value2"]}


def test_front_matter_empty():
    #
    # Givens
    #

    # Markdown content w/ out front matter
    markdown = _dedent(
        """
        alpha beta

        gamma epsilon
        """
    )

    #
    # Whens
    #

    # I parse front matter
    front_matter = myst.front_matter(markdown)

    #
    # Thens
    #

    # front matter should be empty
    assert front_matter == {}


def _dedent(value: str) -> str:
    return textwrap.dedent(value).strip()
