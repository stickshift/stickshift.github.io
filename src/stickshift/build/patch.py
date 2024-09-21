"""Patch Jekyll installation."""

from pathlib import Path
import re

import click

from stickshift import md5, shell


@click.command()
def patch_jekyll():
    """Patch Jekyll installation."""

    # kramdown path
    kramdown_path = Path(shell("bundle show kramdown"))

    # Read emphasis source
    path = kramdown_path / "lib/kramdown/parser/kramdown/emphasis.rb"
    contents = path.read_text()

    # Patch EMPHASIS_START
    contents = re.sub(r"^(\s*EMPHASIS_START =).*$", r"\1 /(?:\\*\\*?)/", contents, flags=re.MULTILINE)

    # Write emphasis source
    path.write_text(contents)


if __name__ == "__main__":
    patch_jekyll()
