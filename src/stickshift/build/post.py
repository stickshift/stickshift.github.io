"""Build a Jekyll post from Jupyter notebook."""

from pathlib import Path
import re
import shutil
from textwrap import dedent

import click

from stickshift import md5, shell


@click.command()
@click.option("--notebook", type=click.Path(exists=True, path_type=Path))
@click.option("--markdown", type=click.Path(path_type=Path))
def build_post(notebook: Path, markdown: Path):
    """Build a Jekyll post from Jupyter notebook."""
    # Create target directory
    markdown.parent.mkdir(parents=True, exist_ok=True)

    # Replace target if it exists
    markdown.unlink(missing_ok=True)

    # Front matter
    content = "---\n"
    content += "layout: post\n"

    value = shell(f"cat {notebook} | jq -r '.metadata.stickshift.title'")
    if value != "null":
        content += f"title: \"{value}\"\n"

    value = shell(f"cat {notebook} | jq -r '.metadata.stickshift.description'")
    if value != "null":
        content += f"description: \"{value}\"\n"

    value = shell(f"cat {notebook} | jq -r '.metadata.stickshift.image'")
    if value != "null":
        content += f"image: \"{value}\"\n"

    content += "---\n"

    markdown.write_text(content)

    # Content
    build_path = Path(".build") / markdown.stem
    build_path.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"jupyter nbconvert"
        f" --to markdown"
        f" --TagRemovePreprocessor.enabled=True"
        f" --TagRemovePreprocessor.remove_cell_tags skip-publish"
        f" --output-dir {build_path}"
        f" {notebook}"
    )
    shell(cmd)

    # Read content in
    content = (build_path / markdown.name).read_text()

    # Images
    images_path = build_path / f"{markdown.stem}_files"
    if images_path.exists():
        for f in images_path.iterdir():
            if f.is_file():
                # Copy image to markdown directory
                target_path = markdown.parent / f"{md5(f.name)}.png"
                shutil.copy(f, target_path)

                # Replace reference in markdown
                pattern = rf"{markdown.stem}_files/.*\.png"
                content = re.sub(pattern, target_path.name, content)

    # Append markdown content to final location
    with markdown.open(mode="a") as f:
        f.write(content)


if __name__ == "__main__":
    build_post()
