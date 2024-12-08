import logging
from logging import StreamHandler
from pathlib import Path
import sys

import click
from jinja2 import Environment, FileSystemLoader

from stickshift import myst

__all__ = [
    "build_article",
]

logger = logging.getLogger(__name__)

_default_site_front_matter = {
    "title": "Pattern Recognition",
    "root_url": "/index.html",
    "repo_url": "https://github.com/stickshift/stickshift.github.io",
}

_default_article_front_matter = {
    "title": "Article Title",
    "subtitle": "Article Subtitle",
    "published": "January 1, 2000",
    "author": {
        "name": "Andrew Young",
        "bio": "Software architect, AI researcher, entrepreneur, and startup veteran with a passion for creativity and good coffee.",
        "github": "stickshift",
        "linkedin": "andrewsomesyoung",
    },
}


@click.command("build-article")
@click.option(
    "--theme",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to theme.",
)
@click.argument(
    "input-path",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "output-path",
    type=click.Path(path_type=Path),
)
def build_article(theme: Path, input_path: Path, output_path: Path):
    """Transform article from MyST notebook into HTML."""
    # Initialize logging
    _configure_logging()

    # Configure jinja env
    env = Environment(loader=FileSystemLoader(theme / "templates"))

    # Lookup article template
    template = env.get_template("article.html.j2")

    # Load markdown
    markdown = input_path.read_text()

    # Parse front matter
    site_front_matter = _default_site_front_matter | {
        "stylesheets": [
            "/styles/pygments.css",
            f"/styles/{theme.stem}.css",
        ],
    }
    article_front_matter = _default_article_front_matter | myst.front_matter(markdown)
    logger.info("Parsed article front matter")

    # Parse toc
    toc = myst.toc(markdown)
    logger.info("Parsed article toc")

    # Render markdown as html
    body = myst.render(markdown)
    logger.info("Rendered article markdown")

    # Apply template
    context = {
        "site": site_front_matter,
        "page": {
            **article_front_matter,
            "toc": toc,
            "body": body,
        },
    }
    html = template.render(context)

    # Stream html to stdout
    if str(output_path) == "-":
        click.echo(html)
        return

    # Save HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)

    # Summarize
    logger.info(f"Saved {len(html)} bytes to {output_path}")


def _configure_logging():
    # Stream all log messages to stderr
    handler = StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logging.root.addHandler(handler)

    # Bump root level to INFO
    logging.root.setLevel(logging.INFO)


if __name__ == "__main__":
    build_article()
