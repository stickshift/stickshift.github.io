from datetime import datetime
import logging
from logging import StreamHandler
from operator import itemgetter
from pathlib import Path
import sys

import click
from jinja2 import Environment, FileSystemLoader

from stickshift import myst

__all__ = [
    "build_index",
]

logger = logging.getLogger(__name__)

_default_site_front_matter = {
    "title": "Pattern Recognition",
    "root_url": "/index.html",
    "repo_url": "https://github.com/stickshift/stickshift.github.io",
}

_author = {
    "name": "Andrew Young",
    "bio": "Software architect, AI researcher, entrepreneur, and startup veteran with a passion for creativity and good coffee.",
    "github": "stickshift",
    "linkedin": "andrewsomesyoung",
}

_default_article_front_matter = {
    "title": "Article Title",
    "subtitle": "Article Subtitle",
    "published": "2000-01-01",
    "author": _author,
    "draft": False,
}


@click.command("build-index")
@click.option(
    "--theme",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to theme.",
)
@click.argument(
    "output-path",
    type=click.Path(path_type=Path),
)
def build_index(theme: Path, output_path: Path):
    """Generate site index."""
    # Initialize logging
    _configure_logging()

    # Parse front matter
    site = _default_site_front_matter | {
        "stylesheets": [
            "/styles/fonts.css",
            "/styles/pygments.css",
            f"/styles/{theme.stem}.css",
        ],
    }

    articles_path = Path("articles")

    # Load all articles
    articles = []
    for article_path in articles_path.iterdir():
        if not article_path.is_dir():
            continue

        article_id = article_path.name

        # Read article index
        index_path = article_path / "index.md"
        if not index_path.exists():
            raise ValueError(f"index.md not found for article {article_id}")

        # Parse article metadata
        article = _default_article_front_matter | myst.front_matter(index_path.read_text())
        article["id"] = article_id
        articles.append(article)

    logger.info(f"Found {len(articles)} articles")

    # Filter out drafts
    before_count = len(articles)
    articles = [a for a in articles if a["draft"] is False]
    after_count = len(articles)
    logger.info(f"Filtered {before_count - after_count} drafts")

    # Sort articles by published date in descending order
    articles.sort(key=itemgetter("published"), reverse=True)

    # Configure jinja env
    env = Environment(loader=FileSystemLoader(theme / "templates"))

    # Add custom filter for date formatting
    def format_date(value: str):
        return f"{datetime.strptime(value, '%Y-%m-%d').date():%B %-d, %Y}"

    env.filters["format_date"] = format_date

    # Lookup index template
    template = env.get_template("index.html.j2")

    # Apply template
    context = {
        "site": site,
        "page": {
            "author": _author,
            "title": "Home",
            "articles": articles,
            "description": "Pattern Recognition - Writing on AI/ML research and related technology topics by Andrew Young.",
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
    build_index()
