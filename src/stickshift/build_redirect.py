import logging
from logging import StreamHandler
from pathlib import Path
import sys

import click
from jinja2 import Environment, FileSystemLoader

from stickshift import myst

__all__ = [
    "build_redirects",
]

logger = logging.getLogger(__name__)


@click.command("build-redirect")
@click.option(
    "--theme",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to theme.",
)
@click.argument(
    "target-url",
    type=str,
)
@click.argument(
    "output-path",
    type=click.Path(path_type=Path),
)
def build_redirects(theme: Path, target_url: str, output_path: Path):
    """Redirect old links to new ones."""
    # Initialize logging
    _configure_logging()

    # Configure jinja env
    env = Environment(loader=FileSystemLoader(theme / "templates"))

    # Lookup redirect template
    template = env.get_template("redirect.html.j2")

    # Prepare
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Render template
    html = template.render(target_url=target_url)

    # Write
    output_path.write_text(html)

    # Summarize
    logger.info(f"Wrote {len(html)} bytes to {output_path}")


def _configure_logging():
    # Stream all log messages to stderr
    handler = StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logging.root.addHandler(handler)

    # Bump root level to INFO
    logging.root.setLevel(logging.INFO)


if __name__ == "__main__":
    build_redirects()
