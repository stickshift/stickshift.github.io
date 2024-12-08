# Pattern Recognition

> Writing on AI/ML research and related technology topics by Andrew Young.

## Toolchain

Site is generated using a variety of tools from the Executable Books ecosystem. 

* **MyST Markdown**: All articles are written in Markdown notebooks.
* **Jupyter:** Markdown notebooks are transformed into Jupyter notebooks and executed. 
* **HTML:** Jupyter notebooks are transformed into HTML pages using Jinja templates.

## Content Structure

### Articles

Site content is organized as a series of independent articles represented as files stored in directory `articles/{article_id}`. Articles are all written in MyST formatted executable Markdown `articles/{article_id}/index.md`.

Each article has it's own Python venv with it's own set of dependencies managed by `articles/{article_id}/pyproject.toml` in the article directory. This allows the articles to explore arbitrary approaches without conflicting with each other. It also prevents later articles from inadvertently breaking earlier ones. Each article's venv is represented as a separate Jupyter kernel which allows Jupyter to access the article-specific venvs.

#### Front Matter

Each article has a front matter section at the top of `articles/{article_id}/index.md`. For the kernel specs to work, you need to specify the article_id as the kernel name.

```yaml
---
title: Lorem Ipsum
subtitle: Dolor Sit Amet Consectetur
kernelspec:
  name: article_xyz
  display_name: article_xyz
---
```

#### Managing Dependencies

Article dependencies are managed by the `articles/{article_id}/pyproject.toml` and `articles/{article_id}/uv.lock` files.

```shell
# Add package to article venv
uv add {package_name} --directory articles/{article_id}

# Remove package from article venv
uv remove {package_name} --directory articles/{article_id}
```

## Build and Test

```shell
source environment.sh

# Build article artifacts
make articles

# Assemble artifacts into final site
make site

# Serve local site
make deploy
```

## License

This work is licensed under CC BY-SA 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by-sa/4.0/