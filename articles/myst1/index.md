---
title: MyST Demo
subtitle: Demonstration of MyST Markdown Elements
published: "2024-11-23"
banner: "resources/banner.png"
draft: true
kernelspec:
  name: myst1
---

# Typography

## Paragraphs

Lorem ipsum odor amet, consectetuer adipiscing elit. Habitant quis dui fames eu a proin semper dictum. Magna efficitur interdum pharetra nisi mattis? Cubilia adipiscing vel viverra diam metus nascetur. Iaculis rutrum erat orci feugiat a euismod eu suscipit. Dolor rhoncus amet vitae lobortis augue. Primis at per ex at ultrices molestie cubilia aptent.

Tellus taciti efficitur felis vehicula aliquam vivamus aptent. Dictumst scelerisque torquent ullamcorper vestibulum finibus ipsum suspendisse. Tellus scelerisque ex proin tempus donec pharetra quam. Est ullamcorper magna metus quam posuere. Leo hendrerit laoreet facilisi vulputate quisque sodales integer at porta. Dui nascetur lobortis platea velit aliquet praesent justo lacinia non. Finibus in ex maecenas ex ipsum parturient lacinia.

Nibh nisi nunc hac nisl sed scelerisque, sociosqu vivamus. Eros nostra enim praesent sollicitudin pharetra. Finibus penatibus tristique sociosqu penatibus sollicitudin finibus diam eros habitasse. Orci volutpat varius lobortis egestas inceptos aliquet. Dignissim consectetur velit hendrerit quis accumsan lacus. Tempor scelerisque convallis mi himenaeos bibendum? Semper euismod pharetra tortor penatibus morbi. Malesuada ultrices montes mi semper; tincidunt diam.

## Inline Text Formatting

Standard inline formatting including bold, italic, code, as well as escaped symbols.

```markdown
**strong**, _emphasis_, `literal text`, \*escaped symbols\*
```

**strong**, _emphasis_, `literal text`, \*escaped symbols\*

## Line Breaks

To put a line break, without a paragraph, use a `\` followed by a new line.

```markdown
Fleas \
Adam \
Had 'em.

By Strickland Gillilan
```

Fleas \
Adam \
Had 'em.

By Strickland Gillilan

## Lists

You can use bullet points and numbered lists as you would in standard markdown. Starting a line with either a - or * for a bullet point, and 1. for numbered lists. These lists can be nested using two spaces at the start of the line.

```markdown
- Lists can start with `-` or `*`
  * My other, nested
  * bullet point list!

1. My numbered list
2. has two points
```

- Lists can start with `-` or `*`
  * My other, nested
  * bullet point list!

1. My numbered list
2. has two points

## Subscript & Superscript

Subscript and superscript are supported using `{sub}` and `{sup}` roles.

```markdown
H{sub}`2`O, and 4{sup}`th` of July
```

H{sub}`2`O, and 4{sup}`th` of July

## Abbreviations

```markdown
Well {abbr}`MyST (Markedly Structured Text)` is cool!
```

Well {abbr}`MyST (Markedly Structured Text)` is cool!

# Callouts & Admonitions

Admonitions include note, important, hint, seealso, tip, attention, caution, warning, danger, and error.

```markdown
:::{note}
This is a note.
:::
```

:::{note}
This is a note.
:::

```markdown
:::{warning}
This is a warning!
:::
```

:::{warning}
This is a warning!
:::

# Images and Figures

## Images

````markdown
```{image} https://placehold.co/600x400
:width: 300px
:height: 200px
:align: center
```
````

```{image} https://placehold.co/600x400
:width: 300px
:height: 200px
:align: center
```

## Figures

````markdown
```{figure} https://placehold.co/600x400
:label: fig1

Placeholder
```
````

```{figure} https://placehold.co/600x400
:label: fig1

Placeholder
```

# Math

## Inline Math

```markdown
This math is a role, {math}`e=mc^2`, while this math is wrapped in dollar signs, $Ax=b$.
```

This math is a role, {math}`e=mc^2`, while this math is wrapped in dollar signs, $Ax=b$.

## Equations

````markdown
```{math}
:label: eq1
w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
```
````

```{math}
:label: eq1
w_{t+1} = (1 + r_{t+1}) s(w_t) + y_{t+1}
```

# Tables

```markdown
| foo | bar |
| --- | --- |
| baz | bim |
```

| foo | bar |
| --- | --- |
| baz | bim |

# Executable Code

```{code-cell} python
x = 1
```

```{code-cell} python
y = 2
```

```{code-cell} python
x + y
```

```{code-cell} python
from matplotlib import pyplot as plt
import numpy as np

n = 100
x = np.random.normal(size=n)
y = np.random.normal(size=n)
plt.scatter(x, y)
```
