---
layout: default
title: Home
nav_order: 1
permalink: /
---

# Introduction

Welcome to the fmin.xyz - attempt to build intuitive systematic review of optimization theory, methods and applications. The initial goal of creating literature review for PhD thesis has transformed to the idea of something, that could be useful for people, making comparison of different optimizers easier and nice.

[![](/assets/images/discord.svg)](https://discord.gg/vQv7Gfv)

The site contains following parts:

* {% include link.html title="Theory"%}
* {% include link.html title="Methods"%}
* {% include link.html title="Excersises"%}
* {% include link.html title="Applications"%}

The site has simple built-in search 🔍. You can use it to find anything you want.

# Rules of formatting and contributing
## Math blocks
* All materials are available in `kramdown` version of Markdown. It is pretty simple, some syntax cheatsheet could be found on the [site](https://kramdown.gettalong.org/syntax.html).
* All formulas (both inline and block style) should be arranged with double dollar sign (`$$ x^2 $$` stands for inline $$x^2$$, if you'll arrange it with new line symbols, you'll recieve block style) due to `kramdown` preprocessor.
* If you want to place just one *inline* formula on the paragrah, please write this way: `\$$ x^2 $$` (weird, I know).

## Figures
* Figures and graphs should be in `.svg` vector format if possible.
* By default all pictures are centered and has 75% of linewidth. However, if you will use `#button` suffix at the end of the path (or url) to the picture, it will be displayed in inline style with 150px in width (was primarly done for [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)]() buttons).

## Names
* Titles of the pages should start from the capital letter. Do not use abbreviation style of naming: `Gradient descent`✅, while `Gradient Descent`❎
* You can include the link to some page of the site right in the text through the following syntax: `{% raw %}{% include link.html title="Excersises"%}{% endraw %}`, where you should place only the name of the page in title field. One more example: `{% raw %}{% include link.html title="Gradient descent"%}{% endraw %}`. It is useful, because it is robust to the structure changes in this site.

# Principles

* This site aims on researchers and enthisiasts with some prior expertise in the field (not just the Wikipedia clone).
* The fundamental primitive structure of this site is the page.
* The page should consist short self-sufficient material about the topic.
* If the page is based on paper, bibtex label should be added at the beginning.
* The general structure of the site is flexible and should be improved over the time iteratively.

# Notations and terms

These notations will be used by default (unless otherwise specified)

* $$\theta$$ -- optimizing variable
* $$\mathbb{R}^p$$ -- $$p$$-dimensional Euclidian space
* $$\theta^*$$ -- global optimizer of the problem
* $$\| \cdot \|$$ -- norm of the vector(matrix) (if nothing specified, we often use Euclidian (Frobenius) norm)
* $$\| \cdot \|_*$$ -- dual norm (norm in dual space) (if nothing specified, we often use Euclidian (Frobenius) norm)
