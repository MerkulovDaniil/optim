---
layout: default
title: Home
nav_order: 1
permalink: /
---

# Introduction

Welcome to the fmin.xyz - attempt to build intuitive systematic review of optimization theory, methods and applications. The initial goal of creating literature review for PhD thesis has transformed to the idea of something, that could be useful for people, making comparison of different optimizers easier and nice.

The site contains following parts:

* {% include link.html title="Theory"%}
* {% include link.html title="Methods"%}
* {% include link.html title="Excersises"%}
* {% include link.html title="Applications"%}

# Rules of contributing
* All formulas (both inline and block style) should be arranged with double dollar sign due to `kramdown` preprocessor.
* Pictures and should be in `.svg` vector format if possible.
* Titles of the pages should start from the capital letter. Do not use abbreviation style of naming: `Gradient descent`✅, while `Gradient Descent`❎

# Principles

* This site aims on researchers and enthisiasts with some prior expertise in the field (not the Wikipedia clone)
* The fundamental primitive structure of this site is the page
* The page should consist short self-sufficient material about the topic
* If the page is based on paper, bibtex label should be added at the beginning
* The general structure of the site is flexible and should be improved over the time iteratively

# Notations and terms

These notations will be used by default (unless otherwise specified)

* $$\theta$$ -- optimizing variable
* $$\mathbb{R}^p$$ -- $$p$$-dimensional Euclidian space
* $$\theta^*$$ -- global optimizer of the problem
* $$\| \cdot \|$$ -- norm of the vector(matrix) (if nothing specified, we often use Euclidian (Frobenius) norm)
* $$\| \cdot \|_*$$ -- dual norm (norm in dual space) (if nothing specified, we often use Euclidian (Frobenius) norm)
