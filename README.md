# Optim
💎A site, that contains systematic optimization methods review

## Contribution
### Guide
1. Fork this repository to your own account (Fork button at the top right corner)
1. Create a branch in your repository (locally or remotely). Let's name it `contribution` branch 
1. Commit your changes\ fixes\ additions to this branch
1. Send your pull request to the `master` branch of the original repository.

### Formatting rules
#### Math blocks
* All materials are available in `kramdown` version of Markdown. It is pretty simple, some syntax cheatsheet could be found on the [site](https://kramdown.gettalong.org/syntax.html).
* All formulas (both inline and block style) should be arranged with double dollar sign (`$$ x^2 $$` stands for inline $$x^2$$, if you'll arrange it with new line symbols, you'll recieve block style) due to `kramdown` preprocessor.
* If you want to place just one *inline* formula on the paragrah, please write this way: `\$$ x^2 $$` (weird, I know).

#### Figures
* Figures and graphs should be in `.svg` vector format if possible.
* By default all pictures are centered and has 75% of linewidth. However, if you will use `#button` suffix at the end of the path (or url) to the picture, it will be displayed in inline style with 150px in width (was primarly done for [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)]() buttons).

#### Names
* Titles of the pages should start from the capital letter. Do not use abbreviation style of naming: `Gradient descent`✅, while `Gradient Descent`❎, `GD`❎
* You can include the link to some page of the site right in the text through the following syntax: `{% raw %}{% include link.html title="Excersises"%}{% endraw %}`, where you should place only the name of the page in title field. One more example: `{% raw %}{% include link.html title="Gradient descent"%}{% endraw %}`. It is useful, because it is robust to the structure changes in this site. 

#### Buttons
* `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)](%LINK TO THE SHARED COLAB NOTEBOOK%) buttons)` stands for the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)]() buttons) 
* `{% include tabs.html bibtex = '@article{kingma2014adam,
  title={Adam: A method for stochastic optimization},
  author={Kingma, Diederik P and Ba, Jimmy},
  journal={arXiv preprint arXiv:1412.6980},
  year={2014}
}' file='PATH_TO_THE_FILE'%}` will display
    ![](/assets/images/bibtex_button.png)

## References
* The jekyll template was taken from [Just the docs](https://github.com/pmarsceill/just-the-docs) and modified
