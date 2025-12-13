PYEXDIR = python_examples

src/$(PYEXDIR)/%.md: $(PYEXDIR)/%.ipynb
	jupyter-nbconvert --to markdown --execute --output-dir=src/$(PYEXDIR) $<

notebooks: src/$(PYEXDIR)/*.md
