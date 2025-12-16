PYEXDIR = python_examples

src/$(PYEXDIR)/%.md: $(PYEXDIR)/%.ipynb
	jupyter-nbconvert --to markdown --execute --output-dir=src/$(PYEXDIR) $<

notebooks: src/$(PYEXDIR)/umesh_basics.md src/$(PYEXDIR)/neighbours_utilities.md src/$(PYEXDIR)/input_output.md
