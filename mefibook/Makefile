PYEXDIR = python_examples

src/$(PYEXDIR)/%.md: $(PYEXDIR)/%.ipynb
	jupyter-nbconvert --to markdown --execute --output-dir=src/$(PYEXDIR) $<

notebooks: src/$(PYEXDIR)/umesh_basics.md src/$(PYEXDIR)/input_output.md src/$(PYEXDIR)/topological_tools.md src/$(PYEXDIR)/geometric_tools.md src/$(PYEXDIR)/selection.md
