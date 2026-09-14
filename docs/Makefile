PYEXDIR = python_examples

src/$(PYEXDIR)/%.md: $(PYEXDIR)/%.ipynb
	rm -rf src/$(PYEXDIR)/$*_files
	uv run jupyter nbconvert --to markdown --execute --output-dir=src/$(PYEXDIR) $<

notebooks: src/$(PYEXDIR)/umesh_basics.md src/$(PYEXDIR)/extrusions.md src/$(PYEXDIR)/input_output.md src/$(PYEXDIR)/topological_tools.md src/$(PYEXDIR)/geometric_tools.md src/$(PYEXDIR)/selection.md src/$(PYEXDIR)/fields.md src/$(PYEXDIR)/transfers.md src/$(PYEXDIR)/example_bubbles.md src/$(PYEXDIR)/compare_medcoupling.md
