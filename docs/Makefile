PYEXDIR = python_examples
SPYEXDIR = src/$(PYEXDIR)

src/$(PYEXDIR)/%.md: $(PYEXDIR)/%.ipynb
	rm -rf src/$(PYEXDIR)/$*_files
	uv run jupyter nbconvert --to markdown --execute \
		--TagRemovePreprocessor.enabled=True \
		--TagRemovePreprocessor.remove_input_tags=hide-input \
		--output-dir=src/$(PYEXDIR) $<
	rm -f src/$(PYEXDIR)/$*_files/*.jpg

notebooks: $(patsubst $(PYEXDIR)/%.ipynb,src/$(PYEXDIR)/%.md,$(wildcard $(PYEXDIR)/*.ipynb))
