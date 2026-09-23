SPYEXDIR = .

# --- pandoc slide decks -----------------------------------------------------------
# `make slides`      -> reveal.js HTML deck: interactive, self-contained (4 MB),
#                       works offline (reveal.js JS/CSS + images are embedded).
# `make slides-pdf`  -> beamer PDF deck. Requires a TeX engine with the `pgf`/
#                       tikz packages installed (e.g. `sudo apt install texlive-pictures`)
#                       and lualatex for the ×/·/→ characters.
SLIDES ?= mefikit_vs_medcoupling_short
SLIDES_SRC  = $(SPYEXDIR)/$(SLIDES).md
SLIDES_CSS  = $(SPYEXDIR)/slides.css
SLIDES_HTML = $(SLIDES).html
SLIDES_PDF  = $(SLIDES).pdf
PANDOC      ?= pandoc
REVEALJS_URL = https://unpkg.com/reveal.js@4.6.1

.PHONY: slides slides-html slides-pdf
slides: slides-html

slides-html: $(SLIDES_HTML)

slides-pdf: $(SLIDES_PDF)

$(SLIDES_HTML): $(SLIDES_SRC) $(SLIDES_CSS)
	$(PANDOC) -t revealjs --standalone --embed-resources --slide-level=2 \
		--resource-path=$(SPYEXDIR) \
		--include-in-header=$(SLIDES_CSS) \
		-V revealjs-url=$(REVEALJS_URL) \
		-V theme=black -V slide-number=true \
		-o $@ $(SLIDES_SRC)

$(SLIDES_PDF): $(SLIDES_SRC)
	$(PANDOC) -t beamer --standalone --slide-level=2 \
		-V theme=CEA23 \
		-o $@ $(SLIDES_SRC)
