.PHONY: clean
all: slides.pdf

slides.pdf: slides.tex
	latexmk -pdf
	-sed -i 's/\\\\/\n/g' slides.pdfpc
	-sed -i 's/\\par/\n\n/g' slides.pdfpc

slides.tex: slides.md
	pandoc -s -f markdown -t beamer --lua-filter ./notes.lua -o $@ $<

clean:
	latexmk -C
	rm -f slides.pdfpc slides.tex
