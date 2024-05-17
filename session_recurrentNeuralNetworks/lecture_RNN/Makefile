all: index.html

%.html: %.qmd
	quarto render $<

clean:
	rm -rf index_cache
	rm -rf *.log *.dvi *.aux
	rm -rf Untitled*
	rm -rf __pycache__ auto

gitclean:
	git clean -f -d -x
