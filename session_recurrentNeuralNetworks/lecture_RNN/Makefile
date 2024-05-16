all: lecture.html lecture.qserve.html

%.pdf: %.tex
	xelatex $<

%.svg: %.tex
	xelatex $<

lecture.qserve.qmd: lecture.qmd
	@cat $< | sed -e "s/self-contained: true/self-contained: false/" | sed -e "s/chalkboard: false/chalkboard: true/" > $@

lecture.serve.Rmd: lecture.Rmd
	@cat $< | sed -e "s/self_contained: true/self_contained: false/" | sed -e "s/reveal_plugins: \[\]/reveal_plugins: \[\"notes\"\]/" > $@

%.html: %.qmd
	quarto render $<

clean:
	rm -rf lecture_cache
	rm -rf lecture.serve_cache
	rm -rf *.log *.dvi *.aux
	rm -rf Untitled*
	rm -rf __pycache__ auto

gitclean:
	git clean -f -d -x
