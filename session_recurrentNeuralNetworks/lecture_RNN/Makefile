all: lecture.html lecture.serve.html

%.pdf: %.tex
	xelatex $<

%.svg: %.tex
	xelatex $<

lecture.serve.Rmd: lecture.Rmd
	@cat $< | sed -e "s/self_contained: true/self_contained: false/" | sed -e "s/reveal_plugins: \[\]/reveal_plugins: \[\"notes\"\]/" > $@

%.html: %.Rmd
	R -e "library('rmarkdown'); rmarkdown::render('$<', output_file='$@')"

clean:
	rm -rf lecture_cache
	rm -rf lecture.serve_cache
	rm -rf *.log *.dvi *.aux
	rm -rf Untitled*
	rm -rf __pycache__ auto
