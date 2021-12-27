all:
	R -e "library('rmarkdown'); rmarkdown::render('lecture.Rmd')"

%.pdf: %.tex
	xelatex $<

clean:
	rm -rf lecture_cache
	rm -rf *.log
