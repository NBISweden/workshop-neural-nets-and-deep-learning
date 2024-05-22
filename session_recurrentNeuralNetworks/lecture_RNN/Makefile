all: index.html slides.pdf lecture_rnn.zip

%.html: %.qmd
	quarto render $<

screenshots/slides_1_1280x720.png: index.qmd
	@./node_modules/.bin/decktape http://localhost:8888/index.html slides.pdf --screenshots

slides.pdf: screenshots/slides_1_1280x720.png
	@SLIDES=`find screenshots -type f -name "slides_*.png" | wc -l`; \
	args=`for i in $$(seq 1 $$SLIDES); do echo screenshots/slides_$${i}_1280x720.png; done`; \
	convert $$args slides.pdf

lecture_rnn.zip: index.html
	zip -r lecture_rnn.zip index.html index_files index_cache airline*.png

clean:
	rm -rf index_cache
	rm -rf *.log *.dvi *.aux
	rm -rf Untitled*
	rm -rf __pycache__ auto

gitclean:
	git clean -f -d -x
