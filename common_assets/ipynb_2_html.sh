#!/bin/bash

if [ -n "$1" ]; then
    notebooks=$1
else
    notebooks=$(find ../ -name "*.ipynb" -not -path '*/.*')
fi

scriptspath=$(pwd)

for notebook in $notebooks; do
    isslide=$(grep "\"slide_type\": \"slide\"" $notebook)
    
    if [[ ! -z $isslide ]]; then
        jupyter nbconvert --to slides $notebook --template slides_reveal_split.tpl
	slide=$(echo $notebook| sed 's/.ipynb/.slides.html/')
	echo $slide
    else
	echo "nb"
        jupyter nbconvert --to html $notebook --template=nbextensions
	slide=$(echo $notebook| sed 's/.ipynb/.html/')
	echo $slide
    fi

    echo "Converting $slide to self-contained"
    todir=$(dirname $slide)
    slidefile=$(basename $slide)
    cd $todir
    python $scriptspath/convert_html_to_standalone.py --infile $slidefile --outfile $slidefile
    cd -
done
