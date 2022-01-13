#!/bin/bash

if [ -n "$1" ]; then
    notebooks=$1
    slides=$(echo $1| sed 's/.ipynb/.slides.html/')
else
    notebooks=$(find ../ -name "*.ipynb" -not -path '*/.*')
    slides=$(find ../ -name "*.slides.html" -not -path '*/.*')
fi

for notebook in $notebooks; do
    isslide=$(grep "\"slide_type\": \"slide\"" $notebook)
    
    if [[ ! -z $isslide ]]; then
        jupyter nbconvert --to slides $notebook --template slides_reveal_split.tpl
    else
	echo "nb"
        jupyter nbconvert --to html $notebook --template=nbextensions
	slides=""
    fi
done

scriptspath=$(pwd)
for slide in $slides; do
    echo "Converting $slide to self-contained"
    todir=$(dirname $slide)
    slidefile=$(basename $slide)
    cd $todir
    python $scriptspath/convert_html_to_standalone.py --infile $slidefile --outfile $slidefile
    cd -
done
