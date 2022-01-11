#!/bin/bash

notebooks=$(find ../ -name "*.ipynb" -not -path '*/.*')

for notebook in $notebooks; do
    isslide=$(grep "\"slide_type\": \"slide\"" $notebook)
    
    if [[ ! -z $isslide ]]; then
        jupyter nbconvert --to slides $notebook --template slides_reveal_split.tpl
    else
        jupyter nbconvert --to html $notebook
    fi
done

slides=$(find ../ -name "*.slides.html" -not -path '*/.*')
scriptspath=$(pwd)
for slide in $slides; do
    echo "Converting $slide to self-contained"
    todir=$(dirname $slide)
    slidefile=$(basename $slide)
    cd $todir
    python $scriptspath/convert_html_to_standalone.py --infile $slidefile --outfile $slidefile
    cd -
done
