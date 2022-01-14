# Common assets folder

## Course conda environment `nn_dl_python`

- `conda_envs`
  - `README.md`
  - `nn_dl_python.yaml`
  - `nn_dl_python_win.yaml`
  - `test_imports.py

See subfolder README

## Tools for converting ipynb to html

This works well for standard notebooks, but is unstable and very
tricky if, e.g., nbextensions `Hide input` or `Split cells Notebook`
are used. Also, cells marked as `Skip` in presentations may cause
problems (tip_ try changing to `Notes`). Preliminary observations
suggest that images in format `png` works well, while `jpg` and
pdf` does not.

### A. Use the following with mnbconvert versions pre 6.0, tested with nbconvert version 5.6.1

- `ipynb_2_html.sh`
- `convert_html_to_standalone.py`

#### Usage:

must be run from this folder it seems

```
bash ipynb_2_html.sh <my.ipynb>
```

This will convert `<my.ipynb>` to html using nbconvert and then run
`convert_html_to_standalone.py` to make the converted html
self-contained. The script will automatically recognize if
`<my.ipynb>` is a presentation (output will be `<my.slides.html>` or a
standard nb (output will be `<my.html>`.

### B. Use the following with nbconvert version post 6.0. (e.g., as included in the course conda_env `nn_dl_python`).

- `reveal_custom` folder

#### Usage:

must be run from this folder it seems

```
nbconvert <my.ipynb> --to html --template reveal_custom
```

or

```
nbconvert <my.ipynb> --to slides --template reveal_custom
```

(it's also possible to add `--execute` to force exectuion of the notebook before vonversion).

For making html slef-contained, current state (a hack) is to:

1.convert to html as described just above

2. comment out the nbconvert commnads in `ipynb_2_html.sh` and then run it as usual this script as described unde A. above.

(Tip: Changing the jnb-sizes (to samller) might solve some spacing problems)


## A tool for drawing simple neural nets

Python library for drawing ball-and-string type of neyral network images:

- `draw_neural_net.py`
- `demo_draw_neural_net.ipynb`

#### Usage:

See `demo_draw_neural_net.ipynb`
