import bs4
import os
import base64
import argparse

def mkParser():
    parser     = argparse.ArgumentParser(description = "Convert html document to stand alone html")
    parser.add_argument("--infile",     type = str,    required = True,    help="input html file")
    parser.add_argument("--outfile",    type = str,    required = True,    help="name of output file")
    parser.add_argument("--css",        type = str,    required = False,   help="a css file")

    return parser.parse_args()

def convert(infile, outfile, css = False):
    with open(infile) as inf:
        txt  = inf.read()
        soup = bs4.BeautifulSoup(txt, 'html.parser')
        
    for a in soup.find_all('img'):
        src = a.attrs['src']
        if os.path.exists(src):
            with open(src, "rb") as img_file:
                prepend = "data:image/jpeg;base64,"
                b64     = str(base64.b64encode(img_file.read()).decode())
            a['src'] = prepend+b64
        else:
            print("Couldn't find file: "+src)

    if css:
        with open(css, 'r') as incss:
            content = incss.read()
        style_tags     = soup.find_all('style')
        last_style     = style_tags[-1]
        new_tag        = soup.new_tag('style', type="text/css")
        new_tag.string = content
        soup.body.append(new_tag)

    
    with open(outfile, "w") as outf:
        outf.write(str(soup))


def main():
    args = mkParser()
    convert(args.infile, args.outfile, args.css)
    

if __name__ == "__main__":
    main()
