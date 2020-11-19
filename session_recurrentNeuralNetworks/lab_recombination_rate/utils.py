#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import urllib3
import ipywidgets as widgets
import IPython.display as display

http = urllib3.PoolManager()

def sidebyside(*args, is_remote=True, stack=False, **kwargs):
    """Display remote images side by side

    :param bool is_remote: operate on remote files with urllib3
    :param bool stack: stack images vertically
    """
    if is_remote:
        img = [http.request('GET', x).data for x in args]
    else:
        img = [open(x, 'rb').read() for x in args]
    wdgt = [widgets.Image(value=x, width=kwargs.get("width", 300)) for x in img]
    if stack:
        imggroup = widgets.VBox(wdgt)
    else:
        imggroup = widgets.HBox(wdgt)
    return display.display(imggroup)


def write_vcf(ts, filename):
    """Write tree sequence to bgzip-compressed vcf and index file

    :param tskit.trees.TreeSequence ts: The tree sequence object to save
    :param str filename: output file name
    """
    read_fd, write_fd = os.pipe()
    write_pipe = os.fdopen(write_fd, "w")
    with open(filename, "w") as fh:
        proc = subprocess.Popen(
            ["bcftools", "view", "-O", "z"], stdin=read_fd, stdout=fh
        )
        ts.write_vcf(write_pipe, ploidy=2)
        write_pipe.close()
        os.close(read_fd)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("bcftools view failed with status:", proc.returncode)
    proc = subprocess.Popen(
        ["bcftools", "index", filename]
    )
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("bcftools view failed with status:", proc.returncode)
