#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import subprocess
import tskit
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


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


# FIXME: vectorize
def pad(haps, pos, maxsites=None, frameWidth=0):
    """Pad haplotypes and positions.

    :param np.ndarray haps: genotype matrix
    :param np.array pos: position vector
    :param int maxsites: maximum number of sites in any of the train, test, or vali data sets
    :param int frameWidth: pad sites with a frame in all directions (why?)

    :return (np.ndarray, np.array): tuple of genotype matrix, positions array
    """
    nsites = haps.shape[0]
    padlength = maxsites - nsites
    haps = np.pad(haps, ((0, padlength), (0,0)), "constant", constant_values=2.0)
    pos = np.pad(pos, (0, padlength), "constant", constant_values=-1.0)
    haps = np.array(haps, dtype = "float32")
    pos = np.array(pos, dtype="float32")
    if(frameWidth):
        fw = frameWidth
        haps = np.pad(haps,((fw,fw), (fw,fw)),"constant",constant_values=2.0)
        pos = np.pad(pos,((fw,fw)),"constant",constant_values=-1.0)

    return haps, pos

def fill_pad(haps, pos, maxlen):
    """Fill genotype matrix with ancestral sites. For illustration of
    entire sequence alignment only

    """
    sites = [0] + [int(p) for p in pos] + [maxlen]
    nsamples = haps.shape[1]
    padwidth = sites[:-1]
    for i in range(1, len(sites)):
        padwidth[i-1] = sites[i] - sites[i-1] - 1
    x = np.zeros((padwidth[0], nsamples)).astype("int8")
    for i in range(1, len(padwidth)):
        x = np.append(x, haps[i-1:i, :], axis=0)
        x = np.append(x, np.zeros((padwidth[i], nsamples)).astype("int8"), axis=0)
    return x

# See https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class sequenceBatchGenerator(tf.keras.utils.Sequence):
    """Generator class to provide data in batches"""

    def __init__(self, datadir, maxlen, batchSize=16, frameWidth=0, seed=None):
        self.datadir = datadir
        self.params = pd.read_table(os.path.join(datadir, "params.txt"))
        self.n_data = self.params.shape[0]
        self.maxlen = maxlen
        self.batchSize = batchSize
        self.frameWidth = frameWidth
        self.on_epoch_end()

    def on_epoch_end(self):
        """Shuffle indices to randomize order of training examples on epoch end"""
        self.indices = np.arange(self.n_data)
        np.random.shuffle(self.indices)

    def __len__(self):
        """Get the number of batches per epoch as the number of data points divided by batch size"""
        return int(np.floor(self.n_data / self.batchSize))

    def __getitem__(self, index):
        """Generate data indices for a given batch index (where index <= len(self))"""
        batchIndices = self.indices[index * self.batchSize: (index+1) * self.batchSize]
        X, y = self.__data_generation(batchIndices)
        return X, y

    def __data_generation(self, batchIndices):
        """Required core function to generate data batch"""
        rho = np.reshape(self.params["rho"].to_numpy(), (-1, 1))
        rhoZ = (rho - np.mean(rho, axis=0)) / np.std(rho, axis=0)
        haps = []
        pos = []
        # Read one dataset at a time
        for i in batchIndices:
            H = np.load(os.path.join(self.datadir, f"{i}_haps.npy"))
            P = np.load(os.path.join(self.datadir, f"{i}_pos.npy"))
            # Any shuffling of individuals, reorganization of
            # haplotypes takes place here
            hpad, ppad = pad(H, P, self.maxlen, frameWidth=self.frameWidth)
            haps.append(hpad)
            pos.append(ppad)
        haps = np.array(haps, dtype="float32") # (batchSize, nSites, nsamples)
        pos = np.array(pos, dtype="float32") # (batchSize, nSites)
        return [haps, pos], np.array(rhoZ[batchIndices]) # X, y


def simulate(nreps, outdir, seed=42, murange=(0, 2e-7), rhorange=(0, 2e-7), sample_size=10, Ne=1e4, length=1e4, **kwargs):
    """Simulate tree sequences with msprime and save genotype matrix and positions to outdir

    :params int nreps: number of simulations
    :params str outdir: output directory
    :params int seed: seed for simulations
    :params tuple murange: range of mutation values
    :params tuple rhorange: range of recombination values
    :params int sample_size: number of chromosomes to sample
    :params int Ne: effective population size
    :params int length: sequence length
    :params dict kwargs: keyword arguments passed to msprime
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    mu = [np.random.uniform(murange[0], murange[1]) for _ in range(nreps)]
    rho = [np.random.uniform(rhorange[0], rhorange[1]) for _ in range(nreps)]
    segregating_sites = np.empty(nreps, dtype="int64")
    fh = open(os.path.join(outdir, "params.txt"), "w")
    fh.write("ID\tmu\trho\tS\tpi\n")
    for i in range(nreps):
        ts = msprime.simulate(random_seed=seed, sample_size=sample_size, Ne=Ne, length=length,
                              mutation_rate=mu[i], recombination_rate=rho[i], **kwargs)
        np.save(os.path.join(outdir, f"{i}_haps.npy"), ts.genotype_matrix())
        np.save(os.path.join(outdir, f"{i}_pos.npy"), np.array([s.position for s in ts.sites()], dtype="float32"))
        segregating_sites[i] = ts.get_num_sites()
        fh.write(f"{i}\t{mu[i]}\t{rho[i]}\t{ts.get_num_sites()}\t{ts.diversity()}\n")
    fh.close()


def GRU(x, y):
    """Gated recurrent unit model.

    Based on ReLERNN.networks.GRU_TUNED84

    :params np.ndarray x: haplotype and positions batches
    :params np.ndarray y: target values
    """
    haps, pos = x
    nsites = haps[0].shape[0]
    nsamples = haps[0].shape[1]
    npos = pos[0].shape[0]

    # Define input layer that takes a genotype matrix; recall that genotype matrix stores data as (nsites, nsamples)
    genotype_inputs = layers.Input(shape=(nsites, nsamples))
    # analyze sequences in both directions, GRU more computationally efficient than LSTM
    model = layers.Bidirectional(layers.GRU(21, return_sequences=False))(genotype_inputs)
    # Add dense layer with 256 neurons in output space
    model = layers.Dense(64)(model)

    # Define input layer for positions
    position_inputs = layers.Input(shape=(npos,))
    m2 = layers.Dense(64)(position_inputs)

    # Concatenate genotype matrix and position input layers
    model = layers.concatenate([model, m2])
    # Add Dense and output layer that outputs a (normalized) recombination value
    model = layers.Dense(16)(model)
    output = layers.Dense(1)(model)

    model = Model(inputs=[genotype_inputs, position_inputs], outputs=[output])
    model.compile(optimizer="Adam", loss="mse")
    model.summary()

    return model


def GRU_DROPOUT(x, y):
    """Gated recurrent unit model, including dropout layers.

    Based on ReLERNN.networks.GRU_TUNED84

    :params np.ndarray x: haplotype and positions batches
    :params np.ndarray y: target values
    """
    haps, pos = x
    nsites = haps[0].shape[0]
    nsamples = haps[0].shape[1]
    npos = pos[0].shape[0]

    # Define input layer that takes a genotype matrix; recall that genotype matrix stores data as (nsites, nsamples)
    genotype_inputs = layers.Input(shape=(nsites, nsamples))
    # analyze sequences in both directions, GRU more computationally efficient than LSTM
    model = layers.Bidirectional(layers.GRU(21, return_sequences=False))(genotype_inputs)
    # Add Dense layer with 256 neurons in output space
    model = layers.Dense(64)(model)
    # Add Dropout layer to randomly set input units to 0 with rate 0.35 to prevent overfitting
    model = layers.Dropout(0.35)(model)

    # Define input layer for positions
    position_inputs = layers.Input(shape=(npos,))
    m2 = layers.Dense(64)(position_inputs)

    # Concatenate genotype matrix and position input layers
    model = layers.concatenate([model, m2])
    # Add Dense and output layer that outputs a (normalized) recombination value
    model = layers.Dense(16)(model)
    # Add Dropout layer to randomly set input units to 0 with rate 0.35 to prevent overfitting
    model = layers.Dropout(0.35)(model)
    output = layers.Dense(1)(model)

    model = Model(inputs=[genotype_inputs, position_inputs], outputs=[output])
    model.compile(optimizer="Adam", loss="mse")
    model.summary()

    return model
