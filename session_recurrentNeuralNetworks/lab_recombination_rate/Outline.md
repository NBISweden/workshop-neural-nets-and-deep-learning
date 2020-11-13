

## Session on Recurrent Neural Networks

### Tentative schedule

- 40-45min lecture Dan
- 60-90min exercise 1 maybe annotation
- 15-20min intro to next exercise ReLERNN Per
- 60-90min exercise 2 ReLERNN

(3-4h)

## Lecture
- Will be loosely based on MIT's sequence modelling lecture
  http://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L2.pdf
- Introduction to handling datasets of varying lengths
    - Fixed window
    - Bag-of-words representation
    - Recurrence
- RNN structure
    - Back-propagation through time
    - Weight sharing
    - Vanishing & exploding gradients
    - Gated cells such as LSTM as a solution (extra reading: GRU)
    - Compare to residual (aka skip) connections (He, 2015)
    - Bidirectionality, keras: `Bidirectional` class
    - Depth: Stacking RNNs on top of each other
- Output
    - seq2vec
        - keras: `LSTM(..., return_state=True)`
        - NLP: text classification, sentiment analysis
        - classification: ancient or non-ancient DNA?
    - encoder-decoder (extension of seq2vec)
        - text generation
        - text completion
        - machine translation
    - seq2seq
        - keras: `LSTM(..., return_squences=True)`
        - NLP: part-of-speech, named entity recognition (BIO encoding)
        - protein structure prediction: secondary structure
    - seq2matrix
        - NLP: dependency parsing
        - protein structure prediction: contact map
        - conjecture: keras-expressible only if using sequences of fixed maximum length
            - seems like this is false, Claudio has code to show 
- Handling sequences of varying lengths:
    - way 0: use batch size of one or all same length in one batch
    - way 1: padding
    - way 2: packing
    - masking
- Embeddings for categorical inputs
    - OOV: out of vocabulary token
    - Word
    - Char
    - BPE
    - k-mers
    - Can be pretrained using for example SGNS (Mikolov, 2013)    
    - Compare to to 1-hot encoding
- Note on time and batch dimension:
    - `[B, T, X]`: keras & tensorflow default
    - `[T, B, X]`: pytorch default
- Note on continuous input:
    - Time series prediction
    - Sound and speech
    - Trajectory modelling
    - Environmental modelling

## Exercise 1
Still under the development but the idea is to run training on some simple
languages and looking at outputs and intermediate states (neuron activations)
and generation. Loosely inspired by Karphathy's
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
The students can choose what training data they want to look at
from a set of choices.
If they want to be able to train on CPUs during the exercise they will need
to look at very simple sequences such as:
- synthetic languages like regular languages and CFGs
- memorizing: copying & reverse
- calculator
 
We can provide pretrained models that they can play with:
- natural language
- genomic data
- proteins

### Extra reading material
- Highway layers
- Attention
    - dot-product Luong
    - additive    Cho
- Compare to convolutions
    - 1D CNNs for sequence modelling
    - Transformers: repeated 1x1 convolution & (multi-head) self-attention
- Masked pretraining
- Expressivity: how expressive are LSTMs and GRUs?
    - Memorization capabilities
    - Location in Chomsky Hierarchy
- Recurrent dropout
    - ...and Zoneout
- Lookahead instead of Bidirectionality?

## Intro to exercise 2: recombination rate estimation using RNNs

- Recombination - what is it
- Recombination - why it's important
    - cf Booker 2020: rate correlated with magnitude of e.g. F_ST
    - determine haplotype blocks: crucial to detect signals of selection (cf tsinfer, Relate)
- Recombination - how to estimate (Penalba & Wolf 2020)
    - pedigree-based
    - gamete-based
    - population-based - only viable alternative for natural populations
- Review of current methods (LDHelmet, iSMC, ...)
- deep learning methods: CNN and ReLERNN
- short introduction to msprime and RELERNN


## Exercise 2 ReLERNN

The ReLERNN package contains scripts to 

1. simulate data using a known recombination map
2. train model
3. test model
4. make predictions and bootstrap

The program comes with a small dataset that is small enough to process in less than an hour.
However, the scripts hide the network structure from the user, and therefore 
it is our aim that the students write up the model them themselves, using a coding template.
We will provide simulated data from step 1 above, concentrating the exercise on steps 2-4.

### Outline

Test different scenarios building up to more and more realistic settings

#### Scenario 1: simulated data with flat recombination rate

msprime

#### Scenario 2: simulated data with recombination hotspot

msprime

#### Scenario 3: simulated data with recombination map

stdpopsim; may need to provide training/test/validation data

#### Scenario 4: run ReLERNN on example data from package

ReLERNN provides small test data set that runs in 10 minutes
