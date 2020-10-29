# session_annBuildingBlocks

## This session focus on:

0. (intro: motivation, neurons, perceptrons, learning)
    - classification/regression
1. The basic building blocks of a sigmoid
feed-forward neural net,
    - sigmoid activation function
    - the mean square error loss function
    - gradient descent optimization
    - back-propagation
2. Performance measures for
    - cost-plots
    -
    - ...?
3. Selected additional activation functions and their
main properties (pros/cons):
    - reLu
    - tanh
    - SoftMax
    - ...?
    - Also cover linear and and step (perceptron) elaborate little
    on why they don't work
4. Selected additional loss (cost) functions
    - Cross entropy
    - (log likelihood?)
    - ...?
5. Selected additional optimizations
    - (ADAM?)
    - ...?

**_Note!_** The exact "selections" above will be aligned with the needs
of following sessions

### Thoughts/Questions:
1. Are the following best covered here or in sessions on convolutional
and recurrent anns, respectively?
  - LSTMs
  - Convolutions
2. Anything vital missing?


First sketch of more detailed timeline/schedule for session

- Day 1
  - Motivation intro?
    - history?
    - examples?
    - biological neurons here rather than below?
    - relations
      - AI
      - ML
      - ANN/DL
    - ...
  - ANN BB !
    - Biological neurons
      - Bengt's or claudio's fig
    - First artificial neuron model: perceptron
      - linear model
        - weights
        - bias
      - step activation
      - "Not good for training -- more later"
      - New figs
    - sigmoid neuron
      - Logistic model (without residual variation model)
      - new figs
    - FFANN
      - Layers
        - input
        - hidden
          - "deep"
        - output
      - edges
        - all-to-all between sequential layers
      - new figs
      - "connect log model can do more than just Logistic regression"
        - examples...
        - [playground](http://playground.tensorflow.org/)
    - Learning
      - regression closed form not working
      - Cross validation
      - Loss/cost/etc function MSE
        - residuals, SS
        - MSE cross-validation
      - new figs
    - Gradient descent
      - "Clever hill-climbing"
        - derivative/partial derivative/gradient visually shows which way down is"
          - (no detailed math at this point)
          - learning rate
        - (Bengt's + Claudio's figs)
        - algorithm (Claudio's)
    - Backprop
      - chain rule
        - example: sigmoid function ?
      - use chain rule to split by z (net) and a (out)
        - dJ/dw = dJ/da * da/dz * dz/dw
      - Using Paulo's pedagogical approach! (slight variant)
        - 2 (hmm, or maybe  1, and add another layer later) hidden layer,
        1 neuron per layer, given weights and biases
          - forward pass. (here or earlier?)
          - backward pass
            - by layer
        - add one neuron to last hidden layer (or to both hidden layers) to existing network
          - do new neuron in last hidden layer
          - first hidden layer
            - dJ/da
              - show weighted summation over neurons in last hidden layer
            - do da/dz and dz/dw as well
          - Take-home:
            - Need to do all partial derivatives in one layer before doing
            the one in the preceeding layer
            - Can use vector mulktiplication
              - collect partial derivatives for last layer in column vector G = Gradient
              - collect incoming weights to one neuron in preceeding layer in row vector w
              - do vector multiplication w x G and show that it produces the sum!
              - DP?
            - *Softwares* use matrix multiplication (hand-waving)
              - collect weights vectors in matrix W (as rows)
              - multiply W x G gives dJ/da for whole layer!
              - (also forward pass can be done with matrix algebra)
              - so information is passed through network layers as vectors or
              matrices
              - additionally, all datasets can be fed to the network in one go
                - adds dimension -> tensors
                - *tensors flows through the network.*
  - Lab
    - Keras+tensorflow  basics??
- Day 2
  - ANN BB 2
      - Evaluating learning
        - cost plots
          - slow learning...
        - ...
        - how improve?
      - Alternative activation functions
        - naive
          - step
            - not meaningfully differentiable
            - ...
          - linear
            - not differentiable (in menaing ful way)
            - collapses into single linear model
        - What's a "good" activation function?
          - Non-linear
          - Allows back-propagation
          - Good classifier
          - Avoids vanishing gradient problem
          - Avoids exploding activation problem
          - zero-centered output
          - Computationally efficient
          - Fast convergence
        - sigmoid-ish
          - tanh
          - SoftMax
          - ...
          - evaluate each (e.g., vanishing gradient)
        - ReLu-ish
          - ReLu
          - Variants of ReLU (Swish?)
      - Alternative loss function variants
        - Cross entropy
        - ... (max likelihood)
      - Alternative Optimmization algorithm
        - Stoch gradsient descent
        - ... (ADAM?)
      - Other stuff
        - Learning rate
        - batches
        - Over/underfitting
          - regularization
        - (moments?)
        - kernels
  - Lab playground + keras play with other BB??
