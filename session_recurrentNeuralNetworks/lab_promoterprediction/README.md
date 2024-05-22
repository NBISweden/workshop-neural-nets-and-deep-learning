# Lab session on promoter prediction using recurrent networks

In this lab we look at recurrent neural networks can be used for sequence classification. The data we'll be using are known 
promoter regions from humans and mice, and the task is to  recognize these compared to some randomly currupted versions of them. The lab takes inspiration from [DeePromoter](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2019.00286/full), but uses a less sophisticated architecture to make it easier to understand. The data was found for the re-implementation of DeePromoter [here](https://github.com/egochao/DeePromoter).

As a bonus, this lab shows how we can use language modelling to pretrain an LSTM, and then transfer the learned recurrent layer to the sequence classification task.


The lab is mainly intended to be run on Google Colab, and you can open it by following [this link](https://colab.research.google.com/github/NBISweden/workshop-neural-nets-and-deep-learning/blob/master/session_recurrentNeuralNetworks/lab_promoterprediction/promoter_prediction.ipynb)


## Intended learning outcomes

- build a keras model based on LSTM architecture for sequence prediction
- Understand how to go from text data to encoded tokenized data suitable for feeding into a recurrent network
- Gain insight into the operation of recurrent networks by visualizing and inspecting the recurrent state
- Understand how language modelling can be used as a pretraining task for a recurrent network, and transfer it to a classification task.

