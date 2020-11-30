from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Flatten, Dense, Dropout, Activation, Reshape, Lambda, LSTM, Conv2D, MaxPooling2D
from keras.layers.wrappers import Bidirectional
from keras.utils.io_utils import HDF5Matrix
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.constraints import maxnorm
from keras.optimizers import RMSprop
import tensorflow as tf

import numpy as np
import h5py
import random
from sys import argv
from keras import backend as K
from keras.callbacks import TensorBoard
from time import gmtime, strftime, time

tensorboard = TensorBoard(log_dir="logs/{}".format(time()), embeddings_freq=1, embeddings_metadata={'embed':'embed_meta'})

input_dataset = argv[1]
train_list = argv[2]
test_list = argv[3]

test_list_file = open(test_list)

# for a single-input model with 2 classes (binary):
max_size = 1500
alignment_max_depth = int(argv[4])

nb_epoch = 10
batch = int(argv[6])
window = int(argv[5])

#INPUTS timesteps are columns from the MSA
inputs = Input(shape=(window * alignment_max_depth,))
print('Shape of Input: ' + str(inputs.get_shape()))

dropfrac = float(argv[7]) #0.5
drop = Dropout(dropfrac)

#EMBEDDING
embed_size = int(argv[8]) #three channels like a RGB figure
embed = Embedding(26, embed_size, input_length=window*alignment_max_depth, batch_input_shape=(batch, window*alignment_max_depth), name='embed')

embedded = embed(inputs)

reshape = Reshape((window, alignment_max_depth, embed_size), input_shape=(window*alignment_max_depth, embed_size)) #x*y*channels data_format='channels_last' is the default
embedded2 = reshape(embedded)

print('Shape of embedded (reshaped) layer: ' + str(embedded2.get_shape()))
n_filters = int(argv[9]) #embed_size
conv_width = int(argv[10])
conv_depth = int(argv[11])
pool_width= int(argv[12])
pool_depth= int(argv[13])

after_pool=int(alignment_max_depth/pool_depth)
convolution = Conv2D(n_filters, (conv_width, conv_depth), activation='relu', padding='same') #output shape is window*alignment_max_depth*1
pool = MaxPooling2D(pool_size=(pool_width, pool_depth)) #output shape is window*(alignment_max_depth/10)*1
reshape2 = Reshape((int(window/pool_width), after_pool*n_filters), input_shape=(window, after_pool, n_filters))
after_convolution = convolution(embedded2)
print('Shape of prepool layer: ' + str(after_convolution.get_shape()))
after_convolution = pool(after_convolution)
print('Shape of postpool layer: ' + str(after_convolution.get_shape()))
after_convolution = reshape2(after_convolution)
print('Shape of convoluted layer: ' + str(after_convolution.get_shape()))

bidir_size = after_pool*n_filters
bidir_size2 = bidir_size*2

dense_size1 = int(argv[14])
dense_size2 = int(argv[15])
dense_size_sseq = 3

#LSTM
activation = Activation('softmax')
activation2 = Activation('softmax')

bidir = Bidirectional(LSTM(bidir_size, return_sequences=True), merge_mode='ave')
bidir2 = Bidirectional(LSTM(bidir_size, return_sequences=True), merge_mode='ave')
bidir3 = Bidirectional(LSTM(bidir_size, return_sequences=True), merge_mode='ave')

dense1_sseq = Dense(dense_size1)
dense2_sseq = Dense(dense_size2)


dense_sseq = Dense(dense_size_sseq)

flatten = Flatten()
bidir_output = bidir(after_convolution)
bidir_output = drop(bidir_output)
bidir_output = bidir2(bidir_output)
bidir_output = drop(bidir_output)
bidir_output = bidir3(bidir_output)
bidir_output = drop(bidir_output)

print('Bidir output shape: ' + str(bidir_output.get_shape()))
dense_output1_sseq = dense1_sseq(flatten(bidir_output))
dense_output1_sseq = drop(dense_output1_sseq)


print('Dense1 output shape: ' + str(dense_output1_sseq.get_shape()))

dense_output2_sseq = dense2_sseq(dense_output1_sseq)
dense_output2_sseq = drop(dense_output2_sseq)


print('Dense2 output shape: ' + str(dense_output2_sseq.get_shape()))

dense_output_sseq = dense_sseq(dense_output2_sseq)

predictions_sseq = activation(dense_output_sseq)

print('Prediction shape: ' + str(predictions_sseq.get_shape())) # + ' ' + str(predictions_rsa.get_shape()))

#MODEL
model_sseq = Model(input=inputs, output=predictions_sseq)

print('Compiling the model...')
model_sseq.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

np.set_printoptions(threshold=np.nan)


def generate_inputs(list):

    while 1:
        train_list_file = open(list)
        train_list_data = train_list_file.readlines()
        train_list_file.close()
        #random.shuffle(train_list_data)

        for target in train_list_data:
            print('\nTarget name: ' + target)
            # create numpy arrays of input data
            # and labels, from each line in the file
            X_batch = np.asarray(HDF5Matrix(input_dataset, 'inputs/' + target))  # length x max_depth

            length, max_depth = X_batch.shape[0], X_batch.shape[1]
                                            # 0...00length00...0, max_depth
            X_batch = np.lib.pad(X_batch, [(int(window / 2), int(window / 2)), (0, 0)], 'constant', constant_values=(0, 0))

            X_batch_windows = []

            for i in range(length):
                X_batch_windows.append(X_batch[i:i + window, :alignment_max_depth].reshape(window * alignment_max_depth))

            # length x 1 (sparse, 3 class)
            labels_batch_sseq = np.squeeze(np.asarray(HDF5Matrix(input_dataset, 'labels_sseq/' + target)))

            yield (np.array(X_batch_windows), labels_batch_sseq)



#epochs
for e in range(nb_epoch):
    print('Fit, epoch ' + str(e) + ":")
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    # batches be crazy
    #model_sseq.fit(np.array(X_batch_windows), labels_batch_sseq, nb_epoch=1, shuffle='batch', batch_size=batch, callbacks=[tensorboard])
    model_sseq.fit_generator(generate_inputs(train_list), samples_per_epoch=8891, nb_epoch=1, callbacks=[tensorboard])

    #random.shuffle(train_list_data)
    savefile='../models/epoch_{}_window_{}_depth_{}_convol_pool_{}-{}_bidir_{}_{}_dense_{}_{}_drop{}'.format(e, window, alignment_max_depth, pool_width, pool_depth, bidir_size, bidir_size2, dense_size1, dense_size2, dropfrac)
    model_sseq.save_weights(savefile + '_sseq')

    count = 0
    q3_sseq = 0
    q3_res_sseq = 0

    count_res = 0

    for target in test_list_file:

        print('Test name: ' + target)

        X_batch = np.asarray(HDF5Matrix(input_dataset, 'inputs/' + target)) #length x max_depth
        length, max_depth = X_batch.shape[0], X_batch.shape[1]

        X_batch = np.lib.pad(X_batch, [(int(window / 2), int(window / 2)), (0, 0)], 'constant', # 0...00length00...0, $
                         constant_values=(0, 0))

        X_batch_windows = []

        for i in range(length):
            X_batch_windows.append(X_batch[i:i+window, :alignment_max_depth].reshape(window*alignment_max_depth))

        labels_batch_sseq = np.squeeze(np.asarray(HDF5Matrix(input_dataset, 'labels_sseq/' + target)))

        res_sseq = model_sseq.evaluate(np.array(X_batch_windows), labels_batch_sseq, batch_size=batch)

        print("\nTest results sseq: " + str(res_sseq) + " " + str(int(res_sseq[1]*length)) + "/" + str(length))

        count+=1
        q3_sseq+=res_sseq[1]
        q3_res_sseq += int(res_sseq[1] * length)

        count_res+=length

    test_list_file.seek(0, 0)
    q3avg_sseq = q3_sseq/count
    q3avg_tot_sseq = q3_res_sseq/count_res

    print("Q3 average sseq: " + str(q3avg_sseq) + " " + str(q3avg_tot_sseq))

