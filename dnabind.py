import random
import sys
import tensorflow as tf
import re
import numpy as np
from keras.models import Sequential
from keras import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Bidirectional, Concatenate, Dropout, Reshape
from keras.layers import Conv1D, Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalMaxPooling1D, ZeroPadding1D, MaxPooling1D
from keras.utils import pad_sequences
from tensorflow.keras import optimizers


"""
hyper parameters
"""
lines_read = 10000
filters_num = 32
kernel_size = 6
FC_layer = 16
droput_init = 0.2
dropout_increase = 0
time_dropout = 3
learn_rate = 0.05
mini_batch = 512
epoch_num = 1

"""
trasnform [ACGT]* string into one hot matrix
"""
def one_hot_encode(seq, mapping):
    seq2 = [np.array(mapping[i], dtype='float32') for i in seq]
    return np.array(seq2)

"""
preprocessing function: read the RBNS files, make label for each sequence based on concentration and occurrunces
"""
def find_matching_sequences_with_reverse_rank_labels(files, mapping, sequences_per_file=None, onehot=False):
    pattern_input = r'RBP([\d]+)_input'
    pattern_concentration = r'RBP([\d]+)_([\d]+)nM'
    labels_dic = {}
    # extract the concentration value from each file.
    for file_name in files:
        sequence_label = None
        if re.match(pattern_input, file_name):
            sequence_label = 0
        elif re.match(pattern_concentration, file_name):
            concentration_value = re.search(pattern_concentration, file_name).group(1)
            sequence_label = int(concentration_value)
        if sequence_label is not None:
            labels_dic[file_name] = sequence_label
    max_concentration = max(labels_dic.values())
    
    # multiply by -1 the concentrations, define the input as -32 * max_concentration
    for key, value in labels_dic.items():
        if value == 0:
            labels_dic[key] = -32 * max_concentration
        elif value == 1:
            labels_dic[key] = 0.01
        else:
            labels_dic[key] = -1 * value
    X_train = []
    Y_train = []
    sequence_label = 1
    for file_name in labels_dic.keys():
        input_indicator = False
        if re.match(pattern_input, file_name):
            input_indicator = True
        with open(file_name, 'r') as file:
            counter = 0
            y_adding = [labels_dic[file_name]]*sequences_per_file
            for line in file:
                lean = random.choice([True, False])
                if lean:
                    continue
                seq, occ = line.strip().split()
                occ = float(occ)
                if occ > 1 and not input_indicator:
                    y_adding[counter] = labels_dic[file_name] + occ*occ
                seq = seq + (40-len(seq))*'N'
                X_train.append(one_hot_encode(seq, mapping))
                counter = counter + 1
                if counter == sequences_per_file:
                    break
        Y_train.extend(y_adding)
    return X_train, Y_train

"""
read the RNAcompete sequences and transform them into one hot matrices
"""
def RNAcompete_read(rncmp):
    mapping = {"A": [1., 0., 0., 0.], "C": [0., 1., 0., 0.], "G": [0., 0., 1., 0.], "U": [0., 0., 0., 1.], "N": [.25, .25, .25, .25]}
    rna_test = []
    with open(rncmp,'r') as file:
        for line in file:
            rna_test.append(one_hot_encode(line.strip(),mapping))
    X_test = pad_sequences(rna_test, maxlen=40,value=0.25)
    return X_test

"""
Defining the loss function as negative pearson corrleatin
"""
def pearson_loss(y_true, y_pred):
    # Flatten the arrays if needed
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    # Center the data
    y_true -= tf.keras.backend.mean(y_true)
    y_pred -= tf.keras.backend.mean(y_pred)

    # Calculate the denominator
    denom = tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_true)) * tf.keras.backend.sum(tf.keras.backend.square(y_pred)))

    # Calculate the numerator
    num = tf.keras.backend.sum(y_true * y_pred)

    # Calculate the negative Pearson correlation
    correlation = -num / (denom + tf.keras.backend.epsilon())

    return correlation

"""
A callback class for increasing dropout rate for the fully connected layer
"""
class IncreaseDropoutCallback(tf.keras.callbacks.Callback):
    def __init__(self, start_epoch, increase_rate):
        super(IncreaseDropoutCallback, self).__init__()
        self.start_epoch = start_epoch
        self.increase_rate = increase_rate

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            for layer in self.model.layers:
                if isinstance(layer, Dropout):
                    new_dropout_rate = min(0.5, layer.rate + self.increase_rate)
                    layer.rate = new_dropout_rate
                    print(f"Epoch {epoch + 1}: Increasing dropout rate to {new_dropout_rate:.4f}")
                    break



"""
The neural net, with Input(40,4)->Conv1D(128,6)->globalMaxPooling->Fully Connected Layer (16) ->Output (1)
"""
def basicModel():
    filters = filters_num
    inputs = Input(shape=(40, 4))

    conv1 = Conv1D(filters*4, kernel_size, activation='relu')(inputs)
    conv1 = GlobalMaxPooling1D()(conv1)
    concat = Concatenate()([conv1])
    concat = Flatten()(concat)
    dense = Dense(FC_layer, activation='relu')(concat)
    dense = Dropout(droput_init)(dense)
    output = Dense(1, name='my_dropout')(dense)
    model_1 = Model(inputs, output)
    return model_1

def main():
    # read the argv files
    ofile = sys.argv[1]
    rncmp = sys.argv[2]
    input_file = sys.argv[3]
    rbns = sys.argv[4:len(sys.argv)]
    files = [input_file] + rbns
    
    # preprocessing to X_train, Y_train, X_test.
    mapping = {"A": [1., 0., 0., 0.], "C": [0., 1., 0., 0.], "G": [0., 0., 1., 0.], "T": [0., 0., 0., 1.], "N": [.25, .25, .25, .25]}
    X_train, Y_train = find_matching_sequences_with_reverse_rank_labels(files,mapping,sequences_per_file=lines_read ,onehot=True)
    X_train = pad_sequences(X_train, maxlen=40, value=0.25)
    Y_mapped = np.array(Y_train, dtype='float64')
    X_test = RNAcompete_read(rncmp)
    
    # set the model
    batch_size = mini_batch
    increase_dropout_callback = IncreaseDropoutCallback(start_epoch=time_dropout, increase_rate=dropout_increase)
    model = basicModel()
    opt = optimizers.Adam(learning_rate=learn_rate, epsilon=1e-6)
    model.compile(loss=pearson_loss, optimizer=opt)
    
    # training the model
    history = model.fit(
        X_train, Y_mapped,
        batch_size=batch_size,
        epochs=epoch_num,
        shuffle=True,
        callbacks=[increase_dropout_callback])
    # predict the results for X_test and save the results.
    answers = model.predict(X_test)
    np.savetxt(ofile, answers)
    return

if __name__ == "__main__":
   main()