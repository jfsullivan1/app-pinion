import math
import numpy as np
import os
import random
from util import *
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import *

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def train(batch_size):
    model = TA_GRU()
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0005, decay_rate=0.9, decay_steps=1000000)
    optimizer = Adam(learning_rate)
    criterion = tf.keras.losses.KLDivergence()
    res = []
    epochs = 100
    epoch_accuracies = []
    epoch_losses = []
    for epoch in range(epochs):
        data_manager = DataManager(batch_size, TRAINING_INSTANCES)
        data_manager.load_dataframe_from_file(TRAIN_SET_PATH)
        n_batch = data_manager.n_batches()
        data_manager.reshuffle_dataframe()
        batch_accuracies = []
        batch_losses = []
        print(n_batch)
        for batch_index in range(n_batch):
            print("Batch index", batch_index)
            print("n_batch - 1", n_batch - 1)
            (x, y) = data_manager.next_batch()
            x = tf.Variable(x, dtype=tf.float32)
            y = tf.Variable(y, dtype=tf.float32)

            with tf.GradientTape() as tape:
                logits = model(x)
                scores, corrects = eval_batch(
                    logits, y, batch_size)
                loss = criterion(logits, y)

            loss = tf.cast(loss, tf.float32)
            gradient = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            tf.cast(corrects, tf.float32)
            accuracy = 1 * corrects[0].shape[0] / batch_size
            batch_accuracies.append(accuracy)
            batch_losses.append(loss)
            print("EPOCH ACCURACIES", epoch_accuracies)
            print("EPOCH LOSSES", epoch_losses)
            print("EPOCH %d\tBATCH%d\tACCURACY:%f\tLOSS:%f" %
                  (epoch, batch_index, accuracy, loss))
            if (batch_index + 1) % 200 == 0:

                data_manager.set_current_cursor_in_dataframe_zero()
        epoch_accuracies.append(tf.reduce_mean(batch_accuracies))
        epoch_losses.append(tf.reduce_mean(batch_losses))
        file = open('average_accuracies.txt', 'a+')
        file.write("EPOCH: %d" % epoch)
        file.write(" -EPOCH ACCURACY: %f \n" % epoch_accuracies[epoch])
        file.close()
        file = open('losses.txt', 'a+')
        file.write("EPOCH: %d" % epoch)
        file.write(" -LOSSES AVG: %f \n" % epoch_losses[epoch])
        file.close()
    model.save_weights("model_weights")


def eval_batch(logits, y, batch_size):
    '''
    evaluate the logits of each instance, loss, corrects in a batch
    '''
    # logits are the probabilities of the logsoftmax, should be converted to probabilities of softmax
    # since the size_average parameter==False, the loss is the sumed loss of the batch. The loss is a value rather than a vector
    # CrossEntropyLoss takes in a vector and a class num ( usually a index num of the vector )
    its = tf.exp(logits)
    predition_its = tf.math.reduce_sum(tf.reshape(
        its, [batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT]), axis=1)
    model_training_predicts = tf.math.argmax(predition_its, axis=1)

    predition_y = tf.math.reduce_sum(tf.reshape(
        y, [batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT]), axis=1)
    y_predicts = tf.math.argmax(predition_y, axis=1)

    print("Model Train P", model_training_predicts)
    print("Y Pred", y_predicts)
    print("Equals pred", model_training_predicts == y_predicts)
    corrects = np.nonzero(model_training_predicts == y_predicts)
    print("Corrects", corrects)

    return its, corrects


def eval(model, data_manager, criterion):
    '''
    evaluate the accuracy of all epochs
    currently unused, a good example
    '''
    model.eval()  # Sets the module in evaluation mode. refer to the pytorch nn manual
    confusion_matrix = tf.keras.metrics.ConfusionMatrix(SENTIMENT_COUNT)
    loss_meter = tf.keras.metrics.Mean()
    for i, (x, y) in enumerate(data_manager.next_batch()):
        x = tf.Variable(tf.convert_to_tensor(x), dtype=tf.float32)
        y = tf.Variable(y, dtype=tf.float32)
        loss, scores, corrects = eval_batch(model, x, y, criterion)
        loss_meter.add(loss.data[0])
        confusion_matrix.add(scores.data, y.data)
    acc = 0
    cmvalue = confusion_matrix.value()
    for i in range(model.class_count):
        acc += cmvalue[i][i]
    acc /= cmvalue.sum()
    model.train()
    return loss_meter.value()[0], acc
