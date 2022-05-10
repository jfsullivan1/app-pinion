# -*- coding: utf8 -*-

import logging
#from settings import CLASS_LEVEL1

import os
import sys
import pandas
import numpy
from tqdm import tqdm
import tensorflow as tf
import pickle

from math import ceil

use_gpu = True
if not use_gpu:
    tf.config.set_visible_devices([], 'GPU')
else:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

ROOT_DIR = ''  # the root path, currently is the code execution path

SAVE_DIR = '%s/save' % ROOT_DIR

WORD_EMBEDDING_DIMENSION = 300  # must set before run
TOPIC_EMBEDDING_DIMENSION = 10
TWITTER_LENGTH = 24  # universal twitter length for each twitter, must set before run
USER_SELF_TWEETS = 3  # a user's previous tweet nums, must set before run
NEIGHBOR_TWEETS = 5  # neighbors' previous tweet nums, must set before run
TRAINING_INSTANCES = 9216
TESTING_INSTANCES = 9216

TRAIN_SET_PATH = 'embedding_new.pkl'
TEST_SET_PATH = 'embedding_4.pkl'

TOPIC_SENTIMENT_COUNT = 9
SENTIMENT_COUNT = 3

CHUNK_SIZE_MULTIPLIER_BATCH_SIZE = 2
# 1000 original, smaller better, more topical influence
CONST_TWEET_WEIGHT_A = 1000.0


def get_logger(name):
    '''
    set and return the logger modula,
    output: std
    '''
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    fh = logging.FileHandler(name)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def latest_save_num():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    files = os.listdir(SAVE_DIR)  # list all the files in save dir
    maxno = -1
    for f in files:
        if os.path.isdir(SAVE_DIR+'/'+f):
            try:
                no = int(f)
                maxno = max(maxno, no)
            except Exception as e:
                print(e.message, file=sys.stderr)
                pass
    return maxno


class DataManager():
    '''
    data loading module
    '''

    def __init__(self,
                 param_batch_size,
                 param_training_instances_size):
        self.__current_dataframe_of_pandas = None  # dataframe read by a chunk
        # the cursor shifted in pandas, it's the global cursor shift in dataframe
        self.__current_cursor_in_dataframe = 0
        self.__batch_size = param_batch_size
        self.__training_instances_size = param_training_instances_size

    def load_dataframe_from_file(self, param_filepath_in):
        '''
        read once, initialize dataframe_of_pandas
        '''
        # Dataframe loaded from Pickle file
        print('Loading dataframe...')
        self.__current_dataframe_of_pandas = pandas.read_pickle(TRAIN_SET_PATH)
        print(self.__current_dataframe_of_pandas.shape)
        print("Loaded dataframe.")

    def reshuffle_dataframe(self):
        pass

    def next_batch(self):
        '''
        obtain next batch
        possible out of range, you have to control the tail
        the best way is to call next_batch dataframe_size()//batch_size times
        then call tail_batch
        use reshape
        '''
        batch_size = self.__batch_size
        s = self.__current_cursor_in_dataframe  # start cursor
        t = s + batch_size  # end cursor
        print("S value: ", s)
        print("T value: ", t)

        batch_index = s // batch_size

        print("Loading batch: %d" % (batch_index))

        batch_x = numpy.zeros(
            (batch_size, USER_SELF_TWEETS, 1+NEIGHBOR_TWEETS,
             TWITTER_LENGTH * WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION))
        batch_y = numpy.zeros(
            (batch_size, TOPIC_SENTIMENT_COUNT * SENTIMENT_COUNT))
        for user_i in range(s, t):
            # each user's time serial
            label_shift = TOPIC_SENTIMENT_COUNT * SENTIMENT_COUNT  # one col for label

            batch_x[user_i % batch_size] = self.__current_dataframe_of_pandas.iloc[user_i][label_shift:].values.reshape(
                (USER_SELF_TWEETS, NEIGHBOR_TWEETS+1, TWITTER_LENGTH * WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION))
            # iloc is absolute shift, loc is access by row name. if header==None, then the row name is the int index,
            # which is inherented by splitted chunks
            batch_y[user_i %
                    batch_size] = self.__current_dataframe_of_pandas.iloc[user_i][0: label_shift]

        self.__current_cursor_in_dataframe = t
        return batch_x, batch_y

    def set_current_cursor_in_dataframe_zero(self):
        '''
        if INSTANCE % batch_size == 0, then the tail_batch won't be called, so call this function to reset the __cursor_in_current_frame
        '''
        self.__current_cursor_in_dataframe = 0

    def tail_batch(self):

        batch_size = self.__batch_size
        s = self.__current_cursor_in_dataframe
        t = s + batch_size
        batch_index = s // batch_size

        print("Loading batch: %d" % (batch_index))
        batch_x = numpy.zeros(
            (batch_size, USER_SELF_TWEETS, 1 + NEIGHBOR_TWEETS,
             TWITTER_LENGTH * WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION))
        batch_y = numpy.zeros(
            (batch_size, TOPIC_SENTIMENT_COUNT * SENTIMENT_COUNT))

        # complement the last chunk with the initial last chunk
        last_batch_size = len(self.__current_dataframe_of_pandas) % batch_size
        append_times = batch_size // last_batch_size
        append_tail = batch_size % last_batch_size

        for user_i in range(s, t):
            label_shift = TOPIC_SENTIMENT_COUNT * SENTIMENT_COUNT  # one col for label
            if (user_i % batch_size) < last_batch_size:
                batch_x[user_i % batch_size] = self.__current_dataframe_of_pandas.iloc[user_i][label_shift:].values.reshape(
                    (USER_SELF_TWEETS, 1+NEIGHBOR_TWEETS, TWITTER_LENGTH*WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION))
                batch_y[user_i %
                        batch_size] = self.__current_dataframe_of_pandas.iloc[user_i][0: label_shift]
            else:
                shift_in_last_batch_size = (
                    user_i % batch_size) % last_batch_size
                batch_x[user_i % batch_size] = self.__current_dataframe_of_pandas.iloc[user_i - user_i % batch_size + shift_in_last_batch_size][label_shift:].values.reshape(
                    (USER_SELF_TWEETS, 1+NEIGHBOR_TWEETS, TWITTER_LENGTH*WORD_EMBEDDING_DIMENSION + TOPIC_EMBEDDING_DIMENSION))
                batch_y[user_i % batch_size] = self.__current_dataframe_of_pandas.iloc[user_i -
                                                                                       user_i % batch_size + shift_in_last_batch_size][0: label_shift]

        self.__current_cursor_in_dataframe = 0
        return batch_x, batch_y

    def n_batches(self):
        return ceil(self.__training_instances_size / self.__batch_size)


if __name__ == '__main__':
    pass
