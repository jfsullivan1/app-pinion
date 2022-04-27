import math
import numpy as np
import os
import random
from util import *
import tensorflow as tf
from model import *


def getData(batch_size):
    data_manager = DataManager(batch_size, TESTING_INSTANCES)
    data_manager.load_dataframe_from_file( TEST_SET_PATH )
    n_batch = data_manager.n_batches()
    res=[]
    batch_index = 0
    for batch_index in range(n_batch - 1):
        ( x , y ) = data_manager.next_batch()
        x = tf.Variable(x, requires_grad=False)
        logits = model.forward(x) # What do?
    data = np.loadtxt(file,delimiter = ",", dtype=str) #NEED TO FIGURE OUT NEW DELIMETER, HAVENT BEEN ABLE TO LOOK INSIDE CSV
    return data

