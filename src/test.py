from util import *
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import *
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from train import *


def test(batch_size):
    '''
    Function for loading model weights and
    aggregating accuracies for testing
    '''
    model = TA_GRU()
    model.built = True
    model.load_weights('model_weights')
    data_manager = DataManager(batch_size, TESTING_INSTANCES)
    data_manager.load_dataframe_from_file('embedding_4.pkl')
    n_batch = data_manager.n_batches()
    data_manager.reshuffle_dataframe()
    batch_accuracies = []
    print(n_batch)
    for batch_index in range(n_batch):
        print("Batch index", batch_index)
        print("n_batch - 1", n_batch - 1)
        (x, y) = data_manager.next_batch()
        x = tf.Variable(x, dtype=tf.float32)
        y = tf.Variable(y, dtype=tf.float32)
        logits = model(x)
        scores, corrects = eval_batch(logits, y, batch_size)
        tf.cast(corrects, tf.float32)
        accuracy = 1 * corrects[0].shape[0] / batch_size
        print(accuracy)
        batch_accuracies.append(accuracy)
        if (batch_index + 1) % 200 == 0:
            data_manager.set_current_cursor_in_dataframe_zero()
    test_accuracy = tf.reduce_mean(batch_accuracies)
    print("FINAL ACCURACY", test_accuracy)


test(256)
