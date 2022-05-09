from util import *
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import *
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from train import *
import numpy as np


def test(batch_size):
    '''
    Function for loading model weights and
    aggregating accuracies for testing
    '''
    np.set_printoptions(threshold=sys.maxsize)
    model = TA_GRU()
    model.built = True
    model.load_weights('model_weights')
    data_manager = DataManager(batch_size, TESTING_INSTANCES)
    data_manager.load_dataframe_from_file('embedding_4.pkl')
    n_batch = data_manager.n_batches()
    data_manager.reshuffle_dataframe()
    batch_accuracies = []
    label_list = []
    y_list = []
    print(n_batch)
    for batch_index in range(n_batch):
        print("Batch index", batch_index)
        print("n_batch - 1", n_batch - 1)
        (x, y) = data_manager.next_batch()
        x = tf.Variable(x, dtype=tf.float32)
        y = tf.Variable(y, dtype=tf.float32)
        logits = model(x)
        scores, corrects = eval_batch(logits, y, batch_size)
        accuracy = 1 * corrects[0].shape[0] / batch_size
        batch_accuracies.append(accuracy)
        its = logits
        predition_its = tf.math.reduce_sum(tf.reshape(
            its, [batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT]), axis=1)
        y_predicts = tf.math.argmax(predition_its, axis=1)
        y_list.append(y_predicts)
        labels = tf.math.reduce_sum(tf.reshape(
            y, [batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT]), axis=1)
        labels = tf.math.argmax(labels, axis=1)
        label_list.append(labels)
        if (batch_index + 1) % 200 == 0:
            data_manager.set_current_cursor_in_dataframe_zero()
    label_list = np.array(label_list)
    label_list = np.ndarray.flatten(label_list)
    y_list = np.array(y_list)
    y_list = np.ndarray.flatten(y_list)
    cm = confusion_matrix(label_list, y_list)
    cm = cm.astype(
        'float') / cm.sum(axis=1)[:, np.newaxis]
    test_accuracy = tf.reduce_mean(batch_accuracies)
    figure = plt.figure(figsize=(3, 3))
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    print("FINAL ACCURACY", test_accuracy)

test(256)
