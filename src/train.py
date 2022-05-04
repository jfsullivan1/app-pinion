import math
import numpy as np
import os
import random
from util import *
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import *

#TODO: import tensorflow equivalent of torchnet meter

def train(batch_size):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_memory_growth(gpus[1], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    model = TA_GRU()
    data_manager = DataManager(batch_size, TRAINING_INSTANCES)
    data_manager.load_dataframe_from_file( TRAIN_SET_PATH )
    optimizer = Adam(learning_rate=0.0005)
    criterion = tf.keras.losses.KLDivergence()
    n_batch = data_manager.n_batches()
    res=[]
    epochs = 100
    
    
    for epoch in range(epochs):
        data_manager.reshuffle_dataframe()
        n_batch = data_manager.n_batches()
        
        print(n_batch)
        for batch_index in range(n_batch):
            print("Batch index", batch_index)
            print("n_batch - 1", n_batch - 1)
            ( x , y ) = data_manager.next_batch()
            x = tf.Variable(x, dtype=tf.float32)
            y = tf.Variable(y, dtype=tf.float32)
           
            with tf.GradientTape() as tape:
                logits = model(x)
                scores, corrects = eval_batch( logits, x, y, criterion, batch_size)
                loss = criterion( logits , y )
            
            loss = tf.cast(loss, tf.float32)
            gradient = tape.gradient(loss, model.trainable_variables)
            
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            # loss.backward()
            # optimizer.step()
            tf.cast(corrects, tf.float32)
            accuracy = 1 * corrects[0].shape[0] / batch_size
            print("EPOCH %d\tBATCH%d\tACCURACY:%f\tLOSS:%f" % (epoch, batch_index, accuracy, loss))
            if ( batch_index + 1 ) % 200 == 0:
                
                data_manager.set_current_cursor_in_dataframe_zero()
            

def eval_batch(logits,x,y,criterion, batch_size):
    '''
    evaluate the logits of each instance, loss, corrects in a batch
    '''
    # if on_cuda:
    #     x , y = x.cuda() , y.cuda()
    # else:
    #     x , y = x.cpu() , y.cpu()

     # batch_size * dim
    # logits are the probabilities of the logsoftmax, should be converted to probabilities of softmax

    # since the size_average parameter==False, the loss is the sumed loss of the batch. The loss is a value rather than a vector

     # CrossEntropyLoss takes in a vector and a class num ( usually a index num of the vector )
    
    # print("Logits:", logits)
    # print("Labels:", y)

    its = tf.exp( logits )
    predition_its = tf.math.reduce_sum( tf.reshape(its, [batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT]), axis = 1)
    model_training_predicts = tf.math.argmax(predition_its, axis = 1)
    # model_training_predicts = torch.max( predition_its, 1)[ 1 ]
    # model_training_predicts = torch.max( logits , 1)[ 1 ] # [0] : max value of dim 1 [1]: max index of dim 1 LongTensor
    
    # model_training_predicts = torch.fmod( model_training_predicts, SENTIMENT_COUNT )
    # prediction_y : probabilitie of [ 0 1 2]
    # predicts: index of prediction
    # predition_y = torch.sum( y.view( batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT ), dim = 1, keepdim = False )
    predition_y = tf.math.reduce_sum( tf.reshape(y, [batch_size, TOPIC_SENTIMENT_COUNT, SENTIMENT_COUNT]), axis = 1 )
    # y_predicts = torch.max( predition_y, 1)[ 1 ] # the index instead of the value
    y_predicts = tf.math.argmax(predition_y, axis = 1)
    
    print("Model Train P",model_training_predicts)
    print("Y Pred", y_predicts)
    print("Equals pred", model_training_predicts == y_predicts)
    corrects = np.nonzero( model_training_predicts == y_predicts )
    print("Corrects", corrects)
    
    return its , corrects

def eval(model, data_manager, criterion):
    '''
    evaluate the accuracy of all epochs
    currently unused, a good example
    '''
    model.eval()#Sets the module in evaluation mode. refer to the pytorch nn manual
    #confusion_matrix=meter.ConfusionMeter(CLASS_COUNT)
    # NOTE: meter is a pytorch module
    confusion_matrix = tf.keras.metrics.ConfusionMatrix(SENTIMENT_COUNT)
    #loss_meter=meter.AverageValueMeter()
    loss_meter = tf.keras.metrics.Mean()
    for i,(x,y) in enumerate(data_manager.next_batch()):
        #x=Variable(torch.from_numpy(x).float(logger),volatile=True)
        x = tf.Variable(tf.convert_to_tensor(x), dtype=tf.float32)
        # y=Variable(torch.LongTensor(y),volatile=True)
        y = tf.Variable(y, dtype=tf.float32)
        loss,scores,corrects=eval_batch(model,x,y,criterion)
        loss_meter.add(loss.data[0])
        confusion_matrix.add(scores.data,y.data)
    acc=0
    cmvalue=confusion_matrix.value()
    for i in range(model.class_count):
        acc+=cmvalue[i][i]
    acc/=cmvalue.sum()
    model.train()
    return loss_meter.value()[0],acc