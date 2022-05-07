import math
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Found at https://stackoverflow.com/questions/58464790/is-there-an-equivalent-function-of-pytorch-named-index-select-in-tensorflow
def tf_index_select(input_, dim, indices):
    """
    input_(tensor): input tensor
    dim(int): dimension
    indices(list): selected indices list
    """
    shape = input_.get_shape().as_list()
    if dim == -1:
        dim = len(shape)-1
    shape[dim] = 1
    
    tmp = []
    for idx in tf.unstack(indices):
        begin = [0]*len(shape)
        begin[dim] = tf.cast(idx, tf.int32)
        tmp.append(tf.slice(input_, begin, shape))
    res = tf.concat(tmp, axis=dim)

    return res
 
class TA_GRU(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        # Hyperparameters
        
        self.word_embed_size = 300
        self.topic_embedding_size = 10
        self.fix_sentence_length = 24
        self.rnn_tweet_hidden_size = self.word_embed_size // 4
        self.attention_size = (self.topic_embedding_size + self.word_embed_size//4 ) // 64 * 64 + (self.topic_embedding_size + self.word_embed_size//4 ) % 64
        self.user_self_tweets = 3 # Number of previous tweets from a user
        self.neighbor_tweets = 5 # Number of previous tweets from a neighbor
        self.constant_tweet_weight_a = 1000.0 # Default is 1000.0
        self.rnn_hidden_size = (self.word_embed_size + self.topic_embedding_size) // 4
        self.class_count = (9 * 3) # topic_sentiment_count * sentiment_count
        

        #self.ids_only_wordembed_dim = torch.autograd.Variable( torch.LongTensor( [ i for i in range( 0 , self.fix_sentence_length * self.word_embed_size ) ] ) ).cuda()
        self.ids_only_wordembed_dim = tf.Variable([ i for i in range( 0 , self.fix_sentence_length * self.word_embed_size ) ], dtype=tf.float32)
        self.ids_only_topic_embedding = tf.Variable([ i for i in range( self.fix_sentence_length * self.word_embed_size, self.fix_sentence_length * self.word_embed_size + self.topic_embedding_size) ],  dtype=tf.float32)
        #self.ids_only_topic_embedding = torch.autograd.Variable( torch.LongTensor( [ i for i in range( self.fix_sentence_length * self.word_embed_size, self.fix_sentence_length * self.word_embed_size + self.topic_embed_size ) ] ) ).cuda()
        self.ids_seq_last = tf.Variable( [ self.user_self_tweets -1 ], dtype=tf.float32)
        self.rnn_tweet = tf.keras.layers.LSTM( self.rnn_tweet_hidden_size, input_shape = [self.word_embed_size], return_sequences=True)

        print(self.ids_only_wordembed_dim)
        print(self.ids_only_topic_embedding)
        print(self.ids_seq_last)
        
        
        self.linear_tweet = tf.keras.layers.Dense(self.word_embed_size, 
                                                  input_shape=[self.rnn_tweet_hidden_size])
                                                  
        self.A_alpha = tf.Variable(tf.random.normal([self.fix_sentence_length], stddev=0.1))
        self.B_alpha = tf.Variable(tf.random.normal([1], stddev=0.1))
        
        self.A_tweets = tf.Variable(tf.random.normal([(self.neighbor_tweets + 1)], stddev=0.1))
        self.B_tweets = tf.Variable(tf.random.normal([1], stddev=0.1))

        self.rnn = tf.keras.layers.GRU(self.rnn_hidden_size, input_shape=[(self.word_embed_size+self.topic_embedding_size)], return_sequences=True)
        self.linear = tf.keras.layers.Dense(self.class_count, input_shape=[self.rnn_hidden_size])
        # self.logsoftmax = torch.nn.LogSoftmax( dim = 1) # dim= 0 means sum( a[i][1][3]) = 1

    def call(self, inputs):
        '''
        from input to output
        '''
        ( batch_size , user_tweet_count , neighbor_tweet_count_add_one , twitter_length_size_x_word_embed_add_topic_embed_size ) = inputs.shape
         # var_only_wordembed_dim = param_input.index_select( 3 , self.ids_only_wordembed_dim ) #var only has word embed line
        var_only_wordembed_dim = tf_index_select(inputs, 3, self.ids_only_wordembed_dim) #var only has word embed line

        #var_only_wordembed_dim = var_only_wordembed_dim.view( batch_size, user_tweet_count, neighbor_tweet_count_add_one, self.fix_sentence_length, self.word_embed_size )
        #var_only_wordembed_dim = var_only_wordembed_dim.view( -1, self.fix_sentence_length, self.word_embed_size )
        var_only_wordembed_dim = tf.reshape(var_only_wordembed_dim, [batch_size, user_tweet_count, neighbor_tweet_count_add_one, self.fix_sentence_length, self.word_embed_size])
        var_only_wordembed_dim = tf.reshape(var_only_wordembed_dim, [-1, self.fix_sentence_length, self.word_embed_size])
        var_only_wordembed_dim_permuted = tf.transpose(var_only_wordembed_dim, perm=[1, 0, 2])
        #var_rnn_tweet_output, (var_rnn_tweet_output_h, var_rnn_tweet_output_c) = self.rnn_tweet( var_only_wordembed_dim_permuted)
        #var_rnn_tweet_output = var_rnn_tweet_output.permute(1, 0, 2)
        var_rnn_tweet_output = self.rnn_tweet( var_only_wordembed_dim_permuted)
        
        # var_twitter_embedded = torch.mean( var_rnn_tweet_output, dim=1 ) #default squeezed
        # var_twitter_embedded = self.linear_tweet( var_twitter_embedded )
        # var_twitter_embedded = var_twitter_embedded.view(batch_size, self.user_self_tweets, neighbor_tweet_count_add_one, self.word_embed_size )
        var_rnn_tweet_output = tf.transpose(var_rnn_tweet_output, perm=[1, 0, 2])
        var_twitter_embedded = tf.math.reduce_mean( var_rnn_tweet_output, axis=1 ) #default squeezed
        var_twitter_embedded = self.linear_tweet( var_twitter_embedded )

       

        # var_only_topic_embedding = param_input.index_select( 3, self.ids_only_topic_embedding )
        var_only_topic_embedding = tf_index_select(inputs, 3, self.ids_only_topic_embedding)
    
        # var_only_wordembed_dim = var_only_wordembed_dim.view( batch_size, user_tweet_count, neighbor_tweet_count_add_one, self.fix_sentence_length, self.word_embed_size )
        var_only_wordembed_dim = tf.reshape(var_only_wordembed_dim, [batch_size, user_tweet_count, neighbor_tweet_count_add_one, self.fix_sentence_length, self.word_embed_size])
        
        # var_only_wordembed_dim = var_only_wordembed_dim.view( -1, self.fix_sentence_length, self.word_embed_size )
        var_only_wordembed_dim = tf.reshape(var_only_wordembed_dim, [-1, self.fix_sentence_length, self.word_embed_size])

        # var_only_wordembed_dim_permuted = var_only_wordembed_dim.permute( 1, 0, 2 ) #transpose
        var_only_wordembed_dim_permuted = tf.transpose(var_only_wordembed_dim, perm=[1, 0, 2]) #transpose (permute in torch)
        
        # var_rnn_tweet_output, (var_rnn_tweet_output_h, var_rnn_tweet_output_c) = self.rnn_tweet( var_only_wordembed_dim_permuted)
        var_rnn_tweet_output = self.rnn_tweet( var_only_wordembed_dim_permuted)

        # var_rnn_tweet_output = var_rnn_tweet_output.permute(1, 0, 2)  
        var_rnn_tweet_output = tf.transpose(var_rnn_tweet_output, perm=[1, 0, 2]) #transpose in tensorflow
        
        # var_twitter_embedded = torch.mean( var_rnn_tweet_output, dim=1 ) #default squeezed
        var_twitter_embedded = tf.math.reduce_mean( var_rnn_tweet_output, axis=1 ) #default squeezed
        # Using original line
        var_twitter_embedded = self.linear_tweet( var_twitter_embedded )
        # var_twitter_embedded = var_twitter_embedded.view(batch_size, self.user_self_tweets, neighbor_tweet_count_add_one, self.word_embed_size )
        var_twitter_embedded = tf.reshape(var_twitter_embedded, [batch_size, self.user_self_tweets, neighbor_tweet_count_add_one, self.word_embed_size])

        # --- get twitter attention ---

        # Using original line
        var_twitter_embedded = var_twitter_embedded * self.constant_tweet_weight_a
        # var_twitter_and_topic_embedded = torch.cat( ( var_twitter_embedded , var_only_topic_embedding ) ,dim = 3 )
        var_twitter_and_topic_embedded = tf.concat( ( var_twitter_embedded , var_only_topic_embedding ), axis=3) 
        # var_twitter_and_topic_embedded = var_twitter_and_topic_embedded.mul( 1.0/ ( self.constant_tweet_weight_a + 1) ) 
        var_twitter_and_topic_embedded = var_twitter_and_topic_embedded * ( 1.0 / ( self.constant_tweet_weight_a + 1) )
        
        #==========The original for tweet attention

        #var_twitter_and_topic_embedded = var_twitter_and_topic_embedded.permute( 0, 1, 3, 2 )
        var_twitter_and_topic_embedded = tf.transpose(var_twitter_and_topic_embedded, perm=[0, 1, 3, 2])

        # var_user_tweet_context = torch.mv( var_twitter_and_topic_embedded.contiguous().view(-1 , self.neighbor_tweets + 1) , self.A_tweets) + self.B_tweets.expand( batch_size, user_tweet_count, self.word_embed_size + self.topic_embed_size ).contiguous().view(-1)
        var_user_tweet_context = tf.linalg.matvec(tf.reshape(var_twitter_and_topic_embedded, [-1, self.neighbor_tweets + 1]), self.A_tweets) + tf.reshape(tf.broadcast_to(self.B_tweets, [batch_size, user_tweet_count, self.word_embed_size + self.topic_embedding_size]), [-1])

        #==========The latest for tweet attention

        #var_user = var_user_tweet_context.view( batch_size, user_tweet_count, self.word_embed_size + self.topic_embed_size)
        var_user = tf.reshape(var_user_tweet_context, [batch_size, user_tweet_count, self.word_embed_size + self.topic_embedding_size])

        #==========The original for permute
        #var_user = var_user.permute( 1 , 0 , 2 )  # permute to (seq_len, batch, input_size)
        var_user = tf.transpose(var_user, perm=[1 , 0 , 2])
        
        # Using original line
        var_rnn_output = self.rnn( var_user ) # None means that h_0 = 0
        
        # var_rnn_output = var_rnn_output.permute( 1 , 0 , 2 )
        var_rnn_output = tf.transpose(var_rnn_output, perm=[1, 0, 2])

        #==========The latest for permute

        
        # var_seq_last = var_rnn_output.index_select( 1 , self.ids_seq_last )
        var_seq_last = tf_index_select(var_rnn_output, 1, self.ids_seq_last)
        
        # var_seq_last = var_seq_last.squeeze()
        var_seq_last = tf.squeeze(var_seq_last)
        
        # Using original line
        var_linear_output = self.linear(var_seq_last)

        # Using original line
        var_logsoftmax_output = tf.nn.softmax( var_linear_output, axis=1 )

        return var_logsoftmax_output
