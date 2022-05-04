import logging
#from settings import CLASS_LEVEL1

import os,sys
import pandas
import numpy
from tqdm import tqdm
import tensorflow as tf
import dask.dataframe as dd
import pickle

def pickle_dataframes():
    if not os.path.exists("embedding_8.pkl"):
        df = pandas.concat([chunk for chunk in tqdm(pandas.read_csv('../BrexitDataset/BrexitTweets.snstrain_08.wordembbed.csv', chunksize=1, dtype = numpy.float32, header = None, encoding = 'utf-8',  sep = '\s+' , engine = 'c'), desc='Loading dataframe')])
        df.to_pickle("embedding_8.pkl")
    if not os.path.exists("embedding_7.pkl"):
        df = pandas.concat([chunk for chunk in tqdm(pandas.read_csv('../BrexitDataset/BrexitTweets.snstrain_07.wordembbed.csv', chunksize=1, dtype = numpy.float32, header = None, encoding = 'utf-8',  sep = '\s+' , engine = 'c', nrows=10), desc='Loading dataframe')])
        df.to_pickle("embedding_7.pkl")

pickle_dataframes()