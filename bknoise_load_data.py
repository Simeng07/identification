# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 11:36:18 2015

@author: tarrega
"""
import os
import pickle
import gzip
from sklearn.utils import shuffle

def load_bknoise_data(dataset = 'bkclass0403.pkl.gz'):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    print '... loading data'
    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    train_set = shuffle(train_set[0],train_set[1], random_state=0)
    f.close()
    numpoint = len(train_set[0])
    
    print 'number of train points:',numpoint
    return train_set,valid_set,test_set

if __name__ == '__main__':
    ts, vs, tes = load_bknoise_data()
    print ts[0].shape, ts[1].shape
    
    
    