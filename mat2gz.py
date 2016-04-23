# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 15:15:31 2014

@author: huang

translate from matlab data to python. 
save to .gz file. 

"""
import cPickle as pickle
from scipy.io import loadmat
import gzip
import numpy as np
 
print 'load train.mat'
x = loadmat('train.mat')
trainx = x['x'].T
ty = x['y']
ny = np.size(ty)
trainy = np.arange(ny, dtype=np.int64)
for i in xrange(ny):
    trainy[i]=ty[i]-1
print 'train shape:',trainx.shape,trainy.shape
print 'what is y?',type(trainy),trainy.shape,trainy.dtype

print 'load test.mat'
trainset = (trainx,trainy)
x = loadmat('test.mat')
testx = x['x'].T
ty = x['y']
ny = np.size(ty)
testy = np.arange(ny, dtype=np.int64)
for i in xrange(ny):
    testy[i]=ty[i]-1
print 'test shape:',testx.shape,testy.shape
testset = (testx,testy)
validset = testset

f = gzip.open('bkclass0403.pkl.gz', 'wb')        
pickle.dump([trainset, validset , testset], f)
f.close()

print 'finished'

def main():
    pass    
        
if __name__ == '__main__':  
    #testdb()
    main()
    #a = os.getcwd()
    #print go_up(a,1)
    