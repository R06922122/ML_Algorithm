# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:21:58 2018

@author: User
"""

import numpy as np
#import time

Data = []
file = 'hw2_lssvm_all.dat.txt'
with open(file, 'r') as f:
    for line in f:
        data = line.split()
        x = np.array([float(v) for v in data[:-1]])
        x = np.insert(x, 0, 1)
        y = int(data[-1])
        Data.append((x,y))
    Data = np.array(Data)

def guassian(xm, xn):
    ans = np.dot(xm, xn)
    return ans

def kMatrix_bulid( data):
        X = data[:,0]
        k = np.array([i for i in X])
        K = np.dot(k.T, k)
        return K
        
def LRR( lamda, train, test):
    nr_train = len(train)
    nr_test = len(test)
    Y = np.array(train[:,1]).reshape(-1,1)
    X = np.array([i for i in train[:,0]])
    K = kMatrix_bulid( train)
    beta = np.dot(np.linalg.inv(lamda*np.eye(11) + K), X.T)
    w = np.dot(beta, Y)

    
    E_in = 0
    for feature, label in train:
        ans = np.sign(np.dot(w.T, feature))
        if ans != np.sign(label):
            E_in += 1
    
    E_out = 0
    for feature, label in test:
        ans = np.sign(np.dot(w.T, feature))
        if ans != np.sign(label):
            E_out += 1
    return(E_in/nr_train, E_out/nr_test,w)
    
    
#start = time.time()
train = np.array(Data[:400])
test = np.array(Data[400:])
#Gamma = [32, 2, 0.125]
Lamda = [0.01, 0.1, 1, 10, 100]
print("Question 13、14")
print('-------------------------------')
print('|Lamda|\tE_in|\tE_out|')
for lamda in Lamda:
    E_in , E_out, beta = LRR(lamda, train, test)
    print('|{}\t|{}\t|{}\t|'.format(lamda, E_in, E_out))
print('-------------------------------')

#end = time.time()
#print(end-start)