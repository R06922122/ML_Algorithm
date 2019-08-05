# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:53:40 2018

@author: Moras
"""
import numpy as np
#import time

Data = []
file = 'hw2_lssvm_all.dat.txt'
with open(file, 'r') as f:
    for line in f:
        data = line.split()
        x = np.array([float(v) for v in data[:-1]])
#        np.insert(x, 0, 1)
        y = int(data[-1])
        Data.append((x,y))
    Data = np.array(Data)

def guassian(gamma, xm, xn):
    x = xm - xn
    ans = np.exp(-gamma*np.dot(x, x))
    return ans

def kMatrix_bulid(gamma, data):
        nr = len(data)
        K = np.zeros((nr,nr))
        for i, xi in enumerate(data):
            for j, xj in enumerate(data[:i+1]):
                K[i, j] = guassian(gamma, xi, xj)
        for i in range(nr):
            for j in range(nr):
                K[i, j] = K[j, i]
        return K
    
def KRR(gamma, lamda, train, test):
    nr_train = len(train)
    nr_test = len(test)
    Y = np.array(train[:,1]).reshape(-1,1)
    K = kMatrix_bulid(gamma, train[:,0])    
    beta = np.dot(np.linalg.inv(lamda*np.eye(nr_train) + K), Y)
    
    E_in = 0
    for feature, label in train:
        kernel = [guassian(gamma, feature, v) for v in train[:, 0]]
        ans = np.sign(np.dot(beta.T, kernel))
        if ans != np.sign(label):
            E_in += 1
    
    E_out = 0
    for feature, label in test:
        kernel = [guassian(gamma, feature, v) for v in train[:, 0]]
        ans = np.sign(np.dot(beta.T, kernel))
        if ans != np.sign(label):
            E_out += 1
    return(E_in/nr_train, E_out/nr_test, K)
    
#start = time.time()

train = np.array(Data[:400])
test = np.array(Data[400:])
Gamma = [32, 2, 0.125]
Lamda = [0.001, 1, 1000]
print("Question 11、12")
print('-------------------------------')
print('|Gamma\t|Lamda\t|E_in\t|E_out|')
for gamma in Gamma:
    for lamda in Lamda:
        E_in , E_out, K = KRR(gamma, lamda, train, test)
        print('|{}\t|{}\t|{}\t|{}|'.format(gamma, lamda, E_in, E_out))
print('-------------------------------')
    
#end = time.time()
#
#print(end-start)
