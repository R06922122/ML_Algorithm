# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:21:58 2018

@author: User
"""
#import time
import numpy as np
np.random.seed(500)



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

def bootstrap_data(data):
    ans = []
    Number = np.random.randint(0,400,400)
    for number in Number:
        ans.append(data[number])
    return np.array(ans)

#def guassian(xm, xn):
#    ans = np.dot(xm, xn)
#    return ans

def kMatrix_bulid( data):
        X = data[:,0]
        k = np.array([i for i in X])
        K = np.dot(k.T, k)
        return K
        
def LRR( lamda, train):

    Y = np.array(train[:,1]).reshape(-1,1)
    X = np.array([i for i in train[:,0]])
    K = kMatrix_bulid(train)
    beta = np.dot(np.linalg.inv(lamda*np.eye(11) + K), X.T)
    w = np.dot(beta, Y)
    return w
    
    
#start = time.time()
    
print("requires about 15 seconds")

each_voteFinal = []
train = np.array(Data[:400])
test = np.array(Data[400:])
Lamda = [0.01, 0.1, 1, 10, 100, 100000]

print('Data preparing\n')

for lamda in Lamda:
    tmp = []
    for i in range(250):
        train_data = bootstrap_data(train)
        tmp.append(LRR(lamda, train_data))
    each_voteFinal.append(tmp)
    
    
print("Trial Begins")
    
    
    

#%%
E_in = []
E_out= []
#print("E_in predict")
for i, country in enumerate(each_voteFinal):
    missentence = 0
#    print("country{} voting".format(i+1))
    for i in range(len(train)):
#        if i%90 ==0:    
#            print("{}%\tbe\tjugied".format(i/4+10))
        x, y = train[i]
        peopleConsensus = 0
        for person in country:
#                if i%50 ==0:    
#                    print("number {} voting".format(i))
            peopleConsensus += np.sign(np.dot(x, person))
        if np.sign(peopleConsensus) != y:
            missentence += 1
    E_in.append(missentence/400)
      
#print("\n")
#print("E_out predict")
for i, country in enumerate(each_voteFinal):
   missentence = 0
#   print("country{} voting".format(i+1))
   for i in range(len(test)):
#       if i % 33 ==0:
#           print("{}%\tbe\tjugied".format(i+1))
       x, y = test[i]
       peopleConsensus = 0
       for person in country:
           peopleConsensus += np.sign(np.dot(x, person))
       if np.sign(peopleConsensus) != y:
           missentence += 1
   E_out.append(missentence/len(test))
    


def answer():
    print("Question 15、16")
    print('------------------------')
    print("|Lamda\t|E_in\t|E_out|")
    for i in range(len(Lamda)):
        print("|{}\t|{}\t|{}|".format(Lamda[i], E_in[i], E_out[i]))
    print('------------------------')

    
answer()    

#end = time.time()
#print(end - start)