import math
import scipy
import random
import numpy as np
from libsvm.svmutil import *

train_y, train_x = svm_read_problem('./train.scale')
test_y, test_x = svm_read_problem('./test.scale')

train_class = []
for i in range(len(train_y)):
    if train_y[i] not in train_class:
        train_class.append(train_y[i])

'''  problem 11
for i in range(len(train_y)):
    if train_y[i] != 5.0:
        train_y[i] = 0.0
prob  = svm_problem(train_y, train_x, isKernel=True)
param = svm_parameter('-c 10 -t 0')
m = svm_train(prob, param)
coef = m.get_sv_coef()
sv = m.get_SV()
w = dict()
for i in range(len(coef)):
    for j in sv[i]:
        sv[i][j] *= coef[i][0]
        if j not in w:
            w[j] = sv[i][j]
        else:
            w[j] += sv[i][j]
sum = 0.0
for i in w.values():
    sum += i*i
print(math.sqrt(sum))
'''


'''problem 12 13
for i in range(len(train_y)):
    if train_y[i] != 6.0:
        train_y[i] = 0.0
prob  = svm_problem(train_y, train_x, isKernel=True)
param = svm_parameter('-c 10 -t 1 -d 3 -r 1 -g 1')
m = svm_train(prob, param)
p_label, p_acc, p_val = svm_predict(train_y, train_x, m)
print(p_acc)
'''

'''problem 14 15
for i in range(len(train_y)):
    if train_y[i] != 1.0:
        train_y[i] = 0.0
for i in range(len(test_y)):
    if test_y[i] != 1.0:
        test_y[i] = 0.0
prob  = svm_problem(train_y, train_x, isKernel=True)
param = svm_parameter('-c 0.1 -t 2 -g 1')
m = svm_train(prob, param)
p_label, p_acc, p_val = svm_predict(test_y, test_x, m)
print(p_acc)
'''
#problem 16
par = [0.1, 1, 10, 100, 1000]
score = {0.1:0, 1:0, 10:0, 100:0, 1000:0}
for i in range(len(train_y)):
    if train_y[i] != 1.0:
        train_y[i] = 0.0
for t in range(1000):
    val_X = []
    val_Y = []
    train_X = []
    train_Y = []
    random_ind = random.sample(list(range(len(train_x))), k=200)
    max = 0
    max_ind = 0
    for i in range(len(train_x)):
        if i in random_ind:
            val_X.append(train_x[i])
            val_Y.append(train_y[i])
        else:
            train_X.append(train_x[i])
            train_Y.append(train_y[i])
        
    for j in par:
        prob  = svm_problem(train_Y, train_X, isKernel=True)
        param = svm_parameter('-c 0.1 -t 2 -g '+str(j))
        m = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(val_Y, val_X, m)
        if p_acc[0] > max:
            max = p_acc[0]
            max_ind = j  

    score[max_ind] += 1
print(score)