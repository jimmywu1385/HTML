import numpy as np
from liblinear.liblinearutil import *

def transform_fullorder(x_data, output):
    for i in range(len(x_data)):
        output.append([])  
        data_len = len(x_data[i])
        output[i].append(1.0)

        for j in range(data_len):
                output[i].append(x_data[i][j])

        for j in range(data_len):
            for k in range(data_len)[j:]:
                output[i].append(x_data[i][j]*x_data[i][k])

        for j in range(data_len):
            for k in range(data_len)[j:]:
                for l in range(data_len)[k:]:
                    output[i].append(x_data[i][j]*x_data[i][k]*x_data[i][l]) 

def main():
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    train_size = 0
    test_size = 0

    with open('hw4_train.dat.txt', 'r') as f:
        while True:
            line = f.readline()
            if line == '\n' or len(line) == 0:
                break
            x = [float(i) for i in line.split()[:-1]]     
            train_x.append(x)
            train_y.append(float(line.split()[-1]))
            train_size+=1

    with open('hw4_test.dat.txt', 'r') as f:
        while True:
            line = f.readline()
            if line == '\n' or len(line) == 0:
                break
            x = [float(i) for i in line.split()[:-1]]     
            test_x.append(x)
            test_y.append(float(line.split()[-1]))
            test_size+=1    
    
    train_X = []
    train_Y = train_y.copy()
    test_X = []
    test_Y = test_y.copy()

    transform_fullorder(train_x, train_X)
    cv_err =0.0
    for i in range(5):      
        prob = problem(train_Y[:i*40]+train_Y[(i+1)*40:], train_X[:i*40]+train_X[(i+1)*40:])
        param = parameter('-s 0 -c 0.01 -e 0.000001 -q')
        m = train(prob, param)
        p_label, p_acc, p_val = predict(train_Y[i*40:(i+1)*40], train_X[i*40:(i+1)*40], m, '')
        cv_err+=p_acc[0]
    print(1-cv_err/500.0)

    transform_fullorder(test_x, test_X)         
    p_label, p_acc, p_val = predict(test_Y, test_X, m, '-q')
       
if __name__ == '__main__':
    main()
