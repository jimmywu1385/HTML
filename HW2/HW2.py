import numpy as np
import random
import math
def train_generate(x_data, y_data):
    for i in range(200):
        y = random.choice([-1, 1])
        if y == 1:                                      # mean = [2, 3]  cov = [[0.6, 0], [0, 0.6]]
            x1 = random.normalvariate(2, math.sqrt(0.6))
            x2 = random.normalvariate(3, math.sqrt(0.6))
            x_data.append([1.0, x1, x2])
            y_data.append([1.0])
        else:                                           # mean = [0, 4]  cov = [[0.4, 0], [0, 0.4]]
            x1 = random.normalvariate(0, math.sqrt(0.4))
            x2 = random.normalvariate(4, math.sqrt(0.4))
            x_data.append([1.0, x1, x2])
            y_data.append([-1.0])

    for i in range(20):                                 #outlier data
        x1 = random.normalvariate(6, math.sqrt(0.3))
        x2 = random.normalvariate(0, math.sqrt(0.1))
        x_data.append([1.0, x1, x2])
        y_data.append([1.0])
        

def test_generate(x_data, y_data):
    for i in range(5000):
        y = random.choice([-1, 1])
        if y == 1:                                        # mean = [2, 3]  cov = [[0.6, 0], [0, 0.6]]
            x1 = random.normalvariate(2, math.sqrt(0.6))
            x2 = random.normalvariate(3, math.sqrt(0.6))
            x_data.append([1.0, x1, x2])
            y_data.append([1.0])
        else:                                             # mean = [0, 4]  cov = [[0.4, 0], [0, 0.4]]
            x1 = random.normalvariate(0, math.sqrt(0.4))
            x2 = random.normalvariate(4, math.sqrt(0.4))
            x_data.append([1.0, x1, x2])
            y_data.append([-1.0]) 

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def main():
    iter = 100
    e_in = 0
    e_out = 0
    train_size = 200
    test_size = 5000


    for i in range(iter):
        random.seed(i)
        x_traindata = []
        y_traindata = []
        train_generate(x_traindata, y_traindata)
        x_traindata = np.array(x_traindata)
        y_traindata = np.array(y_traindata)

        w_op = np.linalg.pinv(x_traindata).dot(y_traindata)
        predict_trainy = x_traindata.dot(w_op)



        e_in+=((np.sign(predict_trainy)!=y_traindata).sum()/train_size)
        #e_in+=np.linalg.norm(predict_trainy-y_traindata)/train_size

     ###################################################################

        x_testdata = []
        y_testdata = []
        test_generate(x_testdata, y_testdata)
        x_testdata = np.array(x_testdata)
        y_testdata = np.array(y_testdata)

        predict_testy = x_testdata.dot(w_op)

        e_out+=((np.sign(predict_testy)!=y_testdata).sum()/test_size)
        #e_out+=((np.sign(predict_testy)!=y_testdata).sum()/test_size)
        #e_out+=np.linalg.norm(predict_testy-y_testdata)/test_size        

    e_in/=iter
    e_out/=iter
    print(e_in, e_out, abs(e_in-e_out))
     
if __name__ == '__main__':
    main()