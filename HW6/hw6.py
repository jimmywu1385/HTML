import numpy as np
import math

def main():
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    train_size = 0
    test_size = 0
    feature = 0
    T = 460
    u_set = [[]]
    g_set = []
    at = []

    with open('hw6_train.dat.txt', 'r') as f:
        ind = 0
        while True:
            line = f.readline()
            if line == '\n' or len(line) == 0:
                break
            x = []
            feature = len(line.split())-1
            for i in line.split()[:-1]:
                x.append(float(i))
            x.append(ind)
            ind+=1
            train_x.append(x)
            train_y.append(float(line.split()[-1]))
            train_size+=1

    with open('hw6_test.dat.txt', 'r') as f:
        while True:
            line = f.readline()
            if line == '\n' or len(line) == 0:
                break
            x = [float(i) for i in line.split()[:-1]]     
            test_x.append(x)
            test_y.append(float(line.split()[-1]))
            test_size+=1    
    
    for i in range(train_size):
        u_set[0].append(1/train_size)
    u_sum = 1

    min_ein_g = 1
    for i in range(T):
        # obtain gt
        min = 1
        min_ind = [0, 0, 0]
        for j in range(feature):
            sort_train = sorted(train_x, key = lambda s:s[j])
            for k in range(2):
                min_f = 0
                min_ind_f = 0
                sum_f = 0
                base = 0
                for l in range(train_size-1):
                    if (2*k-1) * train_y[sort_train[l][feature]] == 1.0:
                        sum_f+=u_set[i][sort_train[l][feature]]
                    else:
                        sum_f-=u_set[i][sort_train[l][feature]]
                        base+=u_set[i][sort_train[l][feature]]
                    if sum_f < min_f:
                        min_f = sum_f
                        min_ind_f = l+1
                min_f = min_f+base
                if min_ind_f == 0:
                    min_ind_f = sort_train[0][j] - 1
                else:
                    min_ind_f = (sort_train[min_ind_f][j] + sort_train[min_ind_f-1][j]) / 2
                
                if min_f<min:
                    min = min_f
                    min_ind = [2*k-1, j, min_ind_f, min_f/u_sum]
    
        g_set.append(min_ind)

        #compute parm
        parm = math.sqrt((1-min_ind[3])/min_ind[3])

        # update ut to ut+1
        u_sum = 0
        u_set.append([]) 
        for j in range(train_size):
            if min_ind[0] * np.sign(train_x[j][min_ind[1]] - min_ind[2]) == train_y[j]:
                u_set[i+1].append(u_set[i][j]/parm)
                u_sum+=u_set[i][j]/parm
            else:
                u_set[i+1].append(u_set[i][j]*parm)
                u_sum+=u_set[i][j]*parm

        # compute at 
        at.append(math.log(parm))

        # compute ein_g
        ein_g = 0
        for j in range(train_size):
            sum = 0
            for k in range(i+1):
                sum+= at[k] * g_set[k][0] * np.sign(train_x[j][g_set[k][1]] - g_set[k][2])
            if np.sign(sum) != train_y[j]:        
                ein_g+=1/train_size
        #print(ein_g)

        if ein_g < min_ein_g:
            min_ein_g = ein_g
    print(min_ein_g)

    eout_g = 0
    for i in range(test_size):
            sum = 0
            for k in range(T):
                sum+= at[k] * g_set[k][0] * np.sign(test_x[i][g_set[k][1]] - g_set[k][2])
            if np.sign(sum) != test_y[i]:        
                eout_g+=1/test_size
    print(eout_g)

if __name__ == '__main__':
    main()