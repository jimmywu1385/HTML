import numpy as np
import random

def main():
    n = 100
    iter = 1000
    x_data = []
    y_data = []
    length = 0.0

    with open('hw1_train.dat.txt', 'r') as f:
        while True:
            line = f.readline()
            if line == '\n' or len(line) == 0:
                break
            x = [float(i) for i in line.split()[:-1]]     # 14題 x = [float(i)*2 for i in line.split()[:-1]]
            x.append(1.000000)                            # 16題 x.append(0.000000)
            x_data.append(np.array([x]))
            y_data.append(np.array([float(line.split()[-1])]))

    for i in range(iter):
        random.seed(i)
        w =np.zeros((1,11))
        condition = True
        safe_time = 0

        while condition:
            if safe_time >= 5*n:
                break
            ind = random.randint(0,n-1)  
            if np.sign(np.dot(x_data[ind], w.T)) != y_data[ind]:    # 15題 if np.sign(np.dot(x_data[ind]/np.linalg.norm(x_data[ind]), w.T)) != y_data[ind]:
                w += y_data[ind]*x_data[ind]                        # 15題 w += y_data[ind]*(x_data[ind]/np.linalg.norm(x_data[ind]))
                safe_time = 0
            else:
                safe_time += 1
        
        length += np.linalg.norm(w)*np.linalg.norm(w)

    print(length/iter)
        
if __name__ == '__main__':
    main()
