#################################################
##### CS492 Homework#2 20130479 Jaryong Lee #####
#################################################

import numpy as np
from scipy.sparse import csr_matrix
import random
import matplotlib.pyplot as plt

def load_data ( file_name ):
    print "Load data from %s ..." % file_name 

    lines = [line.strip().split() for line in open(file_name).readlines()]
    data_set = [([line[0]] + [ele.split(":") for ele in line[1:]]) for line in lines]
    # data_set [ ['24', ('1', '10'), ('2', '353'), ... ('13', '123')], [...], ... [...] ]

    return data_set

def csr_matrix_from_indices ( x, dim_y, dim_x ):
    row = []
    col = []
    data = []
    for i in range(len(x)):
        row.extend([i for ele in x[i]])
        col.extend([int(ele[0])-1 for ele in x[i]])
        data.extend([float(ele[1]) for ele in x[i]])
    sparse = csr_matrix ( (data, (row, col)), shape = (dim_y, dim_x), dtype = float)
    return np.mat(sparse.toarray())

def split_set ( num, ratio, data_set ):
    print "Split data set to train and test set randomly ..."

    argnum = max([int(line[-1][0]) for line in data_set])
    full_splited_yx = []
    for _ in range(num):
        random.shuffle(data_set)
        num_training = int (ratio * len(data_set))
        splited_set = [data_set[:num_training], data_set[num_training:]]
        splited_y = [np.array([line[0] for line in set], dtype=float) for set in splited_set]
        splited_x_temp = [[line[1:] for line in set] for set in splited_set]
        splited_x = [csr_matrix_from_indices (x, len(x), argnum) for x in splited_x_temp]
        splited_yx_train = [splited_y[0], splited_x[0]]
        splited_yx_test = [splited_y[1], splited_x[1]]
        full_splited_yx.append([splited_yx_train, splited_yx_test])

    return full_splited_yx

def print_with_bar ( print_temp ):
    print "-" * len(print_temp)
    print print_temp
    print "-" * len(print_temp)

if __name__ == "__main__":
    print_with_bar ("Homework #2. 20130479. Jaryong Lee.")

    ############################
    ##### HYPER PARAMETERS #####
    ############################

    data_set = load_data ("housing_scale")
    splited_data_set = split_set (num_set, 0.8, data_set)

    # TEST split_set
    # splited_data_set = split_set (2, 0.8, list(np.random.random((10, 5))))
    # print splited_data_set
    
    #########################################
    # SVM
    #########################################

    plt.plot(plt_iter, cost_analytic, plt_iter, cost_gradient)
    plt.ylabel('Cost')
    plt.show()
