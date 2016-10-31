#################################################
##### CS492 Homework#2 20130479 Jaryong Lee #####
#################################################

import cvxopt 
import numpy as np
from scipy.sparse import csr_matrix
import random
#import matplotlib.pyplot as plt

def load_data ( file_name ):
    print "Load data from %s ..." % file_name 

    load_data = open(file_name).readlines()
    cols = []
    for line in load_data:
        col = [int(ele.split(":")[0]) for ele in line.split()[1:]]
        cols.extend(col)
    return load_data, max(cols)

def load_spmat_data ( data_lines, max_col ):
    print "Form sparse matrix ..." 

    row, col, data = [[], [], []]
    max_row = len(data_lines)

    for rownum, line in enumerate(data_lines):
        matdat = [ele.split(":") for ele in line.strip().split()[1:]]
        for column, value in matdat:
            colnum = int(column)-1
            # Append to row, col, data
            row.append(rownum)
            col.append(colnum)
            data.append(int(value))
    return cvxopt.spmatrix(data, row, col, (max_row, max_col))

def pick_ith_over_n ( lst, i, n ):
    return lst[(i-1)*len(lst)/n : i*len(lst)/n]
    
def print_with_bar ( print_temp ):
    print "-" * len(print_temp)
    print print_temp
    print "-" * len(print_temp)

if __name__ == "__main__":
    print_with_bar ("Homework #2. 20130479. Jaryong Lee.")

    ############################
    ##### HYPER PARAMETERS #####
    ############################

    data_lines, max_col = load_data ("a1a")
    random.shuffle(data_lines)
    data = load_spmat_data (pick_ith_over_n(data_lines, 1, 5), max_col)
    print data

    #########################################
    # SVM
    #########################################
    """
    plt.plot(plt_iter, cost_analytic, plt_iter, cost_gradient)
    plt.ylabel('Cost')
    plt.show()
    """
