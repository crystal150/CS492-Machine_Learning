#################################################
##### CS492 Homework#2 20130479 Jaryong Lee #####
#################################################

import cvxopt 
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
    row, col, data = [[], [], []]
    max_row = len(data_lines)

    for rownum, line in enumerate(data_lines):
        matdat = [ele.split(":") for ele in line.strip().split()[1:]]
        for column, value in matdat:
            colnum = int(column)-1
            # Append to row, col, data
            row.append(rownum)
            col.append(colnum)
            data.append(float(value))
    return cvxopt.spmatrix(data, row, col, (max_row, max_col))

def load_values ( data_lines ):
    return [float(line.strip().split()[0]) for line in data_lines]

def pick_ith_over_n ( lst, i, n ):
    fr = i*len(lst)/n
    to = (i+1)*len(lst)/n
    # Train set, Validation set
    return lst[:fr] + lst[to:], lst[fr:to]
    
def print_with_bar ( print_temp ):
    print "-" * len(print_temp)
    print print_temp
    print "-" * len(print_temp)

def SVM ( X, y, C ):
    Y = cvxopt.cvxopt.spdiag(y)
    P = Y * X * X.T * Y
    q = cvxopt.matrix([-1.0]*len(y))
    G = cvxopt.spmatrix([1.0]*len(y) + [-1.0]*len(y), range(2*len(y)), 2*range(len(y)))
    h = cvxopt.matrix([C]*len(y) + [0.0]*len(y))
    A = cvxopt.matrix(y).T
    b = cvxopt.matrix(0.0)

    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = sol['x']

    supp_alpha, supp_i = min([(value, i) for i, value in enumerate(alpha) if value >= C/2])
    ALPHA = cvxopt.cvxopt.spdiag(alpha)

    w = X.T*Y*alpha
    b = y[supp_i] - X[supp_i, :]*X.T*Y*alpha

    return w, b

if __name__ == "__main__":
    print_with_bar ("Homework #2. 20130479. Jaryong Lee.")

    ############################
    ##### HYPER PARAMETERS #####
    ############################

    piece_num = 5
    C_list = [2.0**i for i in range(-10, 10)]

    #####################
    ##### DATA LOAD #####
    #####################

    data_lines, max_col = load_data ("a1a")
    random.shuffle(data_lines)

    #################################
    # SVM - NO KERNEL TRICK VERSION #
    #################################

    best_C = 0
    lowest_val_error = 100
    for C in C_list:
        train_errors = []
        validation_errors = []
        # Take prediction errors for train & validation --> total 10 times
        for pick_num in range(piece_num):
            train, validation = pick_ith_over_n(data_lines, pick_num, piece_num)
            train_X = load_spmat_data (train, max_col)
            train_y = load_values (train)
            train_Y = cvxopt.cvxopt.spdiag (train_y)

            w, b = SVM (train_X, train_y, C)
            
            train_correct = len([ele for ele in train_Y*(train_X*w + b) if ele >= 0])
            train_errors.append(100.0*train_correct/len(train_y))
            
            validation_X = load_spmat_data (validation, max_col)
            validation_y = load_values (validation)
            validation_Y = cvxopt.cvxopt.spdiag (validation_y)

            validation_correct = len([ele for ele in validation_Y*(validation_X*w + b) if ele >= 0])
            validation_errors.append(100.0*validation_correct/len(validation_y))

        train_error = sum(train_errors) / len(train_errors)
        validation_error = sum(validation_errors) / len(validation_errors)
        print "C: %s, train_error: %s, validation_error: %s" % (C, train_error, validation_error)
        if validation_error < lowest_val_error:
            lowest_val_error = validation_error
            best_C = C
    print "Best C: %s" % best_C

    # Draw training prediction error avg
    # Draw validation prediction error avg
    # Find best C for second one!

    """
    plt.plot(plt_iter, cost_analytic, plt_iter, cost_gradient)
    plt.ylabel('Cost')
    plt.show()
    """
