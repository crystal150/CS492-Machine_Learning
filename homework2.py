#################################################
##### CS492 Homework#2 20130479 Jaryong Lee #####
#################################################

import cvxopt 
import random
from numpy import sqrt, pi, average, var
from numpy.random import multivariate_normal
from numpy.linalg import inv, det
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

    _, supp_i = min([(value, i) for i, value in enumerate(alpha) if value >= C/2])

    w = X.T*Y*alpha
    b = y[supp_i] - X[supp_i, :]*X.T*Y*alpha

    return w, b


def SVM_no_kernel_trick (train_set, test_set, max_col, cross_validation_pieces, C_list):
    best_C = 0
    lowest_val_error = 100
    for C in C_list:
        train_errors = []
        validation_errors = []
        # Take prediction errors for train & validation
        # Total 10 times
        for pick_num in range(cross_validation_pieces):
            train, validation = pick_ith_over_n(train_set, pick_num, piece_num)

            # Load train data
            train_X = load_spmat_data (train, max_col)
            train_y = load_values (train)
            train_Y = cvxopt.cvxopt.spdiag (train_y)

            # Get w, b from SVM computation
            w, b = SVM (train_X, train_y, C)
            
            # Calculate train error
            train_wrong = len([ele for ele in train_Y*(train_X*w + b) if ele < 0])
            train_errors.append(100.0*train_wrong/len(train_y))
            
            # Load test data
            validation_X = load_spmat_data (validation, max_col)
            validation_y = load_values (validation)
            validation_Y = cvxopt.cvxopt.spdiag (validation_y)

            # Calculate validation error
            validation_wrong = len([ele for ele in validation_Y*(validation_X*w + b) if ele < 0])
            validation_errors.append(100.0*validation_wrong/len(validation_y))

        # Calculate average train, validation errors
        train_error = sum(train_errors) / len(train_errors)
        validation_error = sum(validation_errors) / len(validation_errors)
        print "C: %s, train_error: %s, validation_error: %s" % (C, train_error, validation_error)
        # To find best C, update lowest validation error & C
        if validation_error <= lowest_val_error:
            lowest_val_error = validation_error
            best_C = C

    print "Best C: %s" % best_C
    
    # Load train data from total train set
    train_X = load_spmat_data (train_set, max_col)
    train_y = load_values (train_set)
    train_Y = cvxopt.cvxopt.spdiag (train_y)

    # Get w, b from SVM computation
    w, b = SVM (train_X, train_y, best_C)

    # Calculate train error
    train_wrong = len([ele for ele in train_Y*(train_X*w + b) if ele < 0])
    train_error = 100.0*train_wrong/len(train_y)
    
    # Load test data
    test_X = load_spmat_data (test_set, max_col)
    test_y = load_values (test_set)
    test_Y = cvxopt.cvxopt.spdiag (test_y)
    
    # Calculate test error
    test_wrong = len([ele for ele in test_Y*(test_X*w + b) if ele < 0])
    test_error = 100.0*test_wrong/len(test_y)

    print_with_bar ("train_error: %s\ntest_error: %s" %( train_error, test_error ))

    # Draw training prediction error avg for each C
    # Draw validation prediction error avg for each C

    """
    plt.plot(plt_iter, cost_analytic, plt_iter, cost_gradient)
    plt.ylabel('Cost')
    plt.show()
    """

def Gaussian_kernel ( X, Y, sigma ):
    sample_num_X = X.size[0]
    sample_num_Y = Y.size[0]
    one_vector_X = cvxopt.matrix([1.0]*sample_num_X)
    one_vector_Y = cvxopt.matrix([1.0]*sample_num_Y)
    sqX = X*X.T
    sqY = Y*Y.T
    XY = X*Y.T
    diag_X = sqX[::sample_num_X+1]
    diag_Y = sqY[::sample_num_Y+1]

    edm = one_vector_X*diag_Y.T - 2*XY + diag_X*one_vector_Y.T
    return cvxopt.exp(-edm/(2*sigma**2))
    

def SVM_gaussian_kernel ( X, y, C, gaussian ):
    Y = cvxopt.cvxopt.spdiag(y)
    P = Y * gaussian * Y
    q = cvxopt.matrix([-1.0]*len(y))
    G = cvxopt.spmatrix([1.0]*len(y) + [-1.0]*len(y), range(2*len(y)), 2*range(len(y)))
    h = cvxopt.matrix([C]*len(y) + [0.0]*len(y))
    A = cvxopt.matrix(y).T
    b = cvxopt.matrix(0.0)

    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = sol['x']

    _, supp_i = min([(value, i) for i, value in enumerate(alpha) if value >= C/2])

    # w = X.T*Y*alpha
    b = y[supp_i] - gaussian[supp_i, :]*Y*alpha

    return alpha, b

def SVM_kernel_trick (train_set, test_set, max_col, cross_validation_pieces, C_list, sigma_list):
    best_C = 0
    best_sigma = 0
    lowest_val_error = 100
    for C, sigma in [(C, sigma) for C in C_list for sigma in sigma_list]:
        train_errors = []
        validation_errors = []
        # Take prediction errors for train & validation
        # Total 10 times
        for pick_num in range(cross_validation_pieces):
            train, validation = pick_ith_over_n(train_set, pick_num, piece_num)

            # Load train data
            train_X = load_spmat_data (train, max_col)
            train_y = load_values (train)
            train_Y = cvxopt.cvxopt.spdiag (train_y)
            train_gaussian = Gaussian_kernel(train_X, train_X, sigma)

            # Get w, b from SVM computation
            alpha, b = SVM_gaussian_kernel (train_X, train_y, C, train_gaussian)
            
            # Calculate train error
            train_wrong = len([ele for ele in train_Y*(train_gaussian*train_Y*alpha + b) if ele <= 0])
            train_errors.append(100.0*train_wrong/len(train_y))
            
            # Load test data
            validation_X = load_spmat_data (validation, max_col)
            validation_y = load_values (validation)
            validation_Y = cvxopt.cvxopt.spdiag (validation_y)
            validation_gaussian = Gaussian_kernel(validation_X, train_X, sigma)
            
            # Calculate validation error
            validation_wrong = len([ele for ele in
                validation_Y*(validation_gaussian*train_Y*alpha + b) if ele <= 0])
            validation_errors.append(100.0*validation_wrong/len(validation_y))

        # Calculate average train, validation errors
        train_error = sum(train_errors) / len(train_errors)
        validation_error = sum(validation_errors) / len(validation_errors)
        print "C: %s, sigma: %s, train_error: %s, validation_error: %s" % (C, sigma, train_error, validation_error)
        # To find best C, update lowest validation error & C
        if validation_error <= lowest_val_error:
            lowest_val_error = validation_error
            best_C = C
            best_sigma = sigma

    print "Best C: %s, Best sigma: %s" % (best_C, best_sigma)
    # Load train data from total train set
    train_X = load_spmat_data (train_set, max_col)
    train_y = load_values (train_set)
    train_Y = cvxopt.cvxopt.spdiag (train_y)
    train_gaussian = Gaussian_kernel(train_X, train_X, best_sigma)

    # Get w, b from SVM computation
    alpha, b = SVM_gaussian_kernel (train_X, train_y, best_C, train_gaussian)

    # Calculate train error
    train_wrong = len([ele for ele in train_Y*(train_gaussian*train_Y*alpha + b) if ele <= 0])
    train_error = 100.0*train_wrong/len(train_y)
    
    # Load test data
    test_X = load_spmat_data (test_set[:1000], max_col)
    test_y = load_values (test_set[:1000])
    test_Y = cvxopt.cvxopt.spdiag (test_y)
    test_gaussian = Gaussian_kernel(test_X, train_X, best_sigma)
    
    # Calculate test error
    test_wrong = len([ele for ele in test_Y*(test_gaussian*train_Y*alpha + b) if ele <= 0])
    test_error = 100.0*test_wrong/len(test_y)

    print_with_bar ("train_error: %s\ntest_error: %s" %( train_error, test_error ))

    # Draw training prediction error avg for each C
    # Draw validation prediction error avg for each C

    """
    plt.plot(plt_iter, cost_analytic, plt_iter, cost_gradient)
    plt.ylabel('Cost')
    plt.show()
    """

def Gaussian_Probability ( X, mu, variance ):
    sample_num_X = X.size[0]
    sample_num_mu = mu.size[0]
    one_vector_X = cvxopt.matrix([1.0]*sample_num_X)
    one_vector_mu = cvxopt.matrix([1.0]*sample_num_mu)
    sqX = X*X.T
    sqmu = mu*mu.T
    Xmu = X*mu.T
    diag_X = sqX[::sample_num_X+1]
    diag_mu = sqmu[::sample_num_mu+1]

    edm = one_vector_X*diag_mu.T - 2*Xmu + diag_X*one_vector_mu.T
    return cvxopt.exp(cvxopt.matrix([[-edm[:, i]/(2*variance[i])] for i in range(edm.size[1])]))

def Gaussian_Mixture_Models():
    K = 2
    n = 1000
    mean = [30, 30]
    cov = [[100, 0], [0, 30]]  # diagonal covariance
    cov = cvxopt.spdiag([100, 30])
    print cov
    X = cvxopt.matrix(multivariate_normal(mean, cov, n/2), (n, 1))
    
    mu = cvxopt.matrix([average(X)/2, average(X)*3/2])
    variance = cvxopt.matrix([1, 1])
    PI = cvxopt.matrix([1.0/K]*K)
    try:
        while (True):
            # E algorithm
            normal = Gaussian_Probability(X, mu, variance)
            normal_sum_k = normal*PI
            normal_j = normal*cvxopt.spdiag(PI)
            
            P = cvxopt.matrix([normal_j[i, :]/normal_sum_k[i] for i in range(n)])

            # M algorithm
            N = cvxopt.matrix([sum(P[:, i]) for i in range(P.size[1])])
            x_subs_mu = cvxopt.matrix([[X-mu[i]] for i in range(K)])

            mu_new = cvxopt.matrix([sum(X.T*P[:, i])/N[i] for i in range(K)])
            variance_new = cvxopt.matrix([(x_subs_mu[:, i].T*cvxopt.spdiag(P[:, i])*x_subs_mu[:,
                i])[0] / N[i] for i in range(K)])
            PI = N/n
            if 1-(PI.T*PI)[0] <= 10**(-5):
                print "hello"
                raise ZeroDivisionError()
            if sum(abs(mu-mu_new)) <= 10**(-5) and sum(abs(variance-variance_new)) <= 10**(-10): break
            mu = mu_new
            variance = variance_new

        print_with_bar("GMM with 2 single Gaussian models")
        print "mu:\n%svariance:\n%spi:\n%s" %(mu, variance, PI)

    except ZeroDivisionError:
        print_with_bar("It is close to the single Gaussian model")
        dominant = max([(value, i) for i, value in enumerate(PI)])[1]
        print "mu:\n[%s]\nvariance:\n[%s]" %(mu[dominant], variance[dominant])








if __name__ == "__main__":
    print_with_bar ("Homework #2. 20130479. Jaryong Lee.")

    ############################
    ##### HYPER PARAMETERS #####
    ############################

    piece_num = 5
    C_list = [2.0**i for i in range(8, 10)]
    sigma_list = [9+i/2.5 for i in range(1, 21)]

    #####################
    ##### DATA LOAD #####
    #####################

    train_set, train_max_col = load_data ("a1a")
    test_set, test_max_col = load_data ("a1a.t")
    max_col = max(train_max_col, test_max_col)
    print "Shuffle train data set"
    random.shuffle(train_set)

    while (True):
        while (True):
            sel = raw_input("1: Support Vector Machines\n2: Kernel trick\n3: Gaussian Mixture Models\nq: quit\n>>> ")
            if (sel == '1' or sel == '2' or sel == '3' or sel == 'q'): break
            print "Please input 1 | 2 | 3\n>>> "
            
        if sel == '1':
            #################################
            # SVM - NO KERNEL TRICK VERSION #
            #################################

            SVM_no_kernel_trick(train_set, test_set, max_col, piece_num, C_list)

        elif sel == '2':
            ##############################
            # SVM - KERNEL TRICK VERSION #
            ##############################

            SVM_kernel_trick(train_set, test_set, max_col, piece_num, C_list, sigma_list)

        elif sel == '3':
            ###########################
            # GAUSSIAN MIXTURE MODELS #
            ###########################
            Gaussian_Mixture_Models()

        elif sel == 'q':
            # Quit
            break
