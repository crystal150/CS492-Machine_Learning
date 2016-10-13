# 20130479 Jaryong Lee

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



def cost_function_linear( y, x, beta ):
    # print "Calculate cost ..."

    temp_square = (beta.dot(x.transpose())-y)
    return np.sum(temp_square.transpose().dot(temp_square))/(2*y.shape[0])

def anlaytic_solution ( y, x ):
    # print "Calculate beta analytically ..."

    return (np.linalg.pinv(x.transpose().dot(x))).dot(x.transpose()).dot(y.transpose())

def gradient_descent ( y, x, beta, type='linear' ):
    # print "Calculate gradient descent ..."

    if type == 'linear':
        return (beta.dot(x.transpose())-y).dot(x)/y.shape[0]
    elif type == 'logistic':
        return (sigmoid(beta.dot(x.transpose()))-y).dot(x)/y.shape[0]

def prediction_error_linear ( y, x, beta ):
    # print "Calculate prediction error ..."

    return np.sum(np.abs(y - beta.dot(x.transpose())))/y.shape[0]

def sigmoid ( z ):
    return 1 / (1 + np.exp(-z))

def cost_function_logistic ( y, x, beta ):
    # print "Calculate cost ..."

    temp_sig = sigmoid(beta.dot(x.transpose())).transpose()
    return np.sum((-y).dot(np.log(temp_sig)) - (1-y).dot(np.log(1-temp_sig)))/y.shape[0]

def prediction_error_logistic ( y, x, beta ):
    # print "Calculate prediction error ..."

    return np.sum(np.abs(y - sigmoid(beta.dot(x.transpose()))))/y.shape[0]

def backtracking_line_search ( y, x, beta, step_size, back, ratio):
    # print "Calculate gradient descent ..."
    gradient = gradient_descent (y, x, beta, 'logistic')
    while True:
        cost_actual = cost_function_logistic (y, x, beta - step_size * gradient)
        cost_expected = cost_function_logistic (y, x, beta) + back * step_size * np.sum(gradient.transpose().dot(gradient))
        if cost_actual < cost_expected:
            break
        step_size *= ratio

    return beta - step_size * gradient

if __name__ == "__main__":
    print_with_bar ("Homework #1. 20130479. Jaryong Lee.")

    num_set = 10
    iteration = 100
    linear_gradient_step_size = 0.15
    logistic_gradient_step_size = 1
    logistic_backtracking_step_size = 10
    backtracking_hyperparameter = 0.01
    backtracking_ratio = 0.5

    data_set = load_data ("housing_scale")
    splited_data_set = split_set (num_set, 0.8, data_set)

    # TEST split_set
    # splited_data_set = split_set (2, 0.8, list(np.random.random((10, 5))))
    # print splited_data_set
    

    # Linear Regression --- Analytic Solution
    print_with_bar ("Linear Regression --- Analytic Solution")
    print "Calculate mean prediction error from beta by anlaytic solution ..."

    first = True
    cost_analytic = []
    errors_linear_analytic = []
    for splited_data in splited_data_set:
        train_set_y, train_set_x = splited_data[0]
        test_set_y, test_set_x = splited_data[1]
        
        beta_linear_analytic = anlaytic_solution (train_set_y, train_set_x)
        # TEST anlaytic_solution
        # print "beta : %s" % beta_linear_analytic

        error = prediction_error_linear (test_set_y, test_set_x, beta_linear_analytic)
        # TEST prediction_error_linear
        # print "error : %s" % error

        errors_linear_analytic.append(error)
        cost_analytic.append(cost_function_linear (test_set_y, test_set_x, beta_linear_analytic))

    cost_analytic = iteration * [np.mean(cost_analytic)]
        
    print_with_bar ( "Prediction error: %s by %s splited set --- Analytic Solution" % (np.mean(errors_linear_analytic), num_set) )


    # Linear Regression --- Gradient Descent
    print_with_bar ("Linear Regression --- Gradient Descent")
    print "Calculate mean prediction error from beta by gradient descent ..."

    
    first = True
    cost_gradient = []
    errors_linear_gradient = []
    for splited_data in splited_data_set:
        train_set_y, train_set_x = splited_data[0]
        test_set_y, test_set_x = splited_data[1]

        beta_linear_gradient = np.matrix(np.zeros(train_set_x.shape[1]))
        
        for _ in range(iteration):
            gradient = gradient_descent (train_set_y, train_set_x, beta_linear_gradient)
            # TEST gradient_descent
            # print "gradient : %s" % gradient
            
            beta_linear_gradient -= linear_gradient_step_size * gradient
            if first:
                cost_gradient.append (cost_function_linear (test_set_y, test_set_x,
                    beta_linear_gradient))

        error = prediction_error_linear (test_set_y, test_set_x, beta_linear_gradient)
        first = False
        # TEST prediction_error_linear
        # print "error : %s" % error

        errors_linear_gradient.append(error)
    print_with_bar ("Prediction error: %s by %s iteration %s splited set --- Gradient Descent" % (np.mean(errors_linear_gradient), iteration, num_set) )

    plt_iter = range(iteration)
    plt.plot(plt_iter, cost_analytic, plt_iter, cost_gradient)
    plt.ylabel('Cost')
    plt.show()


    data_set = load_data ("a3a")
    splited_data_set = split_set (num_set, 0.8, data_set)


    # Logistic Regression --- Gradient Descent
    print_with_bar ("Logistic Regression --- Gradient Descent")
    print "Calculate mean prediction error from beta by gradient descent ..."

    
    first = True
    cost_gradient = []
    errors_logistic_gradient = []
    for splited_data in splited_data_set:
        train_set_y, train_set_x = splited_data[0]
        test_set_y, test_set_x = splited_data[1]
        train_set_y = (train_set_y + 1) / 2
        test_set_y = (test_set_y + 1) / 2

        beta_logistic_gradient = np.matrix(np.zeros(train_set_x.shape[1]))
        
        for _ in range(iteration):
            gradient = gradient_descent (train_set_y, train_set_x, beta_logistic_gradient,
                    'logistic')
            # TEST gradient_descent
            # print "gradient : %s" % gradient
            
            beta_logistic_gradient -= logistic_gradient_step_size * gradient
            if first:
                cost_gradient.append (cost_function_logistic (test_set_y, test_set_x,
                    beta_logistic_gradient))

        error = prediction_error_logistic (test_set_y, test_set_x, beta_logistic_gradient)
        first = False
        # TEST prediction_error_linear
        # print "error : %s" % error

        errors_logistic_gradient.append(error)
    print_with_bar ("Prediction error: %s by %s iteration %s splited set --- Gradient Descent" % (np.mean(errors_logistic_gradient), iteration, num_set) )
    
    plt_iter = range(iteration)
    plt.plot(plt_iter, cost_gradient)
    plt.ylabel('Cost')
    plt.show()



    # Logistic Regression --- Backtracking Line Search
    print_with_bar ("Logistic Regression --- Backtracking Line Search")
    print "Calculate mean prediction error from beta by gradient descent ..."

    
    first = True
    cost_gradient = []
    errors_logistic_backtracking = []
    for splited_data in splited_data_set:
        train_set_y, train_set_x = splited_data[0]
        test_set_y, test_set_x = splited_data[1]
        train_set_y = (train_set_y + 1) / 2
        test_set_y = (test_set_y + 1) / 2

        beta_logistic_backtracking = np.matrix(np.zeros(train_set_x.shape[1]))
        
        for _ in range(iteration):
            beta_logistic_backtracking = backtracking_line_search (train_set_y, train_set_x,
                    beta_logistic_backtracking, logistic_backtracking_step_size,
                    backtracking_hyperparameter, backtracking_ratio)
            if first: cost_gradient.append (cost_function_logistic (test_set_y, test_set_x,
                beta_logistic_backtracking))

        error = prediction_error_logistic (test_set_y, test_set_x, beta_logistic_backtracking)
        first = False
        # TEST prediction_error_linear
        # print "error : %s" % error

        errors_logistic_backtracking.append(error)
    print_with_bar ("Prediction error: %s by %s iteration %s splited set --- Backtracking Line Search" % (np.mean(errors_logistic_backtracking), iteration, num_set) )
    
    plt_iter = range(iteration)
    plt.plot(plt_iter, cost_gradient)
    plt.ylabel('Cost')
    plt.show()
