from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for idx, c in enumerate(data["is_correct"]):
        _theta = theta[data["user_id"][idx]]
        _beta = beta[data["question_id"][idx]]
        log_lklihood += ((c * (_theta - _beta)) - np.log(1 + np.exp(_theta - _beta)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def neg_log_likelihood3(data, theta, gamma, beta, alpha):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param gamma: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for idx, c in enumerate(data["is_correct"]):
        _theta = theta[data["user_id"][idx]]
        _gamma = gamma[data["user_id"][idx]]
        _beta = beta[data["question_id"][idx]]
        _alpha = alpha[data["question_id"][idx]]
        _x = _alpha * (_theta - _beta)
        log_lklihood += (
            (c * np.log(_gamma + np.exp(_x))) -
            (np.log(1 + np.exp(_x))) +
            ((1 - c) * np.log(1 - _gamma))
        )
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Initialize the partial derivative to be ready for summation
    N, D = theta.shape[0], beta.shape[0]
    partial_theta = np.zeros(N)
    partial_beta = np.zeros(D)
    for idx, c in enumerate(data["is_correct"]):
        _theta = theta[data["user_id"][idx]]
        _beta = beta[data["question_id"][idx]]
        # Summation
        partial_theta[data["user_id"][idx]] += (c - sigmoid(_theta - _beta))
        partial_beta[data["question_id"][idx]] += (- c + sigmoid(_theta - _beta))
    theta += (lr * partial_theta)
    beta += (lr * partial_beta)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def update_theta_beta3(data, lr, theta, gamma, beta, alpha):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param gamma: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Initialize the partial derivative to be ready for summation
    N, D = theta.shape[0], beta.shape[0]
    partial_theta = np.zeros(N)
    partial_gamma = np.zeros(N)
    partial_beta = np.zeros(D)
    partial_alpha = np.zeros(D)
    for idx, c in enumerate(data["is_correct"]):
        _theta = theta[data["user_id"][idx]]
        _gamma = gamma[data["user_id"][idx]]
        _beta = beta[data["question_id"][idx]]
        _alpha = alpha[data["question_id"][idx]]
        _x = _alpha * (_theta - _beta)
        # Summation
        partial_theta[data["user_id"][idx]] += (
            ((_alpha * c * np.exp(_x)) / (_gamma + np.exp(_x))) -
            (_alpha * sigmoid(_x))
        )
        partial_gamma[data["user_id"][idx]] += (
            (c / (_gamma + np.exp(_x))) -
            ((1 - c) / (1 - _gamma))
        )
        partial_beta[data["question_id"][idx]] += (
            -((_alpha * c * np.exp(_x)) / (_gamma + np.exp(_x))) +
            (_alpha * sigmoid(_x))
        )
        partial_alpha[data["question_id"][idx]] += (
            (((_theta - _beta) * c * np.exp(_x)) / (_gamma + np.exp(_x))) -
            ((_theta - _beta) * sigmoid(_x))
        )
    theta += (lr * partial_theta)
    gamma += (lr * partial_gamma)
    beta += (lr * partial_beta)
    alpha += (lr * partial_alpha)
    gamma[gamma >= 1] = 0.99
    gamma[gamma <= 0] = 0.01
    alpha[alpha <= 0] = 0.01
    alpha[alpha >= 2] = 1.99
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, gamma, beta, alpha


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(542)
    beta = np.zeros(1774)

    train_acc_lst = []
    train_nllk_lst = []
    val_acc_lst = []
    val_nllk_lst = []

    train_neg_lld = 0
    train_score = 0
    val_neg_lld = 0
    val_score = 0
    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_score = evaluate(data=data, theta=theta, beta=beta)
        train_acc_lst.append(train_score)
        train_nllk_lst.append(train_neg_lld)

        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(val_score)
        val_nllk_lst.append(val_neg_lld)

        # print("Train_NLLK: {} \t Train_Score: {} \t"
        #       "Val_NLLK: {} \t Val_Score: {}".format(train_neg_lld, train_score,
        #                                              val_neg_lld, val_score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    print("Train_NLLK: {} \t Train_Score: {} \t"
          "Val_NLLK: {} \t Val_Score: {}".format(train_neg_lld, train_score,
                                                 val_neg_lld, val_score))
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_acc_lst, train_nllk_lst, \
        val_acc_lst, val_nllk_lst


def irt3(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(542) + 0.5
    gamma = np.zeros(542) + 0.5
    beta = np.zeros(1774) + 0.5
    alpha = np.zeros(1774) + 0.5

    train_acc_lst = []
    train_nllk_lst = []
    val_acc_lst = []
    val_nllk_lst = []

    train_neg_lld = 0
    train_score = 0
    val_neg_lld = 0
    val_score = 0
    for i in range(iterations):
        train_neg_lld = neg_log_likelihood3(data,
                                            theta=theta, gamma=gamma,
                                            beta=beta, alpha=alpha)
        train_score = evaluate3(data,
                                theta=theta, gamma=gamma,
                                beta=beta, alpha=alpha)
        train_acc_lst.append(train_score)
        train_nllk_lst.append(train_neg_lld)

        val_neg_lld = neg_log_likelihood3(val_data,
                                          theta=theta, gamma=gamma,
                                          beta=beta, alpha=alpha)
        val_score = evaluate3(val_data,
                              theta=theta, gamma=gamma,
                              beta=beta, alpha=alpha)
        val_acc_lst.append(val_score)
        val_nllk_lst.append(val_neg_lld)

        # print("Train_NLLK: {} \t Train_Score: {} \t"
        #       "Val_NLLK: {} \t Val_Score: {}".format(train_neg_lld, train_score,
        #                                              val_neg_lld, val_score))
        theta, gamma, beta, alpha = update_theta_beta3(data, lr,
                                                       theta, gamma,
                                                       beta, alpha)

    print("Train_NLLK: {} \t Train_Score: {} \t"
          "Val_NLLK: {} \t Val_Score: {}".format(train_neg_lld, train_score,
                                                 val_neg_lld, val_score))
    # TODO: You may change the return values to achieve what you want.
    return theta, gamma, beta, alpha, train_acc_lst, train_nllk_lst, \
        val_acc_lst, val_nllk_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def evaluate3(data, theta, gamma, beta, alpha):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param gamma: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (alpha[q] * (theta[u] - beta[q])).sum()
        p_a = gamma[u] + (1 - gamma[u]) * sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
        / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # 3-PL IRT
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr_lst = [0.1, 0.05, 0.01, 0.005, 0.001]
    it_lst = [10, 15, 25, 40, 60]

    # Tune lr:
    # for lr in lr_lst:
    #     print("Learning Rate: {}".format(lr))
    #     irt3(train_data, val_data, lr, 25)

    # # Tune # iteration:
    # for it in it_lst:
    #     print("# iterations: {}".format(it))
    #     irt3(train_data, val_data, 0.005, it)

    chosen_lr = 0.005
    chosen_it = 25
    theta, gamma, beta, alpha, \
    train_acc_lst, train_nllk_lst, \
    val_acc_lst, val_nllk_lst = \
        irt3(train_data, val_data, chosen_lr, chosen_it)

    # chosen_lr_b = 0.01
    # chosen_it_b = 15
    # theta, beta, \
    # train_acc_lst_b, train_nllk_lst_b, \
    # val_acc_lst_b, val_nllk_lst_b = \
    #     irt(train_data, val_data, chosen_lr_b, chosen_it_b)

    # plt.plot(train_nllk_lst, label="3-PL Train")
    # plt.plot(train_nllk_lst_b, label="base Train")
    # plt.plot(val_nllk_lst, label="3-PL Val")
    # plt.plot(val_nllk_lst_b, label="base Val")
    # plt.xlabel("iteration")
    # plt.ylabel("(negative) log-likelihood")
    # plt.title("Training&Validation Negative "
    #           "log-likelihoods Against Iteration for two model")
    # plt.legend()
    # plt.show()
    #
    # valid_acc = evaluate3(val_data, theta, gamma, beta, alpha)
    # test_acc = evaluate3(test_data, theta, gamma, beta, alpha)
    # print("Final Validation Acc: {} Test Acc: {}".format(valid_acc, test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    questions = [100, 700, 1300]
    theta.sort()
    for question in questions:
        plt.plot(theta,
                 sigmoid(alpha[question] * (theta - beta[question])),
                 label="Question {}".format(question))
    plt.xlabel("Theta")
    plt.ylabel("Probability")
    plt.title("Probability vs Theta")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
