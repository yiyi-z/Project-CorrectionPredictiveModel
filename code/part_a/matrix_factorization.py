from utils import *
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    common_term = -(c - (u[n].T @ z[q]))

    u[n] -= lr * common_term * z[q]
    z[q] -= lr * common_term * u[n]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, val_data, k, lr, num_iteration, trace):
    """ Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: Same as above, but is validation dataset
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param trace: 0--not trace, 1--trace
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    train_sel_lst = []
    val_sel_lst = []
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        if trace and (i % 10000 == 0):
            train_sel_lst.append(squared_error_loss(train_data, u, z))
            val_sel_lst.append(squared_error_loss(val_data, u, z))
    mat = u @ z.T
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, train_sel_lst, val_sel_lst


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # # ks = [1, 2, 4, 8, 16]
    # # for k in ks:
    # #     reconstructed_matrix = svd_reconstruct(train_matrix, k)
    # #     acc = sparse_matrix_evaluate(val_data, reconstructed_matrix)
    # #     print("k: {}\tacc: {}".format(k, acc))
    # chosen_k = 8
    # rm = svd_reconstruct(train_matrix, chosen_k)
    # print("final val acc: {}".format(sparse_matrix_evaluate(val_data, rm)))
    # print("final test acc: {}".format(sparse_matrix_evaluate(test_data, rm)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # Tune HyperParameters!
    # lr_lst = [0.1, 0.05, 0.01, 0.005, 0.001]
    # it_lst = [int(5e4), int(1e5), int(5e5), int(1e6), int(5e6)]
    k_lst = [1, 2, 4, 8, 16]
    # mat = als(train_data, val_data, 8, 0.01, 5000, 0)[0]
    # print(mat.shape)
    # for lr in lr_lst:
    #     rm = als(train_data, val_data, 4, lr, int(5e5), 0)[0]
    #     acc = sparse_matrix_evaluate(val_data, rm)
    #     print("lr: {}\tacc: {}".format(lr, acc))
    #
    # for it in it_lst:
    #     rm = als(train_data, val_data, 4, 0.01, it, 0)[0]
    #     acc = sparse_matrix_evaluate(val_data, rm)
    #     print("# iterations: {}\tacc: {}".format(it, acc))

    chosen_lr = 0.01
    chosen_it = int(5e5)
    for k in k_lst:
        rm = als(train_data, val_data, k, chosen_lr, chosen_it, 0)[0]
        acc = sparse_matrix_evaluate(val_data, rm)
        print("k: {}\tacc: {}".format(k, acc))
    chosen_k = 4

    # rm, train_sel_lst, val_sel_lst = als(train_data, val_data,
    #                                      chosen_k, chosen_lr, chosen_it, 1)
    #
    # plt.plot(train_sel_lst, color="blue", label="train")
    # plt.plot(val_sel_lst, color="red", label="val")
    # plt.xlabel("iteration (1e4)")
    # plt.ylabel("Square Error Loss")
    # plt.title("Training&Validation Square Error Loss Against Iteration")
    # plt.legend()
    # plt.show()
    # print("final val acc: {}".format(sparse_matrix_evaluate(val_data, rm)))
    # print("final test acc: {}".format(sparse_matrix_evaluate(test_data, rm)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
