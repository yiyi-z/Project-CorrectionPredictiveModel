# TODO: complete this file.
import random

from item_response import itr_predict, irt
from utils import *
import numpy as np

np.random.seed(0)


def generate_resamples(data, m):
    n = len(data["user_id"])
    resamples = []
    for i in range(m):
        indices = np.random.choice(n, n, replace=True)
        uid_lst = np.array(data["user_id"])[indices].tolist()
        qid_lst = np.array(data["question_id"])[indices].tolist()
        c_lst = np.array(data["is_correct"])[indices].tolist()
        resample = {"user_id": uid_lst,
                    "question_id": qid_lst,
                    "is_correct": c_lst}
        resamples.append(resample)

    return resamples


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    m = 3
    resamples = generate_resamples(train_data, m)
    # lrs = np.array([0.1, 0.05, 0.01, 0.005, 0.001])
    # its = np.array([10, 15, 25, 40, 60, 85])
    # rand_lrs = np.random.choice(lrs, m, replace=True)
    # rand_its = np.random.choice(its, m, replace=True)
    # hypers = []
    # for i in range(m):
    #     hypers.append((rand_lrs[i], rand_its[i]))
    # print(hypers)

    # preds_tr = []
    preds_val = []
    preds_te = []
    for i in range(m):
        # theta, beta, train_acc_lst, train_nllk_lst, val_acc_lst, val_nllk_lst \
        #     = irt(resamples[i], val_data, hypers[i][0], hypers[i][1])
        theta, beta, train_acc_lst, train_nllk_lst, val_acc_lst, val_nllk_lst \
            = irt(resamples[i], val_data, 0.01, 15)
        # preds_tr.append(itr_predict(train_data, theta, beta, binary=False))
        preds_val.append(itr_predict(val_data, theta, beta, binary=False))
        preds_te.append(itr_predict(test_data, theta, beta, binary=False))

    # pred_tr = sum(np.array(preds_tr)) / m
    pred_val = sum(np.array(preds_val)) / m
    pred_te = sum(np.array(preds_te)) / m

    # train_acc = evaluate(train_data, pred_tr)
    val_acc = evaluate(val_data, pred_val)
    test_acc = evaluate(test_data, pred_te)

    # print("training accuracy: " + str(train_acc))
    print("validation accuracy: " + str(val_acc))
    print("test accuracy: " + str(test_acc))


if __name__ == "__main__":
    main()
