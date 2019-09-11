from sklearn.metrics import roc_auc_score
import numpy as np
import math

def cal_auc(labels, preds):
    auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
    return auc

def get_ndcg(pos_test_file,neg100_test_file,topk):
    f_i1 = open(pos_test_file)
    f_i2 = open(neg100_test_file)
    pos_uin_item_score = {}
    neg_uin_item_score = {}
    all_scores = []
    real_y = []
    for line in f_i1:
        new_line = line.strip("\n").split("\t")
        uin = new_line[0]
        item = new_line[1]
        score = float(new_line[2])
        all_scores.append(score)
        real_y.append(1)
        if uin not in pos_uin_item_score:
            pos_uin_item_score[uin] = []
        pos_uin_item_score[uin].append([item,score])
    f_i1.close()

    for line in f_i2:
        new_line = line.strip("\n").split("\t")
        uin = new_line[0]
        item = new_line[1]
        score = float(new_line[2])
        all_scores.append(score)
        real_y.append(0)
        if uin not in neg_uin_item_score:
            neg_uin_item_score[uin] = []
        neg_uin_item_score[uin].append([item,score])
    f_i2.close()
    all_scores = np.array(all_scores)
    real_y = np.array(real_y)

    ndcg = []
    real_count = 0
    all_count = 0
    for uin in pos_uin_item_score:
        pos_score = pos_uin_item_score[uin][0][1]
        cur_neg_scores = neg_uin_item_score[uin]
        sorted_scores = sorted(cur_neg_scores, key = lambda e:e[1],reverse = True)
        scores_list = []
        k = 100
        i = 0
        cur_ndcg = 0
        for item_score in sorted_scores:
            score = item_score[1]
            if pos_score > score:
                k = i
                break
            i += 1
        if k < topk:
            cur_ndcg = math.log(2) / math.log(k+2)
            real_count += 1
        all_count += 1
        ndcg.append(cur_ndcg)
    ndcg_score = np.mean(ndcg)
    return ndcg_score

def get_idcg(truth, topk):
    truth_list = []
    idcg = 0.0
    for i in range(len(truth)):
        if truth[i] == 1:
            truth_list.append(1)
    if len(truth_list) <= topk:
        for i in range(len(truth_list)):
            idcg = idcg + math.log(2) / math.log(i + 2)
    else:
        for i in range(topk):
            idcg = idcg + math.log(2) / math.log(i + 2)
    if idcg == 0:
        idcg = 1.0
        print("zero error")
    return idcg


def cal_ndcg_float(truth, preds, topk):
    truth_pred = []
    for i in range(len(truth)):
        truth_pred.append([truth[i], preds[i]])
    truth_pred_sorted = sorted(truth_pred, key = lambda e:e[1],reverse = True)
    dcg = 0.0
    idcg = get_idcg(truth, topk)
    if len(truth) <= topk:
        for k in range(len(truth)):
            if truth_pred_sorted[k][0] == 1:
                dcg = dcg + math.log(2) / math.log(k + 2)
    else:
        for k in range(topk):
            if truth_pred_sorted[k][0] == 1:
                dcg = dcg + math.log(2) / math.log(k + 2)
    ndcg = dcg/idcg
    return ndcg
