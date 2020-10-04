# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 3:10 下午
# @Author  : jeffery
# @FileName: metric.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances


def acc_cosine(anchor_embedding, positive_embedding, negative_embedding):
    num_correct_cos_triplets = 0.
    num_triplets = 0.
    pos_cos_distance = paired_cosine_distances(anchor_embedding, positive_embedding)
    neg_cos_distances = paired_cosine_distances(anchor_embedding, negative_embedding)
    for idx in range(len(pos_cos_distance)):
        num_triplets += 1.
        if pos_cos_distance[idx] < neg_cos_distances[idx]:
            num_correct_cos_triplets += 1.
    return num_correct_cos_triplets / num_triplets


def acc_manhatten(anchor_embedding, positive_embedding, negative_embedding):
    num_correct_manhatten_triplets = 0.
    num_triplets = 0.
    # Manhatten
    pos_manhatten_distance = paired_manhattan_distances(anchor_embedding, positive_embedding)
    neg_manhatten_distances = paired_manhattan_distances(anchor_embedding, negative_embedding)

    for idx in range(len(pos_manhatten_distance)):
        num_triplets += 1.
        if pos_manhatten_distance[idx] < neg_manhatten_distances[idx]:
            num_correct_manhatten_triplets += 1.

    return num_correct_manhatten_triplets / num_triplets


def acc_euclidean(anchor_embedding, positive_embedding, negative_embedding):
    num_correct_euclidean_triplets = 0.
    num_triplets = 0.
    # Euclidean
    pos_euclidean_distance = paired_euclidean_distances(anchor_embedding, positive_embedding)
    neg_euclidean_distances = paired_euclidean_distances(anchor_embedding, negative_embedding)

    for idx in range(len(pos_euclidean_distance)):
        num_triplets += 1.
        if pos_euclidean_distance[idx] < neg_euclidean_distances[idx]:
            num_correct_euclidean_triplets += 1.
    return num_correct_euclidean_triplets / num_triplets


def cosine_scores(embeddings1, embeddings2):
    return 1 - (paired_cosine_distances(embeddings1, embeddings2))


def manhattan_distances(embeddings1, embeddings2):
    return -paired_manhattan_distances(embeddings1, embeddings2)


def euclidean_distances(embeddings1, embeddings2):
    return -paired_euclidean_distances(embeddings1, embeddings2)


def dot_product_distances(embeddings1, embeddings2):
    return [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]


def pearsonr_cosine(embeddings1, embeddings2, labels):
    cosine_s = cosine_scores(embeddings1, embeddings2)
    p, _ = pearsonr(labels, cosine_s)
    return p


def spearmanr_cosine(embeddings1, embeddings2, labels):
    s, _ = spearmanr(labels, cosine_scores(embeddings1, embeddings2))
    return s


def pearsonr_manhattan(embeddings1, embeddings2, labels):
    p, _ = pearsonr(labels, manhattan_distances(embeddings1, embeddings2))
    return p


def spearmanr_manhattan(embeddings1, embeddings2, labels):
    s, _ = spearmanr(labels, manhattan_distances(embeddings1, embeddings2))
    return s


def pearsonr_euclidean(embeddings1, embeddings2, labels):
    p, _ = pearsonr(labels, euclidean_distances(embeddings1, embeddings2))
    return p


def spearmanr_euclidean(embeddings1, embeddings2, labels):
    s, _ = spearmanr(labels, euclidean_distances(embeddings1, embeddings2))
    return s


def pearsonr_dot(embeddings1, embeddings2, labels):
    p, _ = pearsonr(labels, dot_product_distances(embeddings1, embeddings2))
    return p


def spearmanr_dot(embeddings1, embeddings2, labels):
    s, _ = spearmanr(labels, dot_product_distances(embeddings1, embeddings2))
    return s


def binary_accuracy(pred, label):
    pred, label = convert_2_numpy(pred, label)
    return accuracy_score(label, pred)


def convert_2_numpy(pred, label):
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    return pred, label


# def sk_auc(pred,label):
#     roc_curve
#     return auc()

def sk_micro_precision(pred, label):
    return precision_score(label, pred, average='micro')


def sk_micro_recall(pred, label):
    return recall_score(label, pred, average='micro')


def sk_micro_f1(pred, label):
    return f1_score(label, pred, average='micro')


def sk_macro_precision(pred, label):
    return precision_score(label, pred, average='macro')


def sk_macro_recall(pred, label):
    return recall_score(label, pred, average='macro')


def sk_macro_f1(pred, label):
    return f1_score(label, pred, average='macro')


def sk_accuracy(pred, label):
    return accuracy_score(label, pred)


def sk_macro_accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    acc = correct / len(target)
    return acc


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def union_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)


def intersect_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)


#########################################################################
# MACRO METRICS: calculate metric for each label and average across labels
#########################################################################

def macro_accuracy(yhat, y):
    # y_sum = np.sum(y)
    # y_hat_sum = np.sum(yhat)
    # a = intersect_size(yhat, y, 0)
    # b = (union_size(yhat, y, 0) + 1e-10)
    # a_sum = np.sum(a)
    # b_sum = np.sum(b)
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    # num_mean = np.mean(num)
    return np.mean(num)


def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(yhat, y):
    # a = intersect_size(yhat, y, 0)
    # b = (y.sum(axis=0) + 1e-10)
    # a_sum = np.sum(a)
    # b_sum = np.sum(b)
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    # num_mean = np.mean(num)
    return np.mean(num)


def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = (2 * (prec * rec)) / (prec + rec)
    return f1


##########################################################################
# MICRO METRICS: treat every prediction as an individual binary prediction
##########################################################################

def micro_accuracy(yhatmic, ymic):
    # micro
    yhatmic = yhatmic.ravel()
    ymic = ymic.ravel()

    xx = intersect_size(yhatmic, ymic, 0) / union_size(yhatmic, ymic, 0)
    return xx


def micro_precision(yhatmic, ymic):
    yhatmic = yhatmic.ravel()
    ymic = ymic.ravel()
    return intersect_size(yhatmic, ymic, 0) / (yhatmic.sum(axis=0) + 1e-6)


def micro_recall(yhatmic, ymic):
    yhatmic = yhatmic.ravel()
    ymic = ymic.ravel()
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)


def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def recall_at_k(yhat_raw, y, k=8):
    # num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:, ::-1]
    topk = sortd[:, :k]

    # get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i, tk].sum()
        denom = y[i, :].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)


def precision_at_k(yhat_raw, y, k=8):
    # num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:, ::-1]
    topk = sortd[:, :k]

    # get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i, tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)


def f1_at_k(yhat_raw, y, k=8):
    prec_at_k = precision_at_k(yhat_raw, y, k)
    rec_at_k = recall_at_k(yhat_raw, y, k)
    return 2 * (prec_at_k * rec_at_k) / (prec_at_k + rec_at_k + 1e-6)


# ------------------------------------  8    ---------------------------------------------
def precision_at_8(yhat_raw, y, k=8):
    # num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:, ::-1]
    topk = sortd[:, :k]

    # get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i, tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)


def recall_at_8(yhat_raw, y, k=8):
    # num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:, ::-1]
    topk = sortd[:, :k]

    # get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i, tk].sum()
        denom = y[i, :].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)


def f1_at_8(yhat_raw, y, k=8):
    prec_at_k = precision_at_k(yhat_raw, y, k)
    rec_at_k = recall_at_k(yhat_raw, y, k)
    return 2 * (prec_at_k * rec_at_k) / (prec_at_k + rec_at_k + 1e-6)


# ------------------------------------  15    ---------------------------------------------
def precision_at_15(yhat_raw, y, k=15):
    # num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:, ::-1]
    topk = sortd[:, :k]

    # get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i, tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)


def recall_at_15(yhat_raw, y, k=15):
    # num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:, ::-1]
    topk = sortd[:, :k]

    # get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i, tk].sum()
        denom = y[i, :].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)


def f1_at_15(yhat_raw, y, k=15):
    prec_at_k = precision_at_k(yhat_raw, y, k)
    rec_at_k = recall_at_k(yhat_raw, y, k)
    return 2 * (prec_at_k * rec_at_k) / (prec_at_k + rec_at_k + 1e-6)


def macro_auc(yhat_raw, y):
    if yhat_raw.shape[0] <= 1:
        return
    fpr = {}
    tpr = {}
    roc_auc = {}
    # get AUC for each label individually
    relevant_labels = []
    auc_labels = {}
    for i in range(y.shape[1]):
        # only if there are true positives for this label
        if y[:, i].sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(y[:, i], yhat_raw[:, i])
            if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                auc_score = auc(fpr[i], tpr[i])
                if not np.isnan(auc_score):
                    auc_labels["auc_%d" % i] = auc_score
                    relevant_labels.append(i)

    # macro-AUC: just average the auc scores
    aucs = []
    for i in relevant_labels:
        aucs.append(auc_labels['auc_%d' % i])

    return np.mean(aucs)


def micro_auc(yhat_raw, y):
    ymic = y.ravel()
    yhatmic = yhat_raw.ravel()
    fpr, tpr, _ = roc_curve(ymic, yhatmic)
    return auc(fpr, tpr)
