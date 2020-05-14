from __future__ import print_function
import scipy.sparse as sp
import numpy as np
import random
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence

data_file = "../data/"

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def get_splits(y, idx_train, idx_val, idx_test):
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask

def loss_RE(X_embedding, X_score, idx_split,  sample_num):
    import random
    delta = 0.01
    losses = []
    for i in idx_split:
        credit = X_score[i]
        loss = 0
        if credit > 0.39:
            range_down = max(credit - delta, 0.39)
            range_up = credit + delta
            pos_candidates = np.argwhere((X_score > range_down) & (X_score < range_up)).T
            if pos_candidates.shape[1] > sample_num:
                pos_candidates = random.sample(list(pos_candidates[0]), sample_num)
            else:
                pos_candidates = list(pos_candidates[0])

            neg_candidates = np.argwhere(X_score < range_down).T
            if neg_candidates.shape[1] > sample_num:
                neg_candidates = random.sample(list(neg_candidates[0]), sample_num)
            else:
                neg_candidates = list(neg_candidates[0])
        else:
            range_down = credit - delta
            range_up = max(0.39, credit + delta)
            pos_candidates = np.argwhere((X_score > range_down) & (X_score < range_up)).T
            if pos_candidates.shape[1] > sample_num:
                pos_candidates = random.sample(list(pos_candidates[0]), sample_num)

            neg_candidates = np.argwhere(X_score > range_down).T
            if neg_candidates.shape[1] > sample_num:
                neg_candidates = random.sample(list(neg_candidates[0]), sample_num)

        for pos_idx in pos_candidates:
            dot = np.dot(X_embedding[i], X_embedding[pos_idx])
            if dot > np.log(np.finfo(type(dot)).max):
                loss += 0
            else:
                loss += 1.0/(1+np.exp(dot))
        for neg_idx in neg_candidates:
            dot = np.dot(X_embedding[i], X_embedding[neg_idx])
            if dot > np.log(np.finfo(type(dot)).max):
                loss += 0
            else:
                loss += 1.0 / (1 + np.exp(dot))
        losses.append(loss/sample_num)
    return np.mean(losses)

def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))

def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))

def evaluate_preds(preds, gamma, sample_num, labels, indices):
    split_loss = list()
    split_acc = list()
    f = open(data_file+"grid_score.txt", "r", encoding="utf-8")#grid credit score
    X_score = np.array(eval(f.read()))
    f.close()
    for y_split, idx_split in zip(labels, indices):
        los = loss_RE(preds[1], X_score, idx_split, sample_num)
        los2 = categorical_crossentropy(preds[0][idx_split], y_split[idx_split])
        split_loss.append(los+gamma*los2)
        acc = accuracy(preds[0][idx_split], y_split[idx_split])
        split_acc.append(acc)
    return split_loss, split_acc

def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian

def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian

def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))
    return T_k

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape