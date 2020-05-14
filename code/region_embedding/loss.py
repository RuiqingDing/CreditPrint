#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/2 17:50
# @Author  : Ruiqing Ding
# @Contact : RuiqingDing@outlook.com
# @File    : loss2.py
# @Software: PyCharm
import numpy as np
import keras.backend as K
import tensorflow as tf

data_file = "../data/"

def embed_loss(y_pred, y_true):
    tensor_a = tf.convert_to_tensor(0.0)
    return tensor_a

def embed_loss2(embed_pred, embed_y):
    import random
    sample_num = 10
    delta = 0.01
    grid_num = 1865
    sample_idx = range(grid_num)

    f = open(data_file+"grid_score.txt", "r")#grid credit score
    X_score = np.array(eval(f.read()))
    f.close()

    def calculate_loss(sample_idx, sample_num, embed):
        losses = []
        for i in sample_idx:
            loss = 0
            x = K.reshape(embed[i], (32,1))
            x = K.transpose(x)

            credit = X_score[i]
            if credit > 0.39:
                range_down = max(credit - delta, 0.39)
                range_up = credit + delta
                pos_candidates = np.argwhere((X_score > range_down) & (X_score < range_up))
                pos_candidates = pos_candidates.T

                while pos_candidates.shape[1] == 0:
                    range_down = max(credit - delta - 0.01, 0.39)
                    range_up = credit + delta + 0.01
                    pos_candidates = np.argwhere((X_score > range_down) & (X_score < range_up))
                    pos_candidates = pos_candidates.T
                if pos_candidates.shape[1] > sample_num:
                    pos_candidates = random.sample(list(pos_candidates[0]), sample_num)
                else:
                    pos_candidates = list(pos_candidates[0])

                neg_candidates = np.argwhere(X_score < range_down)
                neg_candidates = neg_candidates.T
                if neg_candidates.shape[1] > sample_num:
                    neg_candidates = random.sample(list(neg_candidates[0]), sample_num)
                else:
                    neg_candidates = list(neg_candidates[0])

            else:
                range_down = credit - delta
                range_up = max(0.39, credit + delta)
                pos_candidates = np.argwhere((X_score > range_down) & (X_score < range_up))
                pos_candidates = pos_candidates.T
                while pos_candidates.shape[1] == 0:
                    range_down = credit - delta - 0.01
                    range_up = max(0.39, credit + delta+0.01)
                    pos_candidates = np.argwhere((X_score > range_down) & (X_score < range_up))
                    pos_candidates = pos_candidates.T
                if pos_candidates.shape[1] > sample_num:
                    pos_candidates = random.sample(list(pos_candidates[0]), sample_num)
                else:
                    pos_candidates = list(pos_candidates[0])

                neg_candidates = np.argwhere(X_score > range_down)
                neg_candidates = neg_candidates.T
                if neg_candidates.shape[1] > sample_num:
                    neg_candidates = random.sample(list(neg_candidates[0]), sample_num)
                else:
                    neg_candidates = list(neg_candidates[0])

            # print(i, x.shape, pos_candidates)
            for pos_idx in pos_candidates:
                a = K.sum(np.dot(x, K.transpose(K.reshape(embed[pos_idx], (32,1)))))
                loss += K.log(1 / (1 + K.exp(a)))
            for neg_idx in neg_candidates:
                b = K.sum(np.dot(x, K.transpose(K.reshape(embed[neg_idx], (32,1)))))
                loss += 1 - K.log(1 / (1 + K.exp(b)))
            loss = K.reshape(loss , shape=[1])
            losses.append(loss)
        losses = tf.concat(losses, 0)
        return K.mean(losses)/sample_num

    loss_pred = calculate_loss(sample_idx, sample_num, embed_pred)
    # loss_y = calculate_loss(sample_idx,sample_num, embed_y)
    # return loss_y - loss_pred
    return loss_pred