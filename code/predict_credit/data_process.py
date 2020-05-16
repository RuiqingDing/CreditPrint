#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/7 9:43
# @Author  : Ruiqing Ding
# @Contact : RuiqingDing@outlook.com
# @File    : data_process.py
# @Software: PyCharm
import time
import numpy as np

file_root = "../data/"
def convert_to_dict(filename):
    f = open(filename, "r")
    text = f.read()
    dict_text = eval(text)
    f.close()
    return dict_text

def load_data(timesteps, data_dim, fea_name):
    print("------------load data---------------")
    f1 = file_root+"dict_y.txt"
    f2 = file_root+"dict_user_trj.txt"
    if (data_dim == 1) & (fea_name == "credit"):
        f3 = file_root+"grid_ratio.txt"
    elif (data_dim == 32) & (fea_name == "gcn_embed"):
        f3 = file_root+"dict_region_embedding_32_sample_10_loss_0.5.txt"
    else:
        print("load data wrong!")

    dict_y = convert_to_dict(f1)
    dict_trj = convert_to_dict(f2)
    dict_grid_info = convert_to_dict(f3)

    X = []
    for day in dict_trj:
        start = time.time()
        day_info = []
        for u in dict_y:
            locations = dict_trj[day][u]
            u_trj = np.zeros((timesteps, data_dim))
            if locations == [0 for t in range(48)]:
                day_info.append(u_trj)
                continue
            loc_arr = np.array(locations)
            index_not_zero_list = list(np.flatnonzero(loc_arr))
            for i in range(timesteps):
                segment = int(48 / timesteps)
                if len(set(range(segment*i, segment*(i+1))).intersection(set(index_not_zero_list))) > 0: #have records
                    grids = locations[segment*i: segment*(i+1)]
                    grids2 = [g for g in grids if g != 0]
                    grids_info = np.zeros((len(grids2), data_dim))
                    for j in range(len(grids2)):
                        try:
                            grids_info[j] = dict_grid_info[grids2[j]]
                        except Exception:
                            print(grids2[j], dict_grid_info[grids2[j]])
                    mean_grid_info = list(grids_info.mean(axis=0))
                    u_trj[i] = mean_grid_info
                else:
                    # if user has no records in this period, then replace with the location in the closest period
                    loc_dif = 48
                    sign = 0
                    for a in range(segment*i, segment*(i+1)):
                        for b in  index_not_zero_list:
                            if abs(a-b) < loc_dif:
                                loc_dif=abs(a-b)
                                sign = b
                    u_trj[i] = dict_grid_info[locations[sign]]
            day_info.append(u_trj)
        day_info = np.array(day_info)
        X.append(day_info)
        print("day: ", day, time.time() - start)
    X = np.array(X)
    y = list(dict_y.values())
    return X, y


def load_data2(timesteps, data_dim, fea_name):
    start = time.time()

    f1 = file_root+"dict_y.txt"
    dict_y = convert_to_dict(f1)
    dict_day = convert_to_dict(file_root+"dict_day.txt")
    X, y = load_data(timesteps, data_dim, fea_name)

    uids = list(dict_y.keys())
    import random
    random.seed(0)
    uids_train = random.sample(uids, int(len(uids) * 0.8))
    uids_train_index = [int(uids.index(u)) for u in uids_train]
    uids_test_index = [i for i in range(len(uids)) if i not in uids_train_index]
      
    y_train = []
    u_train = []
    y_test = []
    u_test = []
    trjs_train = []
    trjs_test = []

    u_trj_day = {}
    for i in uids_train_index:
        u_trj_day[uids[i]] = []
        for d in range(32):
            trj = X[d][i]
            if trj.any == np.zeros(shape=(timesteps, data_dim)).any: continue
            trjs_train.append(trj)
            u_train.append(uids[i])
            y_train.append(dict_y[uids[i]])
            u_trj_day[uids[i]].append(dict_day[d])

    for i in uids_test_index:
        u_trj_day[uids[i]] = []
        for d in range(32):
            trj = X[d][i]
            if trj.any() == np.zeros(shape=(timesteps, data_dim)).any(): continue
            trjs_test.append(trj)
            u_test.append(uids[i])
            y_test.append(dict_y[uids[i]])
            u_trj_day[uids[i]].append(dict_day[d])

    trjs_train = np.array(trjs_train)
    trjs_test = np.array(trjs_test)
    print("load data, time = {0}".format(time.time() - start))

    f = open(file_root+"u_trj_day.txt", "w")
    f.write(str(u_trj_day))
    f.close()
    return trjs_train, trjs_test, y_train, y_test, u_train, u_test