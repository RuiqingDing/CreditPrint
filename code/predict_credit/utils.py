from keras.layers import multiply, Concatenate
from keras.layers.core import *
from keras.models import *
from keras import regularizers
from sklearn.metrics import roc_auc_score
import pandas as pd

file_root = "../data/"
SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(timesteps, inputs):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, timesteps))(a)
    a = Dense(timesteps, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def multi_trj_attention(timesteps, hidden_units, reg, init, manual_feas_col):
    inputs = Input(shape=(timesteps, hidden_units,))
    attention_mul = attention_3d_block(timesteps, inputs)
    ReduceSum = Lambda(lambda z: K.sum(z, axis=2), name="reduce_sum")(attention_mul)

    inputs2 = Input(shape=(manual_feas_col,))
    concat1 = Concatenate()([ReduceSum, inputs2])
    dense1 = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(reg), kernel_initializer=init)(concat1)
    dense2 = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(reg), kernel_initializer=init)(dense1)
    output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(reg), kernel_initializer=init)(dense2)
    model = Model(input=[inputs, inputs2], output=output)
    return model

# add context info
def get_week_data(data, uids, ys, hidden_units):
    f = open(file_root+"u_trj_day.txt", "r")
    u_trj_day = eval(f.read())
    f.close()

    f = open(file_root+"day_context.txt", "r")
    day_context = eval(f.read())
    f.close()
    df = pd.DataFrame(data)
    df["uid"] = 0
    df["uid"] = uids
    df["y"] = 0
    df["y"] = ys
    df["day"] = 0
    uid_y_pair = df.drop_duplicates(subset=["uid", "y"])
    uids2 = uid_y_pair['uid'].tolist()
    y2 = uid_y_pair['y'].tolist()
    cols = [i for i in range(hidden_units)]
    X = []
    for u in uids2:
        u_embed = []
        sub_df = df[df.uid == u]
        day = u_trj_day[u]
        sub_df["day"] = day
        grouped = sub_df.groupby(["uid", "y", "day"])[cols].mean()
        grouped = grouped.reset_index()
        for day in range(7):
            u_day = grouped[grouped.day == day][cols].values.tolist()[0]
            u_day.extend(day_context[day])
            u_embed.append(u_day)
        u_embed = np.array(u_embed)
        X.append(u_embed)
    X = np.array(X)
    return X, y2, uids2

def get_manual_features(uids):
    manual_feas = pd.read_csv(file_root + "manual_feas.csv")
    manual_feas = manual_feas.fillna(0)
    cols2 = ["date", "points_std", "covering_std", "num_of_points", "covering", "entropy", "turning_radius",
             "place_dif", "weekdays_weekends", "ratio_day", "ratio_night"]
    result = pd.DataFrame(columns=cols2)
    for u in uids:
        df = manual_feas[manual_feas.uid == u][cols2]
        result = result.append(df)
    for i in cols2:
        max_i = max(result[i])
        min_i = min(result[i])
        result[i] = (result[i] - min_i) / (max_i - min_i)
    manual_feas = result[cols2].values
    return manual_feas

# calculate auc and print
def calculate_auc(u_test, pred, y_test):
    import pandas as pd
    a = {"pred":pred, "y":y_test, "uid": u_test}
    result = pd.DataFrame(a)
    grouped = result.groupby("uid")["pred"].mean()
    grouped = grouped.reset_index()
    true_y = result[["uid", "y"]].drop_duplicates()
    mm = grouped.merge(true_y, on = "uid")
    pred_mean = mm["pred"].tolist()
    true_y = mm["y"].tolist()
    auc_score = roc_auc_score(true_y, pred_mean)
    print("AUC = {0}".format(auc_score))
    return auc_score


