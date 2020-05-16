from keras.layers import Concatenate, Average
from keras.regularizers import l2
from keras.layers.core import *
from keras.layers.recurrent import GRU
from keras.models import *
from u_loss import u_loss, evaluate_preds
from data_process import load_data2
from utils import attention_3d_block, get_manual_features, multi_trj_attention, get_week_data, calculate_auc
import time
import warnings

warnings.filterwarnings("ignore")

timesteps = 12
data_dim1 = 32
data_dim2 = 1
gamma = 0.5

# GRU model
start_iter = time.time()
file_iter = "train_records/"

inputs = Input(shape=(timesteps, data_dim1,))
inputs2 = Input(shape=(timesteps, data_dim2,))
concat1 = Concatenate()([inputs, inputs2])
hidden_units = 128
gru_out = GRU(hidden_units, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, name="gru1")(concat1)
attention_mul = attention_3d_block(timesteps, gru_out)
ReduceSum = Lambda(lambda z: K.sum(z, axis=2), name="reduce_sum")(attention_mul)
dense_a = Dense(hidden_units * 2, kernel_regularizer=l2(0.005), name="dense_a")(ReduceSum)
dense_b = Dense(hidden_units, kernel_regularizer=l2(0.005), name="dense_b")(dense_a)
output = Dense(1, activation='sigmoid')(dense_b)
model = Model(inputs=[inputs, inputs2], outputs=[output, dense_b])
model.compile(optimizer='adam', loss=['binary_crossentropy', u_loss], loss_weights=[1, gamma], metrics=['accuracy'])
print(model.summary())

# load data(region embedding & region credit score)
trjs_train, trjs_test, y_train, y_test, u_train, u_test = load_data2(timesteps, data_dim1, "gcn_embed")
trjs_train2, trjs_test2, y_train2, y_test2, u_train2, u_test2 = load_data2(timesteps, data_dim2, "credit")

# train model
f_record = open(file_iter + "results.txt", "w")
for iter_index in range(10):
    print("\niter = {}, spend = {}".format(iter_index, time.time() - start_iter))
    f_record.write("iteration = {}\n".format(iter_index))

    # Helper variables for main training loop
    wait = 0
    NB_EPOCH = 200
    best_val_loss = 99999
    es_patience = 20

    # Fit
    import time
    embed_train_last = np.random.rand(trjs_train.shape[0], hidden_units)
    embed_test_last = np.random.rand(trjs_test.shape[0], hidden_units)
    embed_train_now = None
    embed_test_now = None
    for epoch in range(1, NB_EPOCH + 1):
        # Log wall-clock time
        t = time.time()
        if epoch > 1:
            embed_train_last = embed_train_now
            embed_test_last = embed_test_now
        # Single training iteration (we mask nodes without labels for loss calculation)
        model.fit([trjs_train, trjs_train2], [y_train, embed_train_last], batch_size=64, epochs=1, verbose=0,
                  validation_data=([trjs_test, trjs_test2], [y_test, embed_test_last]))
        preds_train = model.predict([trjs_train, trjs_train2], batch_size=64)
        preds_test = model.predict([trjs_test, trjs_test2], batch_size=64)

        embed_train_now = preds_train[1]
        embed_test_now = preds_test[1]

        if epoch == 1: continue
        # Train / validation scores
        train_acc, train_loss = evaluate_preds(preds_train, y_train, u_train, gamma, embed_train_last, epoch)
        test_acc, test_loss = evaluate_preds(preds_test, y_test, u_test, gamma, embed_test_last, epoch)
        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_loss),
              "train_acc= {:.4f}".format(train_acc),
              "val_loss= {:.4f}".format(test_loss),
              "val_acc= {:.4f}".format(test_acc),
              "time= {:.4f}".format(time.time() - t),
              "wait= {:.4f}".format(wait))

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            wait = 0
        else:
            if wait >= es_patience:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1

    reg = 0.0005
    init = 'random_normal'
    fea_test, test_y, test_u = get_week_data(embed_test_now, u_test, y_test, hidden_units)
    fea_train, train_y, train_u = get_week_data(embed_train_now, u_train, y_train, hidden_units)

    manual_test = get_manual_features(test_u)
    manual_train = get_manual_features(train_u)
    manual_fea_col = manual_test.shape[1]
    m = multi_trj_attention(7, hidden_units+5, reg, init, manual_fea_col)
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m.fit([fea_train, manual_train], train_y, epochs=1000, batch_size=64,
          validation_data=([fea_test, manual_test], test_y), verbose=0)

    pred = m.predict([fea_test,manual_test], batch_size=64)
    pred = pred.T.tolist()[0]
    auc_score = calculate_auc(test_u, pred, test_y)
    f_record.write("auc\t{}\n".format(auc_score))
f_record.close()