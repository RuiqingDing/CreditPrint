from __future__ import division
import pandas as pd
import networkx as nx
from loss import embed_loss
import keras.backend as K
from keras.layers import Input, Dropout, Average, Dense, Concatenate, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical

from graph import GraphConvolution
from utils import *
from graph_attention_layer import GraphAttention

data_file = "../data/"
def convert_to_dict(filename):
    f = open(filename, "r")
    text = f.read()
    dict_text = eval(text)
    f.close()
    return dict_text

def read_graph():
    '''preprocessing data'''
    G_inter = nx.read_gml(data_file+"inter_graph.gml")
    G_dis = nx.read_gml(data_file+"grid_graph.gml")
    G_cor = nx.read_gml(data_file+"correlation_graph.gml")
    order = sorted(list(G_inter.nodes()))

    data = pd.read_excel(data_file+"grid_info.xlsx")
    data["grid"] = data["grid"].astype("str")
    grid_dict = dict()
    for index, row in data.iterrows():
        grid = row["grid"]
        grid_dict[grid] = row["default_ratio"]

    data = data[data.grid.isin(order)].sort_values(by="grid", ascending=True)
    labels = data["default_ratio"].tolist()

    # replace label value to 0 and 1
    y = []
    median_value = np.median(labels)
    for i in labels:
        if i > median_value:
            y.append(1)
        else:
            y.append(0)
    y = to_categorical(y)

    adj_inter = nx.to_numpy_matrix(G_inter, nodelist=order)
    adj_dis = nx.to_numpy_matrix(G_dis, nodelist=order)
    adj_cor = nx.to_numpy_matrix(G_cor, nodelist=order)

    # match grid with its index
    idx_map = {i: j for i, j in enumerate(order)}
    idx_map2 = {j: i for i, j in enumerate(order)}
    return adj_inter, adj_dis, adj_cor, y, grid_dict, G_inter, G_dis, G_cor, idx_map, idx_map2

def preprocess(adj, Gc):
    # Get data
    X = sp.csr_matrix(np.eye(Gc.number_of_nodes()))
    X = X.todense()
    adj = sp.coo_matrix(adj)
    A = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    X /= X.sum(1).reshape(-1, 1)
    A = preprocess_adj(A, True)
    return X, A

def get_splits(y, idx_train, idx_val, idx_test):
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])

    idx_train2, idx_val2, idx_test2 = [], [], []
    for i in range(len(y)):
        idx_train2.append(True)
        idx_val2.append(True)
        idx_test2.append(True)

    idx_test2 = np.array(idx_test2)
    idx_train2 = np.array(idx_train2)
    idx_val2 = np.array(idx_val2)
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, idx_train2, idx_val2, idx_test2, train_mask


def get_model(X, vector_dim):
    # GCN
    # Parameters
    dropout_rate = 0.5
    support = 1

    X_in = Input(shape=(X.shape[0],))
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
    dropout1 = Dropout(dropout_rate)(X_in)
    graph_convolution_1 = GraphConvolution(vector_dim, support, activation='elu', kernel_regularizer=l2(5e-4))([dropout1] + G)
    dropout2 = Dropout(dropout_rate)(graph_convolution_1)
    graph_convolution_2 = GraphConvolution(vector_dim, support, activation='elu', kernel_regularizer=l2(5e-4))([dropout2] + G)
    model = Model(inputs=[X_in] + G, outputs=graph_convolution_2)
    return model

def gat_merge_model(adj_dis, adj_inter, adj_cor, Gc_cor, Gc_dis, Gc_inter, labels, vector_dim):
    # add GAL
    # Parameters
    N = 3  # Number of nodes in the graph
    F = vector_dim  # Original feature dimension
    F_ = vector_dim  # Output size of first GraphAttention layer
    n_attn_heads = 1  # Number of attention heads in first GAT layer
    dropout_rate = 0.5  # Dropout rate (between and inside GAT layers)
    l2_reg = 5e-4 / 2  # Factor for l2 regularization

    X1, A1 = preprocess(adj_dis, Gc_dis)
    X2, A2 = preprocess(adj_inter, Gc_inter)
    X3, A3 = preprocess(adj_cor, Gc_cor)

    model1 = get_model(X1, vector_dim)
    model2 = get_model(X2, vector_dim)
    model3 = get_model(X3, vector_dim)

    def transpose(x):
        return K.transpose(x)

    def arrange(r, vector_dim, nodes_num):
        input_list = []
        for i in range(nodes_num):
            input_list.append(K.reshape(r[0][:, i:(i + 1)], [vector_dim, 1]))
            input_list.append(K.reshape(r[1][:, i:(i + 1)], [vector_dim, 1]))
            input_list.append(K.reshape(r[2][:, i:(i + 1)], [vector_dim, 1]))
        X = K.concatenate(input_list)
        X = K.transpose(X)
        return X

    def slice(r, nodes_num):
        r_list = []
        for i in range(nodes_num):
            r_list.append(r[:, 3 * i:(3 * i + 1)])
        gat1 = K.concatenate(r_list)
        gat1 = K.transpose(gat1)
        return gat1

    def slice2(r, nodes_num):
        r_list = []
        for i in range(nodes_num):
            r_list.append(r[:, (3 * i + 1):(3 * i + 2)])
        gat2 = K.concatenate(r_list)
        gat2 = K.transpose(gat2)
        return gat2

    def slice3(r, nodes_num):
        r_list = []
        for i in range(nodes_num):
            r_list.append(r[:, (3 * i + 2):(3 * (i + 1))])
        gat3 = K.concatenate(r_list)
        gat3 = K.transpose(gat3)
        return gat3

    def kron(A_in):
        # get block matrix
        a = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        A = np.eye(K.int_shape(A_in)[1])
        A = np.kron(A, a)
        A = A + np.eye(A.shape[0])
        A = K.variable(A)
        return A

    inp1 = model1.input
    inp2 = model2.input
    inp3 = model3.input
    r1 = model1.output
    r2 = model2.output
    r3 = model3.output
    r1 = Lambda(transpose, name="gcn_output1_transpose")(r1)
    r2 = Lambda(transpose, name="gcn_output2_transpose")(r2)
    r3 = Lambda(transpose, name="gcn_output3_transpose")(r3)
    X = Lambda(arrange, arguments={"vector_dim": vector_dim, "nodes_num": len(labels)}, name="rearrange_gcn_output")(
        [r1, r2, r3])
    A_in = Input(shape=(len(labels),))
    A_kron = Lambda(kron, name="block_matrix")(A_in)
    graph_attention_1 = GraphAttention(F_,
                                       attn_heads=n_attn_heads,
                                       attn_heads_reduction='average',
                                       dropout_rate=dropout_rate,
                                       activation='elu',
                                       kernel_regularizer=l2(l2_reg),
                                       attn_kernel_regularizer=l2(l2_reg))([X, A_kron])
    graph_attention_1_trans = Lambda(transpose, name="gal_transpose")(graph_attention_1)
    gat1 = Lambda(slice, arguments={"nodes_num": len(labels)}, name="slice_gal1")(graph_attention_1_trans)
    gat2 = Lambda(slice2, arguments={"nodes_num": len(labels)}, name="slice_gal2")(graph_attention_1_trans)
    gat3 = Lambda(slice3, arguments={"nodes_num": len(labels)}, name="slice_gal3")(graph_attention_1_trans)
    average1 = Average(name="average_gal")([gat1, gat2, gat3])
    dense1 = Dense(vector_dim, activation='elu', use_bias=True, kernel_regularizer=l2(0.01), input_dim=vector_dim,
                   name="dense")(average1)
    Y = Dense(2, activation="softmax", name="output")(dense1)
    model = Model(inputs=inp1 + inp2 + inp3 + [A_in], outputs=[Y, average1])
    return model

if __name__ == '__main__':
    gamma = 0.5 #loss percent
    sample_num = 10
    vector_dim = 32  # embedding dimension
    grid_num = 1865

    idx_train_val = range(grid_num)
    idx_train = range(grid_num)
    idx_val = range(grid_num)
    idx_test = range(grid_num)

    # prepare data
    adj_inter, adj_dis, adj_cor, labels, grid_dict, G_inter, G_dis, G_cor, idx_map, idx_map2 = read_graph()

    # split label
    y_train, y_val, y_test, idx_train, idx_val, idx_test, idx_train2, idx_val2, idx_test2, train_mask = get_splits(
        labels, idx_train, idx_val, idx_test)

    # build model
    model = gat_merge_model(adj_dis, adj_inter, adj_cor, G_cor, G_dis, G_inter, labels, vector_dim)

    model.compile(optimizer=Adam(),
                  loss=['categorical_crossentropy', embed_loss],
                  metrics= ["accuracy"],
                  loss_weights=[1,gamma])
    model.summary()

    X1, A1 = preprocess(adj_dis, G_dis)
    X2, A2 = preprocess(adj_inter, G_inter)
    X3, A3 = preprocess(adj_cor, G_cor)

    # create partitioned matrix
    A_gat = np.eye(len(labels))

    # Parameters
    epochs = 10000  # Number of training epochs
    es_patience = 20  # Patience fot early stopping
    wait = 0
    best_val_loss = 99999

    # initialize embedding
    embed = np.random.rand(len(labels), vector_dim)
    # embed = data_file+"dict_gcn_correlation_dense_32_class3.txt"
    dict_dist = convert_to_dict(embed)
    embed = []
    for grid in dict_dist:
        embed.append(dict_dist[grid])
    embed = np.array(embed)

    preds = None
    for epoch in range(1, epochs + 1):
        import time

        # Log wall-clock time
        t = time.time()

        # Single training iteration (we mask nodes without labels for loss calculation)
        model.fit([X1, A1] + [X2, A2] + [X3, A3] + [A_gat], [y_train, embed],
                  batch_size=len(labels), epochs=1, shuffle=False, verbose=0)

        preds = model.predict([X1, A1] + [X2, A2] + [X3, A3] + [A_gat], batch_size=len(labels))
        embed = preds[1]
        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, gamma, sample_num, [y_train, y_val], [idx_train, idx_val])

        if epoch % 2 == 0:
            print("Epoch: {:04d}".format(epoch),
                  "train_loss= {:.4f}".format(train_val_loss[0]),
                  "train_acc= {:.4f}".format(train_val_acc[0]),
                  "val_loss= {:.4f}".format(train_val_loss[1]),
                  "val_acc= {:.4f}".format(train_val_acc[1]),
                  "time= {:.4f}".format(time.time() - t))

        if train_val_loss[1] < best_val_loss:
            best_val_loss = train_val_loss[1]
            wait = 0
        else:
            if wait >= epochs:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1

    test_loss, test_acc = evaluate_preds(preds, gamma, sample_num, [y_test], [idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(test_loss[0]),
          "acc= {:.4f}".format(test_acc[0]))

    grid_embedding = dict()
    order = sorted(list(G_cor.nodes()))
    for i in range(len(order)):
        grid = int(order[i])
        grid_embedding[grid] = list(preds[1][i])
    f = open("dict_region_embedding_{0}_sample_{1}_loss_{2}.txt".format(vector_dim, sample_num, gamma), "w")
    f.write(str(grid_embedding))
    f.close()