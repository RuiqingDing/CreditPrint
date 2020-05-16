import numpy as np
import tensorflow as tf

def u_loss(y_pred, y_true):
    tensor_a = tf.convert_to_tensor(0.0)
    return tensor_a

def u_loss_evaluate(embed, embed_last, uids):
    # Calculate the similarity between the average user trajectory of the last epoch and the current trajectory embedding
    loss = 0
    uids_nodup = list(set(uids))
    for uid in uids_nodup:
        loss_u = 0
        idxs = np.argwhere(np.array(uids) == uid)
        sub_embed = embed[idxs]
        sub_embed_last = embed_last[idxs]
        sub_embed_last_avg = np.mean(sub_embed_last, axis=0)
        for i in range(sub_embed.shape[0]):
            dot1 = np.dot(sub_embed[i][0], sub_embed_last_avg[0])
            dot2 = np.dot(sub_embed[i][0], sub_embed[i][0])
            dot3 = np.dot(sub_embed_last_avg[0], sub_embed_last_avg[0])
            m = np.sqrt(dot2 * dot3)
            loss_u -= np.log(1/(1+np.exp(-dot1/m)))
        loss_u = loss_u/(sub_embed.shape[0])
        loss += loss_u
    return loss/len(uids_nodup)

def evaluate_preds(preds, labels, uids, gamma, embed_last, epoch):
    def accuracy(y_pred, y_true):
        count = 1
        for i in range(len(y_pred)):
            if y_pred[i] > 0.5:
                u = 1
            else:
                u = 0
            if u == y_true[i]:
                count += 1
        return count/len(y_pred)
    acc = accuracy(preds[0].T.tolist()[0], labels)

    def binary_crossentropy(y_pred, y_true):
        loss0 = 0
        for i, v in enumerate(y_pred):
            loss0 -= y_true[i] * np.log(y_pred[i]) + (1-y_true[i]) * np.log(1-y_pred[i])
        return (loss0/len(y_pred))
    if epoch == 1:
        loss = binary_crossentropy(preds[0].T.tolist()[0], labels)
    else:
        loss0 = binary_crossentropy(preds[0].T.tolist()[0], labels)
        loss1 = u_loss_evaluate(preds[1], embed_last, uids)
        loss = loss0 + gamma * loss1
    return acc, loss

