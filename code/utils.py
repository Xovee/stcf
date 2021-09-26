import numpy as np
import tensorflow as tf


def load_map(f, eif, scale_factor=4, batch_size=16, fraction=100, pre_train=False, tc=False):
    # generate coarse-grained flow map
    c = tf.keras.layers.AveragePooling2D(scale_factor)(f.reshape(-1, f.shape[-2], f.shape[-1], 1)) * scale_factor ** 2

    # split the datasets into training, validation, and test sets, in proportion to 2:1:1
    f_train, f_val, f_test = np.split(f, [int(len(f) / 2), int(len(f) / 4 * 3)])
    c_train, c_val, c_test = np.split(c, [int(len(c) / 2), int(len(c) / 4 * 3)])
    eif_train, eif_val, eif_test = np.split(eif, [int(len(eif) / 2), int(len(eif) / 4 * 3 )])

    if tc:
        # for temporal-contrasting pre-training network
        c_train_anchor = c_train[1:-1]  # t
        c_train_p1 = c_train[2:]  # t+1
        c_train_p2 = c_train[:-2]  # t-1

        train_ds = tf.data.Dataset.from_tensor_slices((c_train_anchor[:int(len(c_train)*fraction/100)],
                                                       c_train_p1[:int(len(c_train)*fraction/100)],
                                                       c_train_p2[:int(len(c_train)*fraction/100)])) \
            .shuffle(10000).batch(batch_size, drop_remainder=pre_train)
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((c_train[:int(len(c_train) * fraction / 100)],
                                                       f_train[:int(len(c_train) * fraction / 100)],
                                                       eif_train[:int(len(c_train) * fraction / 100)])) \
            .shuffle(10000).batch(batch_size, drop_remainder=pre_train)
    if not pre_train:
        val_ds = tf.data.Dataset.from_tensor_slices((c_val, f_val, eif_val))\
            .batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((c_test, f_test, eif_test))\
            .batch(batch_size)

        return train_ds, val_ds, test_ds
    else:
        return train_ds


def get_negative_mask(b_size):
    # remove similarity score of similar cascades
    # codes from https://github.com/sthalles/SimCLR-tensorflow/blob/master/utils/helpers.py
    negative_mask = np.ones((b_size, b_size * 2), dtype=bool)
    for i in range(b_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + b_size] = 0
    return tf.constant(negative_mask)


def loss_fn(pos_1, pos_2, criterion, b_size, temperature):
    # InfoNCE, codes from https://github.com/sthalles/SimCLR-tensorflow/blob/master/utils/helpers.py
    pos_1 = tf.math.l2_normalize(pos_1, axis=1)
    pos_2 = tf.math.l2_normalize(pos_2, axis=1)

    dot = dot_sim_1(pos_1, pos_2)  # [B, 1, 1]

    l_pos = tf.reshape(dot, (b_size, 1)) / temperature

    negatives = tf.concat([pos_1, pos_2], axis=0)

    loss = 0

    for positives in [pos_1, pos_2]:
        l_neg = dot_sim_2(positives, negatives)

        labels = tf.zeros(b_size, dtype=tf.int32)

        l_neg = tf.reshape(tf.boolean_mask(l_neg, get_negative_mask(b_size)), (b_size, -1)) / temperature

        logits = tf.concat([l_pos, l_neg], axis=1)
        loss += criterion(y_pred=logits, y_true=labels)

    loss = loss / (2 * b_size)

    return loss


def dot_sim_1(x, y):
    # codes from https://github.com/sthalles/SimCLR-tensorflow/blob/master/utils/losses.py
    return tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))


def dot_sim_2(x, y):
    # codes from https://github.com/sthalles/SimCLR-tensorflow/blob/master/utils/losses.py
    return tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)