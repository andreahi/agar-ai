import json
import os
import shutil
import time

import tensorflow as tf
import redis
import numpy as np
from tensorflow.python.saved_model.simple_save import simple_save

r = redis.Redis(host='localhost', port=6379, db=0)


def addNameToTensor(someTensor, theName):
    return tf.identity(someTensor, name=theName)

def shuffle_in_unison(a, b, c, d, e, f):
    rng_state = np.random.get_state()
    np.random.shuffle(a)

    np.random.set_state(rng_state)
    np.random.shuffle(b)

    np.random.set_state(rng_state)
    np.random.shuffle(c)

    np.random.set_state(rng_state)
    np.random.shuffle(d)

    np.random.set_state(rng_state)
    np.random.shuffle(e)

    np.random.set_state(rng_state)
    np.random.shuffle(f)

def build_layer(x, num_units, keep_prob, layer_n = False, dropout = False):
    """Builds a layer with input x; dropout and layer norm if specified."""

    init_s = 0.3


    nn = tf.contrib.layers.fully_connected(
        x,
        num_units,
        activation_fn=tf.nn.leaky_relu,
        normalizer_fn=None if not layer_n else tf.contrib.layers.layer_norm,
        normalizer_params={},
        weights_initializer=tf.random_uniform_initializer(-init_s, init_s)
    )

    if dropout:
        nn = tf.nn.dropout(nn, keep_prob)

    return nn


def forward_pass(x, single_value_inputs, keep_prob):
    init_s = 0.3

    with tf.variable_scope("model_weighted", reuse=tf.AUTO_REUSE):
        nn1 = tf.concat([tf.layers.flatten(x)], axis=1)
        for num_units in [40, 20, 10, 5]:
            if num_units > 0:
                nn1 = build_layer(nn1, num_units, keep_prob, dropout=False)

        nn2 = tf.concat([individual_values], axis=1)
        for num_units in [10, 8, 7, 6]:
            if num_units > 0:
                nn2 = build_layer(nn2, num_units, keep_prob, dropout=False)

        y_0 = tf.layers.dense(build_layer(tf.concat([nn2], axis=1), 5, keep_prob, dropout=False), 1, kernel_initializer=tf.random_uniform_initializer(-init_s, init_s))


        nn1 = tf.concat([tf.layers.flatten(x)], axis=1)
        for num_units in [40, 20, 10, 5]:
            if num_units > 0:
                nn1 = build_layer(nn1, num_units, keep_prob, dropout=False)

        nn2 = tf.concat([individual_values], axis=1)
        for num_units in [10, 8, 7, 6]:
            if num_units > 0:
                nn2 = build_layer(nn2, num_units, keep_prob, dropout=False)

        y_1 = tf.layers.dense(build_layer(tf.concat([nn2], axis=1), 5, keep_prob, dropout=False), 4, kernel_initializer=tf.random_uniform_initializer(-init_s, init_s))



    return addNameToTensor(y_0, "reward_pred"), addNameToTensor(y_1, "action_pred")


num_actions = 4

with tf.Session() as sess:
    max_grad_norm = 50.0
    batch_size = 6300

    food = tf.placeholder(
        shape=[None, 20, 20],
        dtype=tf.float32,
        name="food")

    individual_values = tf.placeholder(
        shape=[None, 4],
        dtype=tf.float32,
        name="individual_values")

    # reward vector
    reward = tf.placeholder(
        shape=[None, 1],
        dtype=tf.float32,
        name="target")

    # reward vector
    next_pred_reward = tf.placeholder(
        shape=[None, 1],
        dtype=tf.float32,
        name="target")

    # weights (1 for selected action, 0 otherwise)
    weights = tf.placeholder(
        shape=[None, num_actions],
        dtype=tf.float32,
        name="{}_w".format("weight"))

    actions_performed = tf.placeholder(
        shape=[None, num_actions],
        dtype=tf.float32,
        name="{}_w".format("weight"))


    actions_target = tf.placeholder(
        shape=[None, num_actions],
        dtype=tf.float32,
        name="{}_w".format("weight"))



    graph = tf.Graph()

    keep_prob = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")

    reward_pred, action_pred = forward_pass(food, individual_values, keep_prob)


    loss = tf.abs(action_pred - actions_target)
    print("shape loss: ", str(loss.get_shape()))
    expected_diff = tf.stop_gradient(tf.square((reward_pred - next_pred_reward)))
    weighted_action_loss = tf.reduce_sum(tf.multiply(expected_diff, tf.multiply(actions_performed, loss)))
    #weighted_loss = loss

    reward_loss = tf.reduce_sum(tf.abs(reward_pred - reward))

    cost = (weighted_action_loss / batch_size)

    cost_reward = (reward_loss / batch_size)


    optimizer = tf.train.AdamOptimizer(0.0001)
    optimizer_reward = tf.train.AdamOptimizer(0.0001)

    train_op = optimizer.minimize(loss)

    train_op_reward = optimizer_reward.minimize(reward_loss)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with graph.as_default():
        sess.run(init)

    if os.path.isdir('model'):
        saver.restore(sess, "model_tmp/model.ckpt")

    while True:

        x_train = []
        reward_train = []
        actions_train = []
        individual_values_train = []
        next_x_train = []
        next_individual_values_train = []

        stored_next_x_train = r.get("next_x_train")
        if stored_next_x_train:
            next_x_train = json.loads(stored_next_x_train)

        stored_x_train = r.get("x_train")
        if stored_x_train:
            x_train = json.loads(stored_x_train)

        stored_reward_train = r.get("reward_train")
        if stored_reward_train:
            reward_train = json.loads(stored_reward_train)

        stored_actions_train = r.get("actions_train")
        if stored_actions_train:
            actions_train = json.loads(stored_actions_train)

        stored_individual_values_train = r.get("individual_values")
        if stored_individual_values_train:
            individual_values_train = json.loads(stored_individual_values_train)

        stored_next_individual_values_train = r.get("next_individual_values_train")
        if stored_next_individual_values_train:
            next_individual_values_train = json.loads(stored_next_individual_values_train)


        new_data = r.get('data')
        if new_data:
            data = json.loads(new_data)
            #print(r.get('data'))

            r.delete("data")

            if not data["screens"]:
                print("Got empty dataset :s ")
                continue



            if next_x_train:
                next_x_train = np.concatenate([next_x_train, np.array(data['screens'])[1:]])
            else:
                next_x_train = np.array(data['screens'])[1:]

            if x_train:
                x_train = np.concatenate([x_train, np.array(data['screens'][0:-1])])
            else:
                x_train = np.array(data['screens'])[0:-1]

            if reward_train:
                reward_train = np.concatenate(
                    [reward_train, np.expand_dims([data['scores'][-1] for _ in data['scores'][0:-1]], axis=1)])
            else:
                reward_train = np.expand_dims([data['scores'][-1] for _ in data['scores'][0:-1]], axis=1)

            if actions_train:
                actions_train = np.concatenate([actions_train, np.array((data['actions'])[0:-1])])
            else:
                actions_train = np.array((data['actions'])[0:-1])


            if individual_values_train:
                individual_values_train = np.concatenate([individual_values_train, np.array(data['individual_values'])[0:-1]])
            else:
                individual_values_train = np.array(data['individual_values'])[0:-1]


            if next_individual_values_train:
                next_individual_values_train = np.concatenate([next_individual_values_train, np.array(data['individual_values'])[1:]])
            else:
                next_individual_values_train = np.array(data['individual_values'])[1:]




            r.set("x_train", json.dumps(x_train.tolist()))
            r.set("reward_train", json.dumps(reward_train.tolist()))
            r.set("actions_train", json.dumps(actions_train.tolist()))
            r.set("individual_values", json.dumps(individual_values_train.tolist()))
            r.set("next_individual_values_train", json.dumps(next_individual_values_train.tolist()))
            r.set("next_x_train", json.dumps(next_x_train.tolist()))

        shuffle_in_unison(next_x_train, next_individual_values_train, x_train, individual_values_train, reward_train, actions_train)
        print("dataset size: ", len(x_train))
        step = 10000
        for _ in range(100):
            for i in range(0, 100000, step):
                if i + step > len(x_train):
                    break

                y_pred_v = sess.run(
                    [reward_pred],
                    feed_dict={food: x_train[i:i+step], individual_values:individual_values_train[i:i+step], keep_prob: 1.0})[0]


                next_y_pred_v = sess.run(
                    [reward_pred],
                    feed_dict={food: next_x_train[i:i+step], individual_values:next_individual_values_train[i:i+step], keep_prob: 1.0})[0]


                actions_target_train = np.zeros([len(reward_train[i:i+step]), num_actions])
                positive_target = (next_y_pred_v - y_pred_v) > 0.
                negative_target = (next_y_pred_v - y_pred_v) <= 0.

                actions_target_train[np.squeeze(positive_target)] = 1
                actions_target_train[np.squeeze(negative_target)] = 0

                _, reward_loss_v = sess.run(
                    [train_op_reward, reward_loss],
                    feed_dict={food: x_train[i:i+step], individual_values:individual_values_train[i:i+step], reward: reward_train[i:i+step], actions_performed: actions_train[i:i+step],
                               actions_target: actions_target_train, next_pred_reward: next_y_pred_v,
                               keep_prob: 0.99})

                _, cost_train, weighted_action_loss_v = sess.run(
                    [train_op, cost, weighted_action_loss],
                    feed_dict={food: x_train[i:i+step], individual_values:individual_values_train[i:i+step], reward: reward_train[i:i+step], actions_performed: actions_train[i:i+step],
                               actions_target: actions_target_train, next_pred_reward: next_y_pred_v,
                               keep_prob: 0.99})

                print("cost_train: " + str(cost_train) + " reward_loss_v: " + str(reward_loss_v) + " weighted_action_loss_v: " + str(weighted_action_loss_v))

        shutil.rmtree('model', ignore_errors=True)

        simple_save(sess,
                    "model",
                    inputs={"input": food},
                    outputs={"action_pred": action_pred, "reward_pred": reward_pred})
        save_path = saver.save(sess, "model_tmp/model.ckpt")

        #action_pred_v = sess.run(
        #    [action_pred],
        #    feed_dict={food: x_train, individual_values: individual_values_train, keep_prob: 1.0})
        #print(action_pred_v)


