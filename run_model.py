import datetime
import json
import os
import random
import time

import tensorflow as tf
import redis
import numpy as np
from tensorboard.compat.tensorflow_stub.errors import NotFoundError
from tensorflow.python.saved_model import tag_constants

os.environ["CUDA_VISIBLE_DEVICES"] = ""

last_reload = datetime.datetime.now()

run_without_model = not os.path.isdir('model');
with tf.Session(graph=tf.Graph()) as sess:

    if not run_without_model:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], "model")

    graph = tf.get_default_graph()

    print([op.values() for op in graph.get_operations()])

    while True:
        r = redis.Redis(host='localhost', port=6379, db=0)
        state = r.get('state')
        if not state:
            time.sleep(.001)
            continue
        r.delete("state")
        data = json.loads(state)
        x = data["food"]
        individual_values = data["individual_values"]
        print(individual_values)
        if not run_without_model:
            actions = sess.run(['action_pred:0', 'reward_pred:0'],
                     feed_dict={'food:0': np.array([x], dtype=float), 'individual_values:0': np.array([individual_values], dtype=float)})[0][0]
            print(actions)

            actions = actions.tolist()
            if random.randint(0, 100) > 80:
                actions = [random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)]
        else:
            actions = [random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)]

        r.set("action", json.dumps(actions))


        if (datetime.datetime.now() - last_reload).seconds > 200 and os.path.isdir('model'):
            try:
                tf.saved_model.loader.load(sess, [tag_constants.SERVING], "model")
                last_reload = datetime.datetime.now()
                print("reloaded model")
            except:
                print("Failed to reload model")



        #print(actions)