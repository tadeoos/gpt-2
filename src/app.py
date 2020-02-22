#!/usr/bin/env python3
import time
import logging

import redis
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder


def get_module_logger(mod_name):
    """
    To use this, do logger = get_module_logger(__name__)
    """
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


logging._warn_preinit_stderr = 0
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run():
    logger.info('MODEL IS UP')
    r = redis.Redis(host='redis', port=6379)
    p = r.pubsub(ignore_subscribe_messages=True)
    p.subscribe('model-titles')

    model_name = os.environ.get("MODEL_NAME", '1558M')
    seed = None
    nsamples = 1
    batch_size = 1
    length = None
    temperature = 0.7
    top_k = 40
    top_p = 1

    models_dir = '/models'
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        logger.info('Initiated. Starting loop...')
        while True:
            msg = p.get_message()
            if not msg:
                time.sleep(0.01)
                continue
            start = time.time()
            raw_text = msg.get('data').decode()
            logger.info(f'Received title: {raw_text}')
            context_tokens = enc.encode(raw_text)
            out = sess.run(
                output, feed_dict={context: [context_tokens]}
            )[:, len(context_tokens):]
            text = enc.decode(out[0])
            r.set(raw_text, text)
            took = time.time() - start
            logger.info(f'Text generation completed. Took {took:.3f} seconds')


if __name__ == '__main__':
    run()

