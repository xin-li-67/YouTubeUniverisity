import os, time
import json, random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq

from sklearn.model_selection import train_test_split

xo = np.load("iseq_n.npy")
yo = np.load("oseq_n.npy")
x = xo[:, :26, :]
y = yo[:, :26]

nxchars = x.shape[2]
print(nxchars)
ltokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', '=', '#leq', '#neq', '#geq', 
           '#alpha', '#beta', '#lambda', '#lt', '#gt', 'x', 'y', '^', '#frac', '{', '}' ,' ']
print("#ltokens: ", len(ltokens))

x_seq_length = len(x[0])
y_seq_length = len(y[0])- 1
print(x_seq_length, y_seq_length)

def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start + batch_size], y[start:start + batch_size]
        start += batch_size

batch_size = 512
nodes = 256
embed_size = 20

# Tensor where we will feed the data into graph
inputs = tf.placeholder(tf.float32, (None, x_seq_length, nxchars), 'inputs')
outputs = tf.placeholder(tf.int32, (None, None), 'output')
targets = tf.placeholder(tf.int32, (None, None), 'targets')

# Embedding layers
output_embedding = tf.Variable(tf.random_uniform((len(ltokens) + 1, embed_size), -1.0, 1.0), name='dec_embedding')
date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)

with tf.variable_scope("encoding") as encoding_scope:
    lstm_enc = tf.contrib.rnn.BasicLSTMCell(nodes)
    _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=inputs, dtype=tf.float32)

with tf.variable_scope("decoding") as decoding_scope:
    lstm_dec = tf.contrib.rnn.BasicLSTMCell(nodes)
    dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=date_output_embed, initial_state=last_state)

#connect outputs to 
logits = tf.contrib.layers.fully_connected(dec_outputs, num_outputs=len(ltokens) + 1, activation_fn=None) 
with tf.name_scope("optimization"):
    # Loss function
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print(y_train[0])

def save(sess):
    saver = tf.train.Saver(None)
    save_path = saver.save(sess, save_path="seq_mod/model", global_step=None)
    print('model saved at %s' % save_path)

with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    restore(sess)
    epochs = 100

    for epoch_i in range(epochs):
        start_time = time.time()
        for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
            _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                feed_dict = {inputs: source_batch, outputs: target_batch[:, :-1], targets: target_batch[:, 1:]})

        accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:, 1:])
        print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss, accuracy, time.time() - start_time))

        source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))

        dec_input = np.zeros((len(source_batch), 1)) + len(ltokens)
        for i in range(y_seq_length):
            batch_logits = sess.run(logits, feed_dict = {inputs: source_batch, outputs: dec_input})
            prediction = batch_logits[:,-1].argmax(axis=-1)
            dec_input = np.hstack([dec_input, prediction[:,None]])

        print('Accuracy on test set is: {:>6.3f}'.format(np.mean(dec_input == target_batch)))     
        if epoch_i % 5 == 0:
            save(sess)
    
    save(sess)

def restore(sess):
    saver = tf.train.Saver(None)
    path = "seq_mod/model"
    saver.restore(sess, save_path=path)
    print('model restored from %s' % path)

with tf.Session() as sess: 
    restore(sess)
    batch_size = 512
    source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))
    dec_input = np.zeros((len(source_batch), 1)) + len(ltokens)
    
    for i in range(y_seq_length):
        batch_logits = sess.run(logits, feed_dict = {inputs: source_batch, outputs: dec_input})
        prediction = batch_logits[:,-1].argmax(axis=-1)
        dec_input = np.hstack([dec_input, prediction[:,None]])

    print('Accuracy on test set is: {:>6.3f}'.format(np.mean(dec_input == target_batch)))

i = 102
print(dec_input[i,:])
print(ltokens)
seq = ""
for c in dec_input[i,1:]:
    c = int(c)
    if c != 28:
        seq += ltokens[c] 
        
print(seq)

seq = ""
for c in target_batch[i,1:]:
    c = int(c)
    
    if c != 28:
        seq += ltokens[c] 

print("Correct: ", seq)

with tf.Session() as sess: 
    restore(sess)
    batch_size = 512
    source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))

    dec_input = np.zeros((len(source_batch), 1)) + len(ltokens)
    for i in range(y_seq_length):
        batch_logits = sess.run(logits, feed_dict = {inputs: source_batch, outputs: dec_input})
        prediction = batch_logits[:, -1].argmax(axis=-1)
        dec_input = np.hstack([dec_input, prediction[:, None]])

    print('Accuracy on test set is: {:>6.3f}'.format(np.mean(dec_input == target_batch)))

    cc = 0
    fcc = 0
    fc = 0
    fl = 0
    nl = 0
    for i in range(len(dec_input)):
        feq = False
        pseq = ""
        for c in dec_input[i,1:]:
            c = int(c)
            if c != 28:
                pseq += ltokens[c] 

        cseq = ""
        cseql = ""
        for c in target_batch[i,1:]:
            c = int(c)

            if c != 28:
                cseq += ltokens[c] 
                cseql += ltokens[c][0]
                if ltokens[c] == "#frac":
                    fc += 1
                    feq = True

        if pseq == cseq:
            cc += 1
            if feq:
                fcc += 1
        cseql = cseql.rstrip()
        if feq:
            fl += len(cseql)
        else:
            nl += len(cseql)
        
    print("Accuracy %.2f %%" % (cc / len(dec_input) * 100))  
    print("Accuracy for fraction equations %.2f %%" % (fcc / fc * 100))
    print("Accuracy for simple equations %.2f %%" % ((cc - fcc) / (len(dec_input) - fc) * 100))
    print("Average length frac: %.1f" % (fl / fc))
    print("Average length simple: %.1f" % (nl / (len(dec_input) - fc)))