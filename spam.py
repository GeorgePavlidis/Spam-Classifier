#!  /usr/bin/python
import pandas as pd
import numpy as np
import os
import time
import datetime
import tensorflow as tf
from tensorflow.contrib import learn
import sys as sys















def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]





vocab_size = 11048
embedding_size = 300
sequence_length = 188
classes = 2
 
data = pd.read_csv("spam.csv", encoding='ISO-8859-1')

print (len(data), data['Unnamed: 2'].isnull().sum())
data = data[pd.isnull(data['Unnamed: 2'])]
data = data[pd.isnull(data['Unnamed: 3'])]
data = data[pd.isnull(data['Unnamed: 4'])]
data=data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)


inputTV = data['v2'].tolist()
outputV = data['v1'].tolist()


for i in range(0,len(data)):

	if(outputV[i]=='spam'):
		outputV[i] = [1,0]
	else:
		outputV[i] = [0,1]




vocab_processor = learn.preprocessing.VocabularyProcessor(188)

## Transform the documents using the vocabulary.
x = np.array(list(vocab_processor.fit_transform(inputTV)))    

## Extract word:id mapping from the object.
vocab_dict = vocab_processor.vocabulary_._mapping

## Sort the vocabulary dictionary on the basis of values(id).
## Both statements perform same task.
#sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])

## Treat the id's as index into list and create a list of words in the ascending order of id's
## word with id i goes at index i of the list.
vocabulary = list(list(zip(*sorted_vocab))[0])


#separete data to train and test sets
dev_sample_index = int(len(outputV)*0.7)
x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
y_train, y_dev = outputV[:dev_sample_index], outputV[dev_sample_index:]

inputT = tf.placeholder(tf.int32, [None, sequence_length], name="inputT_x")

y = tf.placeholder(tf.float32, [None, classes], name="inputT_y")

keep_prob = tf.placeholder(tf.float32)


#convolusional layer weight
weight2 = tf.Variable(tf.random_normal([4, embedding_size ,1, 128]))
bias2 =  tf.Variable(tf.random_normal([128]))
#flat
weight3 = tf.Variable(tf.random_normal([384, 1024]))
bias3 =  tf.Variable(tf.random_normal([1024]))
#out
weight4 = tf.Variable(tf.random_normal([384, classes]))
bias4 =  tf.Variable(tf.random_normal([classes]))


with tf.device('/cpu:0'), tf.name_scope("embedding"):
	#embedding layer weight
	weight1 = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
	embedded_chars = tf.nn.embedding_lookup(weight1, inputT)
	embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

conv = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(embedded_chars_expanded, weight2, strides=[1, 1, 1, 1], padding="VALID") + bias2), ksize=[1, sequence_length-5, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

conv_flat = tf.reshape(conv, [-1, 384])

#fc = tf.nn.relu(tf.matmul(conv_flat, weight3)+ bias3)
fc = tf.nn.dropout(conv_flat, keep_prob)

score = tf.matmul(fc, weight4) + bias4
out = tf.argmax(score, 1)

losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))
#lol = tf.equal(out, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(tf.equal(out, tf.argmax(y, 1)), "float"))





global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-3)
grads_and_vars = optimizer.compute_gradients(losses)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


loss_summary = tf.summary.scalar("loss", losses)
acc_summary = tf.summary.scalar("accuracy", accuracy)





session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
sess = tf.Session(config=session_conf)

out_dir = ".\log"
        # Train Summaries
train_summary_op = tf.summary.merge([loss_summary, acc_summary])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)


        # Dev summaries
dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)


checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
	os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # Write vocabulary
vocab_processor.save(os.path.join(out_dir, "vocab"))

#print (shape(outputV))

sess.run(tf.global_variables_initializer())
#saver.restore(sess, './log/checkpoints/model-1200')

def train_step(x_batch, y_batch):
	global accuracy
	_, step, summaries, loss, acc = sess.run([train_op, global_step, train_summary_op, losses, accuracy],  feed_dict={inputT: x_batch, y: y_batch, keep_prob: 0.5})

	time_str = datetime.datetime.now().isoformat()
	print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
	train_summary_writer.add_summary(summaries, step)

	writer = tf.summary.FileWriter("./log", sess.graph)


def dev_step(x_batch, y_batch, writer=None):
	"""
	Evaluates model on a dev set
	"""

	step, summaries, loss, acc = sess.run([global_step, dev_summary_op, losses, accuracy],feed_dict={inputT: x_batch, y: y_batch, keep_prob: 1.0})
	time_str = datetime.datetime.now().isoformat()
	print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
	if writer:
		writer.add_summary(summaries, step)

#create batches
batches = batch_iter( list(zip(x_train, y_train)), 64, 1)
        # Training loop. For each batch...
for batch in batches:
	x_batch, y_batch = zip(*batch)
	train_step(x_batch, y_batch)
	current_step = tf.train.global_step(sess, global_step)
	if current_step % 100 == 0:
		print("\nEvaluation:")
		dev_step(x_dev, y_dev, writer=dev_summary_writer)
		print("")
	if current_step % 100 == 0:
		path = saver.save(sess, checkpoint_prefix, global_step=current_step)
		print("Saved model checkpoint to {}\n".format(path))


s=0
h=0
s1=0
h1=0
for i in range(len(y_dev)):
	if y_dev[i]==[1,0]:
		s+=1
		if outla[i]==0:
			h+=1
	elif y_dev[i]==[0,1]:
		s1+=1
		if outla[i]==1:
			h1+=1


print ("accuracy at spam ", (h/s)*100,"%", " and accuracy at ham", (h1/s1)*100,"%")
