from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from util import Progbar, minibatches

from evaluate import exact_match_score, f1_score

from qa_model import pad_sequences, my_f1_score, my_em_score

from qa_data import PAD_ID

import datetime

logger = logging.getLogger("hw4")
logging.basicConfig(level=logging.INFO)

max_q_words = 25 # longest sequence to parse
max_c_words = 250
embed_size = 100 #specified in qa_data.py. 50 is default. get_started.sh uses 100
hidden_size = 75 #about halfway between the embedding size and the eventual output size (2)
batch_size =  64 #32 was default but apparantly azure has high memery so going higher is good
num_epochs = 1 #they say we should be running for 5-7 epochs. Keep it at 5 till you know it's your best model. IT should improve most after 1 anyway


def run(load_train_dir, save_train_dir):

	start_time = datetime.datetime.now().replace(microsecond=0)

	print(load_train_dir)
	print(save_train_dir)

	print("creating session")
	sess = tf.Session()

	print("running")
	print("reading in data")
	text_file = open("./data/squad/train.span", "r")
	labels = text_file.read().split("\n")
	text_file.close()

	num_entries = int(len(labels)/1)-1 #take out the /1000 when you run for real. ?? ?  ? ?marks are just to remind me to do that
	#labels = tf.convert_to_tensor(labels)
	

	print("num entries = ", num_entries)
	print("num epochs = ", num_epochs)
	print("batch size = ", batch_size)

	

	print("creating model")
	input_q_placeholder = tf.placeholder(tf.float32, shape = (None, max_q_words, embed_size), name='input_q_placeholder')  
	input_c_placeholder = tf.placeholder(tf.float32, shape = (None, max_c_words, embed_size), name='input_c_placeholder')
	start_answer_placeholder = tf.placeholder(tf.int32, shape = (None, max_c_words), name='start_answer_placeholder')
	end_answer_placeholder = tf.placeholder(tf.int32, shape = (None, max_c_words), name='end_answer_placeholder')
	mask_q_placeholder = tf.placeholder(tf.bool,  shape = (None, max_q_words), name='mask_q_placeholder')
	mask_c_placeholder = tf.placeholder(tf.bool,  shape = (None, max_c_words), name='mask_q_placeholder')

	print("reading labels")

	#start_answer = []
	#end_answer = []
	start_true = []
	end_true = []
	#start_index = []
	#end_index =[]

	for i in range(num_entries):	#batch_size*batch_num, batch_size*(batch_num+1)
		#if (i%1 == 0):
		#	print("Label # ", i ," of ", num_entries)
		nums = labels[i].split()
		start_true.append(int(nums[0]))
		end_true.append(int(nums[1]))
		#start_true.append(start_index) 
		#end_true.append(end_index) 
	start_answer = (tf.one_hot(indices = start_true, depth = max_c_words, dtype = tf.int32).eval(session = sess))
	end_answer = (tf.one_hot(indices = end_true, depth = max_c_words, dtype = tf.int32).eval(session = sess))
	

	text_file = open("./data/squad/train.ids.question", "r")
	inputs_q = text_file.read().split("\n")
	text_file.close()

	print("reading questions")
	myMatrix_q = []
	for i in range(num_entries):
		#if (i%1000 == 0):
		#	print("Question # ", i ," of ", num_entries)
		nums = inputs_q[i].split()
		myArray = []
		for j in range(len(nums)):
			myArray.append(int(nums[j]))
		myMatrix_q.append(myArray)

	text_file = open("./data/squad/train.ids.context", "r")
	inputs_c = text_file.read().split("\n")
	text_file.close()
	
	print("reading contexts")
	myMatrix_c = []
	for i in range(num_entries):
		#if (i%1000 == 0):
		#	print("Context index # ", i ," of ", num_entries)
		nums = inputs_c[i].split()
		myArray = []
		for j in range(len(nums)):
			myArray.append(int(nums[j]))
		myMatrix_c.append(myArray)

		
	text_file = open("./data/squad/train.context", "r")
	inputs_c_text = text_file.read().split("\n")
	text_file.close()

	#print(inputs_c_text[1])

	c_text = []
	for i in range(num_entries):
		words = inputs_c_text[i].split(" ")
		myArray = []
		for j in range(len(words)):
			myArray.append(words[j])
		c_text.append(myArray)

		

	padded_q_inputs, masks_q = zip(*pad_sequences(data = myMatrix_q, max_length = max_q_words))
	padded_c_inputs, masks_c = zip(*pad_sequences(data = myMatrix_c, max_length = max_c_words))



	print("loading embeddings")
	embed_path = "./data/squad/glove.trimmed.100.npz"
	pretrained_embeddings = np.load(embed_path)
	logger.info("Keys")
	logger.info(pretrained_embeddings.keys())
	logger.info("Initialized embeddings.")
	pretrained_embeddings = tf.constant(pretrained_embeddings.f.glove) 

	embedded_q = (tf.nn.embedding_lookup(pretrained_embeddings, padded_q_inputs).eval(session = sess))
	embedded_c = (tf.nn.embedding_lookup(pretrained_embeddings, padded_c_inputs).eval(session = sess))
	print("Done Embedding")

	

	print("encoding")

	mask_int_q = tf.cast(mask_q_placeholder, tf.int32)
	srclen_q = tf.reduce_sum(mask_int_q, 1)
	mask_int_c = tf.cast(mask_c_placeholder, tf.int32)
	srclen_c = tf.reduce_sum(mask_int_c, 1)
	
	scope_q = "scope_q" 
	scope_c = "scope_c" 
	scope_decode = "scope_decode"


	
	LSTM_cell_q = tf.nn.rnn_cell.BasicLSTMCell(num_units = hidden_size)	
	LSTM_cell_c = tf.nn.rnn_cell.BasicLSTMCell(num_units = hidden_size)
	LSTM_cell_decode = tf.nn.rnn_cell.BasicLSTMCell(num_units = hidden_size)

	print("filtering")
	
	normed_q = tf.nn.l2_normalize(input_q_placeholder, dim=2)
	normed_c = tf.nn.l2_normalize(input_c_placeholder, dim=2)

	matrix = tf.matmul(normed_q, normed_c, transpose_b = True)
	attention = tf.reduce_max(matrix, axis = 1)
	attention = tf.expand_dims(attention, axis = 2)
	filtered_c = input_c_placeholder*attention

	outputs_q, state_q = tf.nn.bidirectional_dynamic_rnn(LSTM_cell_q, LSTM_cell_q, input_q_placeholder, srclen_q, scope = scope_q, time_major = False, dtype = tf.float32)
	outputs_c, state_c = tf.nn.bidirectional_dynamic_rnn(LSTM_cell_c, LSTM_cell_c, filtered_c, srclen_c, scope = scope_c, time_major = False, dtype = tf.float32)

	hidden_q = (state_q[0][1], state_q[1][1])
	hidden_q = tf.concat(1, hidden_q)

	q_piazza_int = hidden_q
	q_piazza = tf.expand_dims(q_piazza_int, axis = 2)

	X_piazza = tf.concat(2, outputs_c)
	X_piazza = tf.transpose(X_piazza, [0,2,1])

	intermediate = q_piazza*X_piazza

	p_s_true = tf.reduce_sum(intermediate, axis = 1)   #starting place probabilities
	p_s_true = tf.nn.softmax(p_s_true)


	p_s_false = 1-p_s_true
	p_s = tf.pack([p_s_false, p_s_true], axis = 2)

	p_e_true = p_s_true
	p_e_true = tf.nn.softmax(p_e_true)
	p_e_false = 1-p_e_true
	p_e = tf.pack([p_e_false, p_e_true], axis = 2)

	a_s = tf.argmax(p_s_true, axis=1)
	a_e = tf.argmax(p_e_true, axis=1)
	a_e = tf.maximum(a_s, a_e)

	l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = p_s, labels = start_answer_placeholder)
	l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = p_e, labels = end_answer_placeholder)
	loss = l1+l2
	
	loss_by_question = tf.reduce_sum(loss, axis = 1)

	avgLoss = tf.reduce_mean(loss_by_question)

	train_step = tf.train.AdamOptimizer().minimize(avgLoss)
	tf.global_variables_initializer().run(session = sess)

	print("training")
	num_batches = int(np.floor(num_entries/batch_size))
	print("num batches: ", num_batches)

	print("Creating saver")
	saver = tf.train.Saver()

		
	for j in range (num_epochs):  #epochs
		batch_error_total = 0
		print("epoch %d of %d" % (j+1, num_epochs))
		print("not shuffling yet")

		for i in range(num_batches):  
			if (i%100 == 0):
				print("Batch # ", i ," of ", num_batches)
			batch_q = embedded_q[i*batch_size:(i+1)*batch_size]  
			batch_c = embedded_c[i*batch_size:(i+1)*batch_size]
			batch_mask_q = masks_q[i*batch_size:(i+1)*batch_size]
			batch_mask_c = masks_c[i*batch_size:(i+1)*batch_size]
			start_batch = []
			end_batch = []
			for k in range(batch_size):
				start_batch.append(start_answer[i*batch_size+k])
				end_batch.append(end_answer[i*batch_size+k])			
			_, batch_error, p_s_true_ = sess.run([train_step, avgLoss, p_s_true], feed_dict={input_q_placeholder: batch_q, input_c_placeholder: batch_c, start_answer_placeholder: start_batch, end_answer_placeholder: end_batch, mask_q_placeholder: batch_mask_q, mask_c_placeholder: batch_mask_c})			
			batch_error_total += float(batch_error)/num_batches
		avg_train_error = batch_error_total
		print("epoch %d has average training error of %f" % (j+1, avg_train_error))
		print("saving")
		tf.add_to_collection('vars2', a_e)
		tf.add_to_collection('vars2', a_s)
		tf.add_to_collection('vars2', input_q_placeholder)
		tf.add_to_collection('vars2', input_c_placeholder)
		tf.add_to_collection('vars2', mask_q_placeholder)
		tf.add_to_collection('vars2', mask_c_placeholder)

		saver.save(sess, 'my-model')

		print("Evaluating EM and f1 scores")

		test_samples = np.minimum(num_entries, 100)

		test_q = embedded_q[0:test_samples]
		test_c = embedded_c[0:test_samples]
		batch_mask_q = masks_q[0:test_samples]
		batch_mask_c = masks_c[0:test_samples]

		start_pred, end_pred = sess.run([a_s, a_e], feed_dict={input_q_placeholder: test_q, input_c_placeholder: test_c, mask_q_placeholder: batch_mask_q, mask_c_placeholder: batch_mask_c})

		their_f1 = []
		their_em = []
		my_f1 = []
		my_em = []


		for i in range(test_samples):
			p = c_text[i]
			answer = p[start_pred[i]: end_pred[i]+1]
			answer = ' '.join(answer)
			true_answer = p[start_true[i]: end_true[i]+1]
			true_answer = ' '.join(true_answer)
			their_f1.append(f1_score(answer, true_answer))
			their_em.append(exact_match_score(answer, true_answer))
			my_f1.append(my_f1_score(start_pred[i], end_pred[i], start_true[i], end_true[i]))
			my_em.append(my_em_score(start_pred[i], end_pred[i], start_true[i], end_true[i]))
		their_f1_score = np.average(their_f1)
		their_em_score = np.average(their_em)
		f1 = np.average(my_f1)
		em = np.average(my_em)

		print("Their f1 train score: ", their_f1_score, " em score: ", their_em_score, " on ", test_samples, " samples")
		print("My f1 train score: ", f1, " em score: ", em, " on ", test_samples, " samples")


	print("Evaluating EM and f1 scores on Validation set")

	test_samples = np.minimum(num_entries, 100)

	print("reading labels")
	text_file = open("./data/squad/val.span", "r")
	labels = text_file.read().split("\n")
	text_file.close()

	start_true_val = []
	end_true_val = []

	for i in range(test_samples):	#batch_size*batch_num, batch_size*(batch_num+1)
		nums = labels[i].split()
		start_true_val.append(int(nums[0]))
		end_true_val.append(int(nums[1]))

	text_file = open("./data/squad/val.ids.question", "r")
	inputs_q = text_file.read().split("\n")
	text_file.close()

	print("reading questions")
	myMatrix_q = []
	for i in range(test_samples):
		nums = inputs_q[i].split()
		myArray = []
		for j in range(len(nums)):
			myArray.append(int(nums[j]))
		myMatrix_q.append(myArray)

	text_file = open("./data/squad/val.ids.context", "r")
	inputs_c = text_file.read().split("\n")
	text_file.close()
	
	print("reading contexts")
	myMatrix_c = []
	for i in range(test_samples):
		nums = inputs_c[i].split()
		myArray = []
		for j in range(len(nums)):
			myArray.append(int(nums[j]))
		myMatrix_c.append(myArray)

		
	text_file = open("./data/squad/val.context", "r")
	inputs_c_text = text_file.read().split("\n")
	text_file.close()

	c_text_val = []
	for i in range(test_samples):
		words = inputs_c_text[i].split(" ")
		myArray = []
		for j in range(len(words)):
			myArray.append(words[j])
		c_text_val.append(myArray)


	padded_q_inputs_val, masks_q_val = zip(*pad_sequences(data = myMatrix_q, max_length = max_q_words))
	padded_c_inputs_val, masks_c_val = zip(*pad_sequences(data = myMatrix_c, max_length = max_c_words))

	embedded_q_val = (tf.nn.embedding_lookup(pretrained_embeddings, padded_q_inputs_val).eval(session = sess))
	embedded_c_val = (tf.nn.embedding_lookup(pretrained_embeddings, padded_c_inputs_val).eval(session = sess))

	test_q = embedded_q_val
	test_c = embedded_c_val
	batch_mask_q = masks_q_val
	batch_mask_c = masks_c_val

	start_pred_val, end_pred_val = sess.run([a_s, a_e], feed_dict={input_q_placeholder: test_q, input_c_placeholder: test_c, mask_q_placeholder: batch_mask_q, mask_c_placeholder: batch_mask_c})

	their_f1 = []
	their_em = []
	my_f1 = []
	my_em = []

	print(test_samples)
	print(len(end_true_val))
	for i in range(test_samples):
		p = c_text_val[i]
		answer = p[start_pred_val[i]: end_pred_val[i]+1]
		answer = ' '.join(answer)
		true_answer = p[start_true_val[i]: end_true_val[i]+1]
		true_answer = ' '.join(true_answer)
		their_f1.append(f1_score(answer, true_answer))
		their_em.append(exact_match_score(answer, true_answer))
		my_f1.append(my_f1_score(start_pred_val[i], end_pred_val[i], start_true_val[i], end_true_val[i]))
		my_em.append(my_em_score(start_pred_val[i], end_pred_val[i], start_true_val[i], end_true_val[i]))
	their_f1_score = np.average(their_f1)
	their_em_score = np.average(their_em)
	f1 = np.average(my_f1)
	em = np.average(my_em)

	print("Their f1 valscore: ", their_f1_score, " em score: ", their_em_score, " on ", test_samples, " samples")
	print("My f1 val score: ", f1, " em score: ", em, " on ", test_samples, " samples")
	end_time = datetime.datetime.now().replace(microsecond=0)
	print("Elapsed Time: ", end_time-start_time)

		
load_train_dir = "./train"
save_train_dir = "./train"
run(load_train_dir, save_train_dir)