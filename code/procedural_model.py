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

from qa_model import pad_sequences

from qa_data import PAD_ID

import time

logger = logging.getLogger("hw4")
logging.basicConfig(level=logging.INFO)

max_q_words = 25 # longest sequence to parse
max_c_words = 250
embed_size = 100 #specified in qa_data.py. 50 is default. get_started.sh uses 100
hidden_size = 50 #about halfway between the embedding size and the eventual output size (2)
batch_size =  64 #32 was default but apparantly azure has high memery so going higher is good
num_epochs = 5    #they say we should be running for 5-7 epochs. Keep it at 5 till you know it's your best model. IT should improve most after 1 anyway


def run(load_train_dir, save_train_dir):

	print(load_train_dir)
	print(save_train_dir)

	print("creating session")
	sess = tf.Session()

	print("running")
	print("reading in data")
	text_file = open("./data/squad/train.span", "r")
	labels = text_file.read().split("\n")
	text_file.close()

	num_entries = int(len(labels)/1000)-1 #take out the /1000 when you run for real. ?? ?  ? ?marks are just to remind me to do that
	#labels = tf.convert_to_tensor(labels)
	

	print("num entries = ", num_entries)
	print("num epochs = ", num_epochs)
	print("batch size = ", batch_size)

	

	print("creating model")
	input_q_placeholder = tf.placeholder(tf.float32, shape = (batch_size, max_q_words, embed_size), name='input_q_placeholder')  
	input_c_placeholder = tf.placeholder(tf.float32, shape = (batch_size, max_c_words, embed_size), name='input_c_placeholder')
	start_answer_placeholder = tf.placeholder(tf.int32, shape = (batch_size, max_c_words), name='start_answer_placeholder')
	end_answer_placeholder = tf.placeholder(tf.int32, shape = (batch_size, max_c_words), name='end_answer_placeholder')
	mask_q_placeholder = tf.placeholder(tf.bool,  shape = (batch_size, max_q_words), name='mask_q_placeholder')
	mask_c_placeholder = tf.placeholder(tf.bool,  shape = (batch_size, max_c_words), name='mask_q_placeholder')
	#batch_num = tf.placeholder(tf.int32,  shape = (1), name='batch_num')

	#batch_num = batch_num.eval(session = sess)
	#batch_num = batch_num[0]
	#batch_num = sess.run(batch_num)

	#indexed_tensor = tf.gather(labels, range(batch_size*batch_num, batch_size*(batch_num+1)))

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
	
	#start_answer = tf.one_hot(indices = start_true, depth = max_c_words, dtype = tf.int32).eval(session = sess)
	#end_answer = tf.one_hot(indices = end_true, depth = max_c_words, dtype = tf.int32).eval(session = sess)
	print("shape of start_true")
	print(np.shape(start_true))

	print("shape of start_answer")
	print(np.shape(start_answer))

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
		#if (i%1000 == 0):
		#	print("Context word # ", i ," of ", num_entries)
		words = inputs_c_text[i].split(" ")
		myArray = []
		for j in range(len(words)):
			myArray.append(words[j])
			#print("words j")
			#print(words[j])
		#print(myArray)
		c_text.append(myArray)
		#print("c_text")
		#print(c_text)
		
		

    
	

	padded_q_inputs, masks_q = zip(*pad_sequences(data = myMatrix_q, max_length = max_q_words))
	padded_c_inputs, masks_c = zip(*pad_sequences(data = myMatrix_c, max_length = max_c_words))


	

	print("loading embeddings")
	embed_path = "./data/squad/glove.trimmed.100.npz"
	pretrained_embeddings = np.load(embed_path)
	logger.info("Keys")
	logger.info(pretrained_embeddings.keys())
	logger.info("Initialized embeddings.")
	pretrained_embeddings = tf.constant(pretrained_embeddings.f.glove) 

	

	#embedded_q = []
	#embedded_c = []

	#start_time = time.time()

	#for i in range(num_entries):
	    #if (i%100 == 0):
	    #    elapsed_time = time.time() - start_time
	    #    start_time = time.time()
	    #    print("Embedding question # ",i, "of ", num_entries, " took ",elapsed_time, "seconds")
	    #embedded_q.append(tf.nn.embedding_lookup(pretrained_embeddings, padded_q_inputs[i]).eval(session = sess))
	    #embedded_c.append(tf.nn.embedding_lookup(pretrained_embeddings, padded_c_inputs[i]).eval(session = sess))
	embedded_q = (tf.nn.embedding_lookup(pretrained_embeddings, padded_q_inputs).eval(session = sess))
	embedded_c = (tf.nn.embedding_lookup(pretrained_embeddings, padded_c_inputs).eval(session = sess))
	print("Done Embedding")

	"""
	print("creating model")
	input_q_placeholder = tf.placeholder(tf.float32, shape = (batch_size, max_q_words, embed_size), name='input_q_placeholder')  
	input_c_placeholder = tf.placeholder(tf.float32, shape = (batch_size, max_c_words, embed_size), name='input_c_placeholder')
	start_answer_placeholder = tf.placeholder(tf.int32, shape = (batch_size, max_c_words), name='start_answer_placeholder')
	end_answer_placeholder = tf.placeholder(tf.int32, shape = (batch_size, max_c_words), name='end_answer_placeholder')
	mask_q_placeholder = tf.placeholder(tf.bool,  shape = (batch_size, max_q_words), name='mask_q_placeholder')
	mask_c_placeholder = tf.placeholder(tf.bool,  shape = (batch_size, max_c_words), name='mask_q_placeholder')
	"""
	

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

	print("input placeholder shape")
	print(input_q_placeholder.get_shape())
	print(normed_q.get_shape())

	
	matrix = tf.matmul(normed_q, normed_c, transpose_b = True)

	print(matrix.get_shape())

	attention = tf.reduce_max(matrix, axis = 1)
	print(attention.get_shape())

	attention = tf.expand_dims(attention, axis = 2)
	print(attention.get_shape())
	
	filtered_c = input_c_placeholder*attention
	print(filtered_c.get_shape())

	
	###SEGFAULT HAPPENS AFTER HERE	
		
	print("Building context representation")
	#These two lines gave segfaults. It was caused by time_major being set to true when it should have been false

	print("Mask Length Shape")
	print(srclen_q.get_shape())

	outputs_q, _ = tf.nn.bidirectional_dynamic_rnn(LSTM_cell_q, LSTM_cell_q, input_q_placeholder, srclen_q, scope = scope_q, time_major = False, dtype = tf.float32)
	outputs_c, _ = tf.nn.bidirectional_dynamic_rnn(LSTM_cell_c, LSTM_cell_c, filtered_c, srclen_c, scope = scope_c, time_major = False, dtype = tf.float32)

	print("concat shapes. Why are they still length 100? Because our hidden layer output is size 50 ")
	#####SEGFAULT HAPPENS BEFORE HERE
	q_output_concat = tf.concat(2, outputs_q)

	print(q_output_concat.get_shape())
	c_output_concat = tf.concat(2, outputs_c)

	print(c_output_concat.get_shape())

	print("decoding")
	
	print("This isn't a good way to handle masking. Unless the fact that these are outputs of a masked LSTM takes care of that. It might.")
	combined_q_and_c = tf.concat(1, (q_output_concat, c_output_concat))
	print(combined_q_and_c.get_shape())
	
	#print("Checking to see if segfault is passed")
	#avgLoss =  tf.reduce_mean(q_output_concat)
	

	val, state = tf.nn.dynamic_rnn(cell = LSTM_cell_decode, inputs = combined_q_and_c, dtype=tf.float32, scope = scope_decode)
	val = tf.transpose(val, [1, 0, 2])  #reshape value 
	last_output = tf.gather(val, int(val.get_shape()[0]) - 1)  #take only value from last input to LSTM

	print(last_output.get_shape())


	W_s = tf.Variable(tf.truncated_normal([hidden_size, max_c_words]))
	b_s = tf.Variable(tf.constant(0.1, shape=[max_c_words]))

	W_e = tf.Variable(tf.truncated_normal([hidden_size, max_c_words]))
	b_e = tf.Variable(tf.constant(0.1, shape=[max_c_words]))


	p_s_true = tf.nn.softmax(tf.matmul(last_output, W_s)+b_s)    #starting place probabilities
	print("shape of p_s_true")
	print(p_s_true.get_shape())

	p_s_false = 1-p_s_true
	p_s = tf.pack([p_s_false, p_s_true], axis = 2)
	print("shape of p_s")
	print(p_s.get_shape())
	

	p_e_true = tf.nn.softmax(tf.matmul(last_output, W_e)+b_e)    #ending place probabilities
	p_e_false = 1-p_e_true
	p_e = tf.pack([p_e_false, p_e_true], axis = 2)

	

	a_s = tf.argmax(p_s_true, axis=1)
	a_e = tf.argmax(p_e_true, axis=1)
	a_e = tf.maximum(a_s, a_e)
	print("shape of a_s")
	print(a_s.get_shape())

	

	print("label shape")
	print(start_answer_placeholder.get_shape())
	l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = p_s, labels = start_answer_placeholder)
	l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = p_e, labels = end_answer_placeholder)
	loss = l1+l2
	print("shape of loss")
	print(loss.get_shape())
	
	loss_by_question = tf.reduce_sum(loss, axis = 1)
	print("shape of loss_by_question")
	print(loss_by_question.get_shape())

	avgLoss = tf.reduce_mean(loss_by_question)
	#print("shape of avgLoss")
	#print(avgLoss.get_shape())

	

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
			_, batch_error = sess.run([train_step, avgLoss], feed_dict={input_q_placeholder: batch_q, input_c_placeholder: batch_c, start_answer_placeholder: start_batch, end_answer_placeholder: end_batch, mask_q_placeholder: batch_mask_q, mask_c_placeholder: batch_mask_c})			
			batch_error_total += float(batch_error)/num_batches
		avg_train_error = batch_error_total
		print("epoch %d has average training error of %f" % (j+1, avg_train_error))
		print("saving")
		saver.save(sess, 'my-model')

		

		print("Evaluating EM and f1 scores")

		test_samples = np.minimum(num_entries, batch_size)
		
		em = 0
		f1 = 0
		num_pred = 0
		num_true = 0
		num_same = 0

		test_q = embedded_q[0:test_samples]
		test_c = embedded_c[0:test_samples]
		batch_mask_q = masks_q[0:test_samples]
		batch_mask_c = masks_c[0:test_samples]

		start_pred, end_pred = sess.run([a_s, a_e], feed_dict={input_q_placeholder: test_q, input_c_placeholder: test_c, mask_q_placeholder: batch_mask_q, mask_c_placeholder: batch_mask_c})

		em = 0
		f1 = 0
		their_f1 = []
		their_em = []

		for i in range(test_samples):
			#Their Evaluation
			p = c_text[i]
			#print(p)
			#print("length p = ", len(p))
			#p = np.asarray(p)
			#print("start pred = ", start_pred[i])
			#print("end pred = ", end_pred[i])
			#start = np.minimum(start_pred[i], len(p)-1)
			#end = np.minimum(end_pred[i], len(p)-1)
			answer = p[start_pred[i]: end_pred[i]+1]
			answer = ' '.join(answer)
			#print("predicted answer = ", answer)
			#print("start true = ", start_true[i])
			#print("end true = ", end_true[i])
			true_answer = p[start_true[i]: end_true[i]+1]
			true_answer = ' '.join(true_answer)
			#print("true answer = ", true_answer)
			their_f1.append(f1_score(answer, true_answer))
			em_one = exact_match_score(answer, true_answer)
			their_em.append(em_one)
			#if (em_one == 1):
			
			#print(start_true[i])
			#print("My answer: ", answer)
			#print("True answer: ", true_answer)
			
			#My evaluation
			if ((start_pred[i] == start_true[i]) and (end_pred[i] == end_true[i])):
				em = em + 1/test_samples
			if ((start_pred[i] > end_true[i]) or (start_true[i] > end_pred[i])):
				overlap = 0
			else:
				overlap_1 = end_true[i] - start_pred[i] + 1
				overlap_2 = end_pred[i] - start_true[i] + 1
				overlap = np.min([overlap_1, overlap_2])
			num_pred = num_pred + end_pred[i] - start_pred[i] + 1
			num_true = num_true + end_true[i] - start_true[i] + 1
			num_same = num_same + overlap
		
		precision = num_same/num_pred
		recall = num_same/num_true
		if (num_same != 0):
			#print("precision + recall = ", precision+recall)
			f1 = 2*precision*recall/(precision + recall)

		their_f1_score = np.average(their_f1)
		their_em_score = np.average(their_em)
		print("Their f1 score: ", their_f1_score, " em score: ", their_em_score, " on ", test_samples, " samples")
		print("My f1 score: ", f1, " em score: ", em, " on ", test_samples, " samples")

		
		
	
	#getScores(embedded_q, embedded_c, sess, masks_q, masks_c)

#def getScores(embedded_q, embedded_c, sess, masks_q, masks_c):
	

"""

def encode_w_attn(self, inputs, masks, prev_states, scope = "", reuse = False):
	self.attn_cell = GRUAttnCell(self.size, prev_states)
	with vs.variable_scope(scope, reuse):
		o, _ = tf.nn.dynamic_rnn(self.attn_cell, inputs)

class GRUAttnCell():
	def __init__(self, num_units, encoder_output, scope=None):
		self.hs = encoder_output
		super(GRUCellAttn, self).__init__(num_units)

	def __call__(self, inputs, state, scope = None):
		gru_out, gru_state = super(GRUCellAttn, self).__call__(inputs, state, scope)
		with vs.variable_Score(scope or type(self).__name__):
			with vs.variable_scope("Attn"):
				ht = rnn_cell._linear(gru_out, self._num_units, True, 1.0)
				ht = tf.expand_dims(ht, axis = 1)
			scores = tf.reduce_sum(self.hs*ht, reduction_indices=2, keep_dims = True)
			scores = tf.exp(scores - tf.reduce_max(scores, reduction_indices=1, keep_dims=True))
			scores = scores / (1e-6 + tf.reduce_sum(scores, reduction_indices=1, keep_dims=True))
			context = tf.reduce_sum(self.hs*scores, reduction_indices=1)
			with vs.variable_score("AttnConcat"):
				out = tf.nn.relu(rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))
		return(out, out)
		"""

load_train_dir = "./train"
save_train_dir = "./train"
run(load_train_dir, save_train_dir)
