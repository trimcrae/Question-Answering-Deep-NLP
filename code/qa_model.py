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

from qa_data import PAD_ID

logger = logging.getLogger("hw4")
logging.basicConfig(level=logging.INFO)


#Copied this from assignment 3 q2_rnn. Not sure if I need the whole thing here. It may be addressed in the top of train.py
class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """

    max_q_words = 25 # longest sequence to parse
    max_c_words = 250
    embed_size = 100 #specified in qa_data.py. 50 is default. get_started.sh uses 100
    #Don't think I need these max_lengths as long as words and word vectors are always stored in lists instead of concatenated
    #max_q_length = max_q_words*embed_size
    #max_c_length = max_c_words*embed_size
    # dropout = 0.2 #not using yet but hte .2 comes from the paper on multi-perspective matching
    hidden_size = 25 #about halfway between the embedding size and the eventual output size (2)
    batch_size = 64 #32 was default but apparantly azure has high memery so going higher is good
    n_epochs = 5    #they say we should be running for 5-7 epochs. Keep it at 5 till you know it's your best model. IT should improve most after 1 anyway
    #not sure if I need max_grad_norm or lr here
    #max_grad_norm = 10.
    #lr = 0.001

    #not sure if I need this at all. It just came when I copied the config class over
    def __init__(self, args):
        Cell = args.cell

        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(Cell, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        Conll_output = self.output_path + "{}_predictions.conll".format(Cell)
        self.log_output = self.output_path + "log"

    #copied from the  q2_rnn.py. This can be called seperately for both the context and the question. Not sure from where to call it. ? 
def pad_sequences(data, max_length):
    """Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.
    """
    #print(type(data))
    #print(data)



    #paddings = [0,]
    #padded_q_inputs = tf.pad(tensor = embedded_q_inputs, paddings = paddings)
    
    ret = []
    zero_vector = PAD_ID #"<pad>"#PAD_ID #* Config.embed_size   #is this right? Makes a zero vector with the same length as the usual embedding vector
    print("Starting Padding")
    counter = 0

    for sentence in data:
        counter +=1
        if (counter%10000==0):
            print("Padding sentence # ", counter)
        sentence1 = sentence[:]
        #sentence1 = sentence1.split()
        mask = []
        num_words = len(sentence1)

        if (num_words>=max_length):
            sentence1 = sentence1[0:max_length]
        else:
            for i in range(num_words, max_length):
                sentence1.append(zero_vector) 
        for i in range(max_length):
            if (i<num_words):
                mask.append(True)
            else:
                mask.append(False)
        ret.append((sentence1,  mask))
    return ret

def my_f1_score(s_pred, e_pred, s_true, e_true):

    f1 = 0  
    if ((s_pred > e_true) or (s_true > e_pred)):
        overlap = 0
    else:
        overlap_1 = e_true - s_pred + 1
        overlap_2 = e_pred - s_true + 1
        overlap = np.min([overlap_1, overlap_2])
        num_pred = e_pred - s_pred + 1
        num_true = e_true - s_true + 1
        num_same = overlap  
        precision = num_same/num_pred
        recall = num_same/num_true
        if (num_same != 0):
            f1 = 2*precision*recall/(precision + recall)

    return f1

def my_em_score(s_pred, e_pred, s_true, e_true):
    em = 0
    if ((s_pred == s_true) and (e_pred == e_true)):
        em = 1
    return em    

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, q_inputs, c_inputs, masks_q, masks_c):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        print("inside encode")

        num_qs = len(masks_q)
        q_length = np.zeros(num_qs)
        num_cs = len(masks_c)
        c_length = np.zeros(num_cs)

        for i in range (num_qs):
            q_length[i] = np.sum(masks_q[i])
            c_length[i] = np.sum(masks_c[i])

        q_length = np.sum(masks_q)
        print(q_length)
        c_length = np.sum(masks_c)

        #print("past masking")

        #this for loop may be taking too long. Maybe see if there is a way to do this without a loop
        #for this to work, q_inputs must be a list/tuple
        #in the paper this step comes first and in the handout this comes second. I can chose either and I like it better first
        #if I want to switch it back, the second paper in the handout describes this in detail
        
        x_q, x_c = add_embedding()

        c_filtered = tf.zeros(Config.max_c_words)
        for j in range(Config.max_c_words):
            r = tf.zeros(Config.max_q_words)
            for i in range(Config.max_q_words):
                r[i] = (tf.transpose(q_inputs[i])*c_inputs[j])/(tf.norm(q_inputs[i])*tf.norm(c_inputs[j]))
            r_max = max(r)
            c_filtered[j] = r_max*c_inputs[j]
        
        LSTM_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)

        q_outputs, q_output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = LSTM_cell, cell_bw = LSTM_cell, inputs = q_inputs, state_is_tuple=True, sequence_length = q_length)
        q_output_concat = tf.concat(q_output_states, 2)

        c_outputs, c_output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = LSTM_cell, cell_bw = LSTM_cell, inputs = c_filtered, state_is_tuple=True, sequence_length = c_length)
        c_output_concat = tf.concat(c_output_states, 2)

        #returns seperate variables for the question and context representations
        """
        Cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        (fw_out_q, bw_out_q), _ = tf.nn.bidirectional_dynamic_rnn(Cell, Cell, q_inputs, Config.max_q_words, scope = scope, time_major = true, dtype = dtypes.float32)
        """

        return q_output_concat, c_output_concat


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, q_output_concat, c_output_concat):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        #should this whole method be in train?
        #Where is this trained?
        #Need to incorporate knowledge_rep into inputs
        #maybe a lot os this stuff should go somewhere else and this method should only apply operations

        test = 69   #setting this at random and I'll see what it should be when it eventually disagrees with another tensor

        inputs_encoded = tf.placeholder(tf.float32, [None, test]) #need size of inputs from encoder
        labels = tf.placeholder(tf.float32, [None, 250]) #one probability for each word in the paragraph  

        cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True)
        #these shouldn't diverge this early
        val_q, state_q = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        val_q = tf.transpose(val, [1, 0, 2])  #reshape value 
        last_output_q = tf.gather(val, int(val.get_shape()[0]) - 1)  #take only value from last input to LSTM

        val_c, state_c = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        val_c = tf.transpose(val, [1, 0, 2])  #reshape value 
        last_output_c = tf.gather(val, int(val.get_shape()[0]) - 1)  #take only value from last input to LSTM

        W_s = tf.Variable(tf.truncated_normal([self.hidden_size, int(labels.get_shape()[1])]))
        b_s = tf.Variable(tf.constant(0.1, shape=[labels.get_shape()[1]]))

        W_e = tf.Variable(tf.truncated_normal([self.hidden_size, int(labels.get_shape()[1])]))
        b_e = tf.Variable(tf.constant(0.1, shape=[labels.get_shape()[1]]))

        p_s = tf.nn.softmax(tf.matmul(last_output_q, W_s)+b_s)    #starting place probabilities
        p_e = tf.nn.softmax(tf.matmul(last_output_c, W_e)+b_e)    #ending place probabilities


        return p_s, p_e

class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py         Constructed just means it called the functions from there and passed it in
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        print("In __init__ for QASystem")
        self.encoder = encoder
        self.decoder = decoder

        # ==== set up placeholder tokens ========
        
        #putting there here in case they should be here instead like the header says

        self.input_q_placeholder = tf.placeholder(tf.int32, shape = (None, Config.max_q_words))  
        self.input_c_placeholder = tf.placeholder(tf.int32, shape = (None, Config.max_c_words))
        self.mask_q_placeholder = tf.placeholder(tf.bool,  shape = (None, Config.max_q_words), name='mask_placeholder')
        self.mask_c_placeholder = tf.placeholder(tf.bool,  shape = (None, Config.max_c_words), name='mask_placeholder')
        #should inputs_encoded even be a placeholder? I think it's just calculated from the encoder. Depends on how seperate encoder and decoder are
        #self.inputs_encoded_placeholder = tf.placeholder(tf.float32, [None, ???]) #need size of inputs from encoder. 
        self.start_placeholder = tf.placeholder(tf.int32, shape = (None, Config.max_c_words)) #One number for start word and one for end word. Not anymore
        self.end_placeholder = tf.placeholder(tf.int32, shape = (None, Config.max_c_words))

        # ==== assemble pieces ====
        #not sure what this is for. I guess it just initializes all the functions I wrote
        #should I have global_variable_initializer somewhere?
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.loss = self.setup_loss() 
            self.train_op = self.add_training_op(self.loss)


        # ==== set up training/updating procedure ====
        
        self.train_step = get_optimizer("adam").minimize(self.loss)

        #copied this from build function in assignment3 model.py
        #self.train_op = self.add_training_op(self.loss)
        

        pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """

        #not 100% sure what this should do

        #need to use val versions of these when testing. Not sure how to implement that. ?
        print("setting up")

        text_file = open("./data/squad/train.span", "r")
        labels = text_file.read().split("\n")
        text_file.close()
 
        num_entries = int(len(labels)/10000)-1  #take out the /1000 when you run for real. ?? ?  ? ?marks are just to remind me to do that

        print("reading labels")

        start_answer = []
        end_answer = []
        for i in range(num_entries):
            nums = labels[i].split()
            start_answer.append(int(nums[0]))
            end_answer.append(int(nums[1]))

        text_file = open("./data/squad/train.ids.question", "r")
        inputs_q = text_file.read().split("\n")
        text_file.close()

        print("reading questions")
        myMatrix_q = []
        #inputs_q_ints = []#np.zeros(len(inputs_q))
        for i in range(num_entries):
            nums = inputs_q[i].split()
            myArray = []
            for j in range(len(nums)):
                myArray.append(int(nums[j]))
            #myMatrix_q.append(tf.nn.embedding_lookup(self.pretrained_embeddings, myArray))
            myMatrix_q.append(myArray)

        text_file = open("./data/squad/train.ids.context", "r")
        inputs_c = text_file.read().split("\n")
        #print(inputs_c)
        text_file.close()

        myMatrix_c = []
        for i in range(num_entries):
            nums = inputs_c[i].split()
            myArray = []
            for j in range(len(nums)):
                myArray.append(int(nums[j]))
            #myMatrix_c.append(tf.nn.embedding_lookup(self.pretrained_embeddings, myArray))
            myMatrix_c.append(myArray)

        
        embedded_q = []
        embedded_c = []

        padded_q_inputs, masks_q = zip(*pad_sequences(data = myMatrix_q, max_length = Config.max_q_words))
        padded_c_inputs, masks_c = zip(*pad_sequences(data = myMatrix_c, max_length = Config.max_c_words))

        for i in range(num_entries):
            if (i%100 == 0):
                print("Embedding question # ",i, "of", num_entries)
            embedded_q.append(tf.nn.embedding_lookup(self.pretrained_embeddings, padded_q_inputs[i]))
            embedded_c.append(tf.nn.embedding_lookup(self.pretrained_embeddings, padded_c_inputs[i]))



        #print(myMatrix_c[1])
        #print(type(myMatrix_c[1]))
        #print(len(myMatrix_c))


        #print(self.pretrained_embeddings)

        
        #embedded_q_inputs = tf.nn.embedding_lookup(self.pretrained_embeddings, myMatrix_q)    #should this be a placeholder instead?

        
        #embedded_c_inputs = tf.nn.embedding_lookup(self.pretrained_embeddings, myMatrix_c)
        print("Done Embedding")


        #Moved to train
        #padded_q_inputs, masks_q = zip(*pad_sequences(data = embedded_q_inputs, max_length = Config.max_q_words))
        #padded_c_inputs, masks_c = zip(*pad_sequences(data = embedded_c_inputs, max_length = Config.max_c_words))
        


        #where do embeddings come into play here?

        #masks embeddings
        #masks_q = 
        #masks_c = pad_sequences(data = self.embeddings_c, max_length = Config.max_c_words)

        #thing = Encoder(size, vocab_dims) 
        
        #Moved to train
        #print("encoding")
        #q_encoded, c_encoded = self.encoder.encode(q_inputs = padded_q_inputs, c_inputs = padded_c_inputs, masks_q = masks_q, masks_c = masks_c)
        #print("decoding")
        #p_s, p_e = self.decoder.decode(q_encoded, c_encoded)
        #print("finished decoding")
  


        #raise NotImplementedError("Connect all parts of your system here!")
        session = tf.Session()
        self.train(session, embedded_q, embedded_c, masks_q, masks_c, start_answer, end_answer)


        #this function was here but I added the preds argument to make it more like add_loss_op. Since removed it 
    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        #wtf does this with vs.variable_scope stuff do?
        with vs.variable_scope("loss"):
            #copied from add_loss_op function in q2_rnn.py from assignment 3
            """
            preds = tf.boolean_mask(preds, self.mask_placeholder)
            labels = tf.boolean_mask(self.labels_placeholder, self.mask_placeholder)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(preds, labels))
            """
            l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(self.a_s, self.start_answer)
            l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(self.a_e, self.end_answer)
            self.loss = l1+l2


        
        #return loss



    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """

        #what does it mean by "based on placeholder tokens"  ?
        with vs.variable_scope("embeddings"):
        #copied directly fron q2_rnn with embeddings loading from piazza suggestion
            embed_path = "./data/squad/glove.trimmed.100.npz"
            pretrained_embeddings = np.load(embed_path)
            logger.info("Keys")
            logger.info(pretrained_embeddings.keys())
            logger.info("Initialized embeddings.")
            self.pretrained_embeddings = tf.constant(pretrained_embeddings.f.glove)    
            #self.pretrained_non_tensor = pretrained_embeddings.f.glove 

        #copied from q1_window.py
    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates an embedding tensor and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_window_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_window_features * embedding_size).

        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: You can use tf.reshape to concatenate the vectors. See following link to understand
            what -1 in a shape means.
            https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.
        Returns:
            embeddings: tf.Tensor of shape (None, n_window_features*embed_size)
        """
                                                             
        embeddings_q = tf.Variable(self.pretrained_embeddings)
        embeddings_c = tf.Variable(self.pretrained_embeddings)
        embeddings_q = tf.nn.embedding_lookup(embeddings_q, self.input_q_placeholder)
        embeddings_c = tf.nn.embedding_lookup(embeddings_c, self.input_q_placeholder)
        embeddings_q = tf.reshape(embeddings_q, [-1, self.config.embed_size])     
        embeddings_c = tf.reshape(embeddings_q, [-1, self.config.embed_size])                    
                                                                                                                 
        ### END YOUR CODE
        return embeddings_q, embeddings_c


    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        #feed_dict={self.input_placeholder: inputs_batch, self.labels_placeholder: labels_batch}
        input_feed['train_x'] = train_x
        #output_feed['train_y'] = train_y

        #wtf is this for? why isn't it initialized the same way as the input feed
        output_feed = []

        output_feed['train_y'] = train_y

        outputs = session.run(output_feed, input_feed)


        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}
        input_feed['valid_x'] = valid_x

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []
        output_feed['valid_y'] = valid_y

        #outputs = session.run(output_feed, input_feed) #this line was given in the starter code
        _, cost = session.run([self.train_step, self.loss], feed_dict={x: batch_xs, y_: batch_ys})  #No idea if this is right


        return cost

    def decode(self, session, qs, cs):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        #how is this different from the other decode? This one takes the session and test_x instead of knowledge_rep
        input_feed = {}
        input_feed['qs'] = qs
        input_feed['cs'] = cs

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []
        #output_feed['test_y'] = test_y

        outputs = session.run(output_feed, input_feed)
        #yp = 
        #yp2 = 

        return outputs #yp, yp2

    def answer(self, session, qs, cs):
        #does this need me to modify anything?

        yp, yp2 = self.decode(session, qs, cs)
        #this means self.decode needs to return two probabilities, one for start and one for end

        # I believe this is picking the start and end points based on the maximum from the probabilities of the start and end words
        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)



        #what if a_s comes after a_e?  

        return a_s, a_e

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)    #this means self.test returns cost and cost only


        return valid_cost

    def evaluate_answer(self, session, qs, cs, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        print("Evaluating Answers")

        f1 = 0.
        em = 0.

        text_file = open("./data/squad/train.context", "r")
        inputs_c = text_file.read().split("\n")
        #print(inputs_c)

        context = []
        text_file.close()
        for i in range(sample):
            words = inputs_c[i].split()
            context.append(words)

        prediction = []
        #need to define self.true somewhere
        ground_truth = []

        self.a_s, self.a_e = self.answer(session, qs, cs)

        #these functions are defined in evaluate.py. They are already written and should not be changed
        #Not sure if these indices are the best way to access these
        for i in range(sample):
            prediction.append(context[i][self.a_s, self.a_e + 1])
            ground_truth.append(context[i][self.true_s, self.true_e +1])
            f1 = f1 + f1_score(prediction[i], ground_truth[i])/sample
            em = em + exact_match_score(prediction[i], ground_truth[i])/sample


        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))
            print("With Print, F1: {}, EM: {}, for {} samples".format(f1, em, sample))    #Might be redundent

        return f1, em




        #copied from assignment 3 q3_gru.py. Does this need to be called to be set up?
    def add_training_op(self, loss):
        """Sets up the training Ops.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See
        Args:
            loss: Loss tensor.
        Returns:
            train_op: The Op for training.
        """

        optimizer = get_optimizer("adam") 

        grads_and_vars = optimizer.compute_gradients(self.loss, tf.trainable_variables())
        grads, _vars = zip(*grads_and_vars)
        #grads = [x[0] for x in grads_and_vars]

        if (Config.clip_gradients == True):
            grads, global_norm = tf.clip_by_global_norm(grads, Config.max_grad_norm)    
        self.grad_norm = tf.global_norm(grads)

        grads_and_vars = zip(grads, _vars)
        train_op = optimizer.apply_gradients(grads_and_vars)  

        assert self.grad_norm is not None, "grad_norm was not set properly!"
        return train_op


    def create_feed_dict(self, inputs_batch_q, inputs_batch_c, mask_batch_q, mask_batch_c, start_batch, end_batch, dropout=1 ):
        """Creates the feed_dict """
        feed_dict={self.input_q_placeholder: inputs_batch_q, self.input_c_placeholder: inputs_batch_c, self.start_placeholder: start_batch, self.end_placeholder: end_batch, self.mask_q_placeholder: mask_batch_q, self.mask_c_placeholder: mask_batch_c}
       

         #I added train_on_batch to this myself. Taken from assignment 3 q2_gru. gru example doesnt have dropout but rnn does
    def train_on_batch(self, sess, inputs_batch_q, inputs_batch_c, mask_batch_q, mask_batch_c, start_batch, end_batch):
        feed = self.create_feed_dict(inputs_batch_q = inputs_batch_q, inputs_batch_c = inputs_batch_c, mask_batch_q =mask_batch_q, mask_batch_c = mask_batch_c, start_batch = start_batch, end_batch = end_batch)
        _, loss, grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed)
        return loss, grad_norm

    def train(self, session, embed_q, embed_c, masks_q, masks_c, start_answer, end_answer):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        train_size = len(embed_q)

        print(embed_q[1])
        qs = []
        cs = []

        print("Defining questions and contexts in this instance")
        for i in range(train_size):
            print("Defining pair # ",i)
            qs.append(embed_q[i].eval(session = session))
            cs.append(embed_c[i].eval(session = session))
            #print(qs[i])
        #cs = embed_c.eval()



        #taken from assignment 3 ner_model fit function with some help from deep_neural_net

        
        best_score = 0.
        num_batches = int(np.floor(train_size/Config.batch_size))
        for epoch in range(Config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, Config.n_epochs)
            #taken from assignment 3 q3_gru run_epoch function
            prog = Progbar(target=1 + int(train_size / Config.batch_size))
            losses, grad_norms = [], []
            batch_size = Config.batch_size
            batch_q = qs[0:batch_size]  #100 at a time
            batch_c = cs[0:batch_size]
            for i in range(num_batches):  
                batch_q = qs[i*batch_size:(i+1)*batch_size]  
                batch_c = cs[i*batch_size:(i+1)*batch_size]
                start_batch = start_answer[i*batch_size:(i+1)*batch_size]
                end_batch = end_answer[i*batch_size:(i+1)*batch_size]
                mask_batch_q = masks_q[i*batch_size:(i+1)*batch_size]
                mask_batch_c = masks_c[i*batch_size:(i+1)*batch_size]
                loss, grad_norm = self.train_on_batch(sess = session, inputs_batch_q = batch_q, inputs_batch_c = batch_c, mask_batch_q = mask_batch_q, mask_batch_c = mask_batch_c, start_batch = start_batch, end_batch = end_batch)
                grad_norms.append(grad_norm)
                prog.update(i + 1, [("train loss", loss)])
            f1, em = self.evaluate_answer(session = session, qs=batch_q, cs = batch_c)
            grad = tf.global_norm(grad_norms)
            logger.info("Gradient norm is %d", grad)
            if f1 > best_score:
                best_score = f1
                if saver:
                    logger.info("New best score! Saving model in %s", Config.model_output)
                    saver.save(sess, Config.model_output)
            print("")
            #need to either make self.report function or have this happen at all times.
            if self.report:
                self.report.log_epoch()
                self.report.save()


        #looking at F1 and EM in addition to cost
        #Make sure this runs on devset as well
        

        #implement learning rate annealing (look into tf.train.exponential_decay) to save time

        #need to make Config.batch_size


        #Should this come before or after the loop?

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        return best_score   #should this be here?
        #save model