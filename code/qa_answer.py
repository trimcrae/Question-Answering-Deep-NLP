from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder, pad_sequences
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data


def generate_answers(sess, dataset, rev_vocab, train_dir, embed_path):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    #ckpt = tf.train.get_checkpoint_state(train_dir)
    #v2_path = ckpt.model_checkpoint_path + ".index"
    #logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    #model.saver.restore(session, ckpt.model_checkpoint_path)

    print(len(rev_vocab))
    print(type(rev_vocab))
    print(rev_vocab[0])
    print(type(rev_vocab[0]))


    max_q_words = 60 # longest sequence to parse
    max_c_words = 250

    batch_size = 70
    embed_size = 100
    

    new_saver = tf.train.import_meta_graph('./train/my-model-final2.meta')
    print(tf.train.latest_checkpoint('./'))
    new_saver.restore(sess, tf.train.latest_checkpoint('./train'))
    logging.info("Reading model parameters from my-model-final2.meta")
    a_e = tf.get_collection('vars2')[0]
    a_s = tf.get_collection('vars2')[1]
    input_q_placeholder = tf.get_collection('vars2')[2]
    input_c_placeholder = tf.get_collection('vars2')[3]
    mask_q_placeholder = tf.get_collection('vars2')[4]
    mask_c_placeholder = tf.get_collection('vars2')[5]
    

    context_data, question_data, question_uuid_data = dataset

    print("num uuids")
    print(len(question_uuid_data))
    print("uuid example")
    print(question_uuid_data[0])


    num_questions = int(len(context_data)) # ? ? ? remove the /50 when running for real
    print("num_questions = ", num_questions)

    
    num_batches = int(num_questions/batch_size)


    question_int = []
    context_int = []
    for i in range (num_questions):
        q1 = question_data[i]
        q1 = q1.split(" ")
        q1_int = [int(x) for x in q1]
        question_int.append(q1_int)
        c1 = context_data[i]
        c1 = c1.split(" ")
        c1_int = [int(x) for x in c1]
        context_int.append(c1_int)


    pretrained_embeddings= np.load(embed_path)
    pretrained_embeddings = tf.constant(pretrained_embeddings.f.glove) 
    

    padded_q_inputs, masks_q = zip(*pad_sequences(data = question_int, max_length = max_q_words))  #possible this will exceed memory allotment with unseen error and require batching
    padded_c_inputs, masks_c = zip(*pad_sequences(data = context_int, max_length = max_c_words))

    print("embedding")

    embedded_q = tf.nn.embedding_lookup(pretrained_embeddings, padded_q_inputs).eval(session = sess)
    embedded_c = (tf.nn.embedding_lookup(pretrained_embeddings, padded_c_inputs).eval(session = sess))
    
    answers = []
    start_index = []
    end_index = []

    for i in range(num_batches):  
        if (i%100 == 0):
            print("Batch # ", i ," of ", num_batches)
        batch_mask_q = masks_q[i*batch_size:(i+1)*batch_size]
        batch_mask_c = masks_c[i*batch_size:(i+1)*batch_size]
        batch_q = embedded_q[i*batch_size:(i+1)*batch_size]  
        batch_c = embedded_c[i*batch_size:(i+1)*batch_size]
        
        a_s_, a_e_ = sess.run([a_s, a_e], feed_dict={input_q_placeholder: batch_q, input_c_placeholder: batch_c, mask_q_placeholder: batch_mask_q, mask_c_placeholder: batch_mask_c})
        start_index.extend(a_s_)    #was append but extend makes more sense
        end_index.extend(a_e_)


    #print(start_index)
    #print(len(start_index))

    answer_list = []

    #print("num_questions: ", num_questions)
    #print("num end_indexes: ", len(end_index))
    #print("num start_indexes: ", len(start_index))

    answers = {}
    print('here 1')

    for j in range(len(end_index)):
        #print("length of context: ", len(context_int[j]))
        #print("end_index: ", end_index[j])
        answer_ids = context_int[j][start_index[j] : end_index[j]+1]
        #print(answer_ids)
        answer_words = []
        for i in range(len(answer_ids)):
            answer_words.append(rev_vocab[answer_ids[i]])
        answer_words = ' '.join(answer_words)
        #answer_list.append(answer_words)
        uuid = question_uuid_data[j]
        answers[uuid] = answer_words
    #print(len(answer_list))
    #print(answer_list[0])
    #print(answer_words)
    #print(type(answer_words))

    print('here 2')

    #for loop finding the answer in each context based on each start and end index
    #print(answers[question_uuid_data[0]])

    return answers


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def main(_):

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    context_data, question_data, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)
    dataset = (context_data, question_data, question_uuid_data)

    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    #encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    #decoder = Decoder(output_size=FLAGS.output_size)

    #qa = QASystem(encoder, decoder)

    with tf.Session() as sess:
        train_dir = get_normalized_train_dir(FLAGS.train_dir)
        #initialize_model(sess, qa, train_dir)
        answers = generate_answers(sess, dataset, rev_vocab, train_dir, embed_path)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()
