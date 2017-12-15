from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from procedural_model import run
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)


#not sure what this does but I think a lot of it needs to be taylored to what I'm doing
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.") # 10x the rate in the multiperspective matching paper and .1x the lr given by default
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.") #default was 10 bu I've heard 5 is standard
tf.app.flags.DEFINE_float("dropout", 0.2, "Fraction of units randomly dropped on non-recurrent connections.")#default is .15 but paper used .2
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")# 10 was default but apparantly azure has high memery so going higher is good
tf.app.flags.DEFINE_integer("epochs", 5, "Number of epochs to train.")#they say we should be running for 5-7 epochs. Keep it at 5 till you know it's your best model. IT should improve most after 1 anyway
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.") #not sure what to do with this. Maybe I should leave it ? 
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.") #not sure what this means ? 
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")#specified in qa_data.py. 50 is default there but get_started.sh tells it to do 100
#the rest of this I'm just leaving as is so it should be fine
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

FLAGS = tf.app.flags.FLAGS

#I think this is what loads or creates the model. Not sure. May be good as is ?
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

#Looks like this should be good as is. I think it takes in a known list of vocab ?
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

#More stuff for saving. Should be good as is ?
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

    # what does this mean? vvv ? ? ? Is it already done for me?
    # Do what you need to load datasets from FLAGS.data_dir
    """
    dataset = None
    
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)    #calls __init__ from encoder and decoder functions in qa_model
    decoder = Decoder(output_size=FLAGS.output_size)

    #runs the QASystem __init__ function in qa_model.py
    #qa = QASystem(encoder, decoder)


    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        #initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        #qa.train(sess, dataset, save_train_dir)

        #qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)
    """
    #load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
    print("It actually worked!")
    load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
    save_train_dir  = get_normalized_train_dir(FLAGS.train_dir)
    #save_train_dir = get_normalized_train_dir(FLAGS.train_dir)

    run(load_train_dir, save_train_dir)

if __name__ == "__main__":
    tf.app.run()
