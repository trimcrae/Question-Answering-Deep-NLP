from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score

from qa_data import PAD_ID




boolarr = np.array([[0, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=np.bool)
print(boolarr)
print(np.sum(boolarr))
