import math
import numpy as np
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import EmbeddingWrapper
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import OutputProjectionWrapper
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
import tensorflow as tf

eps = 1e-8

class Model(object):
  def __init__(self, imgSize, vocabSize, embedSize, use_lstm, rnnHiddenSize, rnnLayers, pad, start, batch_size,
    learning_rate, learning_rate_decay_factor, min_learning_rate, training_steps_per_epoch, 
    keep_prob=0.5, max_gradient_norm=5.0, is_training=True):

    if is_training:
      self.global_step = tf.Variable(0, trainable=False)
      self.learning_rate = tf.maximum(
				tf.train.exponential_decay(
					learning_rate,
					self.global_step,
					training_steps_per_epoch,
					learning_rate_decay_factor,
					staircase=True),
        min_learning_rate)
      self.answer_targets_ph = tf.placeholder(tf.int32, shape=[batch_size, 10, 21], name="answer_targets")
      START = tf.constant(value=[start]*batch_size)
    else:
      self.options_ph = tf.placeholder(tf.int32, shape=[batch_size, 10, 100, 20], name="options")
      self.option_targets_ph = tf.placeholder(tf.int32, shape=[batch_size, 10, 100, 21], name="option_targets")
      START = tf.constant(value=[start]*batch_size*100)

    self.image_feature_ph = tf.placeholder(tf.float32, shape=[batch_size, imgSize], name="image_feature")
    
    self.caption_ph = tf.placeholder(tf.int32, shape=[batch_size, 40], name="caption")
    self.caption_length_ph = tf.placeholder(tf.int32, shape=[batch_size], name="caption_length")

    self.questions_ph = tf.placeholder(tf.int32, shape=[batch_size, 10, 20], name="questions")
    self.question_lengths_ph = tf.placeholder(tf.int32, shape=[batch_size, 10], name="question_lengths")

    self.answers_ph = tf.placeholder(tf.int32, shape=[batch_size, 10, 20], name="answers")
    self.answer_lengths_ph = tf.placeholder(tf.int32, shape=[batch_size, 10], name="answer_lengths")

    # Embedding (share)
    with ops.device("/cpu:0"):
      if vs.get_variable_scope().initializer:
        initializer = vs.get_variable_scope().initializer
      else:
        # Default initializer for embeddings should have variance=1.
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)
        embedding = vs.get_variable("embedding", [vocabSize, embedSize], initializer=initializer, dtype=tf.float32)

    START_EMB = embedding_ops.embedding_lookup(embedding, START)

    # split placeholders and embed
    questions = tf.split(value=self.questions_ph, num_or_size_splits=10, axis=1)                # list with length 10; questions[0]: [batch_size, 1, 20]
    questions = [tf.squeeze(input=question, axis=1) for question in questions]                  # list with length 10; questions[0]: [batch_size, 20]
    questions = [embedding_ops.embedding_lookup(embedding, question) for question in questions] # list with length 10; questions[0]: [batch_size, 20, embedSize]

    question_lengths = tf.split(value=self.question_lengths_ph, num_or_size_splits=10, axis=1)
    question_lengths = [tf.squeeze(question_length) for question_length in question_lengths]

    answers = tf.split(value=self.answers_ph, num_or_size_splits=10, axis=1)
    answers = [tf.squeeze(input=answer, axis=1) for answer in answers]
    answers = [embedding_ops.embedding_lookup(embedding, answer) for answer in answers]

    answer_lengths = tf.split(value=self.answer_lengths_ph, num_or_size_splits=10, axis=1)
    answer_lengths = [tf.squeeze(answer_length) for answer_length in answer_lengths]

    if is_training:
      answer_targets = tf.split(value=self.answer_targets_ph, num_or_size_splits=10, axis=1)
      answer_targets = [tf.squeeze(input=answer_target, axis=1) for answer_target in answer_targets]
    else:
      options = tf.split(value=self.options_ph, num_or_size_splits=10, axis=1)                    # list with length 10; options[0]: [batch_size, 1, 100, 20]
      options = [tf.reshape(tensor=option, shape=[-1, 100, 20]) for option in options]
      options = [embedding_ops.embedding_lookup(embedding, option) for option in options]

      option_targets = tf.split(value=self.option_targets_ph, num_or_size_splits=10, axis=1)      # list with length 10; option_targets[0]: [batch_size, 1, 100, 21]
      option_targets = [tf.reshape(tensor=option_target, shape=[-1, 100, 21]) for option_target in option_targets]

    # make RNN cell
    def single_cell():
      return GRUCell(rnnHiddenSize)
    if use_lstm:
      def single_cell():
        return BasicLSTMCell(rnnHiddenSize, state_is_tuple=False)

    make_cell = single_cell
    if rnnLayers > 1:
      def make_cell():
        return MultiRNNCell([single_cell() for _ in range(rnnLayers)], state_is_tuple=False)

    encoder_cell = make_cell()
    decoder_cell = OutputProjectionWrapper(cell=make_cell(), output_size=vocabSize, activation=None)

    # caption feature
    caption = embedding_ops.embedding_lookup(embedding, self.caption_ph)                        # [batch_size, 40, embedSize]
    caption_length = tf.squeeze(self.caption_length_ph)
    with tf.variable_scope('EncoderRNN') as varscope:
      _, captionState = dynamic_rnn(cell=encoder_cell, inputs=caption, sequence_length=caption_length, dtype=tf.float32, scope=varscope)       # [batch_size, encoder_cell.state_size]

    if is_training:
      losses = []
    else:
      ans_word_probs = []

    for r in range(10):
      # 1. question
      with tf.variable_scope('EncoderRNN', reuse=True) as varscope:
        _, questionState = dynamic_rnn(cell=encoder_cell, inputs=questions[r], sequence_length=question_lengths[r], dtype=tf.float32, scope=varscope)
      
      # 2. history
      if r == 0:
        historyState = captionState
      
      # 3. fusion
      concat = tf.concat(values=[self.image_feature_ph, questionState, historyState], axis=1)
      if is_training:
        concat = tf.nn.dropout(x=concat, keep_prob=keep_prob)
      with tf.variable_scope('Fusion', reuse=(r>0)) as varscope:
        encoder_state = tf.contrib.layers.fully_connected(inputs=concat, num_outputs=decoder_cell.state_size, activation_fn=tf.nn.tanh, scope=varscope)

      # 4. decoder
      with tf.variable_scope('DecoderRNN', reuse=(r>0)) as varscope:
        if is_training:
          answer = [tf.squeeze(input=word, axis=1) for word in tf.split(value=answers[r], num_or_size_splits=20, axis=1)]
          decoder_outputs, _ = rnn_decoder(decoder_inputs=[START_EMB]+answer, initial_state=encoder_state, cell=decoder_cell, loop_function=None, scope=varscope) # [batch_size, 21, vocabSize]
        else:
          encoder_state = tf.reshape(tensor=tf.stack(values=[encoder_state]*100, axis=1), shape=[batch_size*100, decoder_cell.state_size])
          option = tf.reshape(tensor=options[r], shape=[-1, 20, embedSize])
          option = [tf.squeeze(input=word, axis=1) for word in tf.split(value=option, num_or_size_splits=20, axis=1)]
          decoder_outputs, _ = rnn_decoder(decoder_inputs=[START_EMB]+option, initial_state=encoder_state, cell=decoder_cell, loop_function=None, scope=varscope) # [batch_size*100, 21, vocabSize]
        decoder_outputs = [tf.log(tf.nn.softmax(decoder_output)+eps) for decoder_output in decoder_outputs]
        decoder_outputs = tf.stack(values=decoder_outputs, axis=1)

      # 5. update history
      with tf.variable_scope('EncoderRNN', reuse=True) as varscope:
        _, historyState = dynamic_rnn(cell=encoder_cell, inputs=questions[r], sequence_length=question_lengths[r], initial_state=historyState, scope=varscope)
        _, historyState = dynamic_rnn(cell=encoder_cell, inputs=answers[r], sequence_length=answer_lengths[r], initial_state=historyState, scope=varscope)
     
      # 6. loss or ans_word_prob
      if is_training:
        # negative log likelihood loss
        target_one_hot = tf.one_hot(indices=answer_targets[r], depth=vocabSize, dtype=tf.float32)
        mask = tf.cast(x=tf.not_equal(x=answer_targets[r], y=pad), dtype=tf.float32)
        logprob = tf.reduce_sum(input_tensor=decoder_outputs*target_one_hot, axis=2)
        loss = -tf.reduce_sum(input_tensor=logprob*mask, axis=1)/tf.reduce_sum(input_tensor=mask, axis=1)
        losses.append(loss)
      else:
        option_target = tf.reshape(tensor=option_targets[r], shape=[-1, 21])
        target_one_hot = tf.one_hot(indices=option_target, depth=vocabSize, dtype=tf.float32)
        mask = tf.cast(x=tf.not_equal(x=option_target, y=pad), dtype=tf.float32)
        logprob = tf.reduce_sum(input_tensor=decoder_outputs*tf.expand_dims(input=mask, axis=2)*target_one_hot,axis=[1,2])
        logprob = tf.reshape(tensor=logprob, shape=[-1, 100])                                   # [batch_size, 100]
        ans_word_probs.append(logprob)
    if is_training:
      losses = tf.stack(values=losses, axis=1)                                                  # [batch_size, 10]
      self.loss = tf.reduce_mean(losses)
      params = tf.trainable_variables()
      gradients = tf.gradients(self.loss, params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.opt_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
    else:
      self.ans_word_probs = tf.stack(values=ans_word_probs, axis=1)                             # [batch_size, 10, 21, vocabSize]

    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)

#Model(imgSize=1024, vocabSize=8848, embedSize=300, use_lstm=True, rnnHiddenSize=256, rnnLayers=1, pad=0, start=1, batch_size=32,learning_rate=1e-4, learning_rate_decay_factor=0.9, min_learning_rate=1e-5, training_steps_per_epoch=2, keep_prob=0.5, max_gradient_norm=5.0, is_training=True)

