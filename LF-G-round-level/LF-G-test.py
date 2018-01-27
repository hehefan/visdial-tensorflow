import os
import sys
import tensorflow as tf

import random
import math
import numpy as np
from opts import args
from dataloader import DataLoader
from model import Model

# seed for reproducibility
tf.set_random_seed(1234)

# ==============================================================================
#                               Loading dataset
# ==============================================================================
dataloader = DataLoader(args, 'val')

args.batchSize = 10

# ==============================================================================
#                               Build Graph
# ==============================================================================

model = Model(imgSize=args.imgFeatureSize,
              vocabSize=dataloader.vocabSize,
              embedSize=args.embedSize,
              use_lstm=args.useLSTM,
              rnnHiddenSize=args.rnnHiddenSize,
              rnnLayers=args.numLayers,
              pad=dataloader.word2ind['<PAD>'],
              start=dataloader.word2ind['<START>'],
              batch_size=args.batchSize,
              learning_rate=None,
              learning_rate_decay_factor=None, 
              min_learning_rate=None, 
              training_steps_per_epoch=None, 
              keep_prob=None, 
              max_gradient_norm=None,
              is_training=False)

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5

with tf.Session(config=config) as sess:
  start_step = 5*args.ckptInterval
  idxs = [i for i in range(dataloader.num_dialogues)]
  for step in range(start_step, 100000000, args.ckptInterval):
    checkpoint_path = os.path.join(args.savePath, "visdial-%d"%step)
    if not os.path.exists(checkpoint_path+'.index'):
      exit(0)
    model.saver.restore(sess, checkpoint_path)
    mrr, rank1, rank5, rank10, mean, numRecords = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for batch_id, (start, end) in enumerate(zip(range(0, dataloader.num_dialogues, args.batchSize), range(args.batchSize, dataloader.num_dialogues, args.batchSize))):
      batchImgFeat, batchQuestion, batchQuestionLength, batchCaption, batchCaptionLength, batchAnswer, batchAnswerLength, batchAnswerID, batchOptions, batchOptionTargets = dataloader.getTestBatch(idxs[start:end])
      if batchAnswerID.shape[0] < args.batchSize:
        break
      input_feed = { model.image_feature_ph:batchImgFeat,
          model.questions_ph:batchQuestion,
          model.question_lengths_ph:batchQuestionLength,
          model.answers_ph:batchAnswer,
          model.answer_lengths_ph:batchAnswerLength,
          model.caption_ph:batchCaption,
          model.caption_length_ph:batchCaptionLength,
          model.options_ph:batchOptions,
          model.option_targets_ph:batchOptionTargets}
      ans_word_probs = sess.run(model.ans_word_probs, input_feed)           # [batch_size, 10, 100]
      labels = batchAnswerID
      for b in range(args.batchSize):
        for r in range(10):
          scores = ans_word_probs[b][r]
          prediction = np.argsort(scores)[::-1]
          lbl = labels[b][r]
          if prediction[0] == lbl:
            rank1 += 1
          if lbl in prediction[:5]:
            rank5 += 1
          if lbl in prediction[:10]:
            rank10 += 1
          idx = np.argwhere(prediction==lbl)[0][0] + 1.0
          mrr += 1.0/idx
          mean += idx
      numRecords += 10*args.batchSize
    mrr /= numRecords
    rank1 /= numRecords
    rank5 /= numRecords
    rank10 /= numRecords
    mean /= numRecords
    print '%5d:'%step
    print '\t%f\t%f\t%f\t%f\t%f'%(mrr, rank1, rank5, rank10, mean)
    sys.stdout.flush() 
