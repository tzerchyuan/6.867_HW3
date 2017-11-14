'''
Tested with Python 3.4.1 and Tensorflow 1.3.0
'''
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import sys
import math
import time
import matplotlib.pyplot as plt

DATA_PATH = 'art_data/'
DATA_FILE = DATA_PATH + 'art_data.pickle'
# DATA_FILE = DATA_PATH + 'augmented_art_data.pickle'
IMAGE_SIZE = 50
NUM_CHANNELS = 3
NUM_LABELS = 11
INCLUDE_TEST_SET = False

POOL1SIZE = 3
STRIDE1SIZE = 1

POOL2SIZE = 3
STRIDE2SIZE = 1

POOL3SIZE = 2
STRIDE3SIZE = 1

POOL4SIZE = 2
STRIDE4SIZE = 1

BATCH_SIZE = 10
NUM_TRAINING_STEPS = 1200
LEARNING_RATE = 0.01

L2_CONST = 0.0  # Set to > 0 to use L2 regularization
DROPOUT_RATE = 0.5  # Set to > 0 to use dropout
POOL1 = True  # Set to True to add pooling after first conv layer
POOL2 = True  # Set to True to add pooling after second conv layer
POOL3 = True
POOL4 = True
BN = True  # Set to True to use batch normalization
FILTERS = 96

class ArtistConvNet:
  def __init__(self, invariance=False):
    '''Initialize the class by loading the required datasets
    and building the graph'''
    self.load_pickled_dataset(DATA_FILE)
    self.epochs = []
    self.training_accuracies = []
    self.validation_accuracies = []
    self.invariance = invariance
    self.stopping_1 = None
    self.stopping_2 = None
    self.accuracy = None
    if invariance:
      self.load_invariance_datasets()
    self.graph = tf.Graph()
    self.build_graph()


  def build_graph(self):
    with self.graph.as_default():
      # Input data
      self.images = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
      self.labels = tf.placeholder(tf.float32, shape=(None, NUM_LABELS))
      self.training = tf.placeholder(tf.bool)

      # Network
      regularizer = tf.contrib.layers.l2_regularizer(scale=L2_CONST)

      conv1 = tf.layers.conv2d(inputs=self.images, filters=FILTERS, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)
      if POOL1:
        conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=POOL1SIZE, strides=STRIDE1SIZE)
      if BN:
        conv1 = tf.layers.batch_normalization(inputs=conv1, axis=3, training=self.training)

      conv2 = tf.layers.conv2d(inputs=conv1, filters=FILTERS, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)
      if POOL2:
        conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=POOL2SIZE, strides=STRIDE2SIZE)
      if BN:
        conv2 = tf.layers.batch_normalization(inputs=conv2, axis=3, training=self.training)

    #   conv3 = tf.layers.conv2d(inputs=conv2, filters=FILTERS, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)

      conv3 = tf.layers.conv2d(inputs=conv2, filters=FILTERS, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)
      if POOL3:
        conv3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=POOL3SIZE, strides=STRIDE3SIZE)
      if BN:
        conv3 = tf.layers.batch_normalization(inputs=conv3, axis=3, training=self.training)
      #
      conv4 = tf.layers.conv2d(inputs=conv3, filters=FILTERS, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)
      if POOL4:
        conv4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=POOL4SIZE, strides=STRIDE4SIZE)
      if BN:
        conv4 = tf.layers.batch_normalization(inputs=conv4, axis=3, training=self.training)
      #
      conv5 = tf.layers.conv2d(inputs=conv4, filters=FILTERS, kernel_size=5, strides=2, padding="same", activation=tf.nn.relu)
      if BN:
        conv5 = tf.layers.batch_normalization(inputs=conv5, axis=3, training=self.training)

      flat = tf.contrib.layers.flatten(inputs=conv4)
      flat = tf.layers.dropout(inputs=flat, rate=DROPOUT_RATE, training=self.training)

      fc1 = tf.layers.dense(inputs=flat, units=64, activation=tf.nn.relu, kernel_regularizer=regularizer)
      if BN:
        fc1 = tf.layers.batch_normalization(inputs=fc1, axis=1, training=self.training)
      fc1 = tf.layers.dropout(inputs=fc1, rate=DROPOUT_RATE, training=self.training)
      logits = tf.layers.dense(inputs=fc1, units=NUM_LABELS, activation=None, kernel_regularizer=regularizer)

      # Compute loss
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
      self.loss += tf.losses.get_regularization_loss()

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        self.optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss)

      self.preds = tf.argmax(logits, 1)
      self.acc = 100*tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.labels, 1), self.preds), dtype=tf.float32))


  def train_model(self, num_steps=NUM_TRAINING_STEPS):
    '''Train the model with minibatches in a tensorflow session'''
    with tf.Session(graph=self.graph) as session:
      session.run(tf.global_variables_initializer())
      print('Initializing variables...')

      for step in range(num_steps):
        offset = (step * BATCH_SIZE) % (self.train_Y.shape[0] - BATCH_SIZE)
        batch_data = self.train_X[offset:(offset + BATCH_SIZE), :, :, :]
        batch_labels = self.train_Y[offset:(offset + BATCH_SIZE), :]

        # Data to feed into the placeholder variables in the tensorflow graph
        feed_dict = {self.images: batch_data, self.labels: batch_labels, self.training: True}
        _, l, acc = session.run([self.optimizer, self.loss, self.acc], feed_dict=feed_dict)
        # if (step % (NUM_TRAINING_STEPS-1) == 0):
        if (step % 100 == 0):
          val_acc = session.run(self.acc, feed_dict={self.images: self.val_X, self.labels: self.val_Y, self.training: False})
          train_acc = session.run(self.acc, feed_dict={self.images: self.train_X, self.labels: self.train_Y, self.training: False})
          print('')
          print('Batch loss at step %d: %f' % (step, l))
          print('Batch training accuracy: %.1f%%' % acc)
          print('Full training accuracy: %.1f%%' % train_acc)
          self.epochs.append(step)
          self.training_accuracies.append(train_acc/100)
          self.validation_accuracies.append(val_acc/100)
          print('Validation accuracy: %.1f%%' % val_acc)
          self.accuracy = (train_acc/100, val_acc/100)

          # stopping early code
        #   if (step / 50) > PATIENCE_1:
        #       # check if the validation is higher than the last PATIENCE-th accuracy
        #       print('accuracy', val_acc/100)
        #       print('last patience acc', self.validation_accuracies[-PATIENCE_1])
        #       print('avg last patience acc', sum(self.validation_accuracies[-PATIENCE_2:])/PATIENCE_2)
          #
        #       # STOPPING 1
        #       if val_acc/100 <= self.validation_accuracies[-PATIENCE_1]:
        #           if (self.stopping_1 and self.stopping_2):
        #               return
        #           else:
        #               if self.stopping_1 == None:
        #                   print('STOPPING EARLY DUE TO STOPPING CRITERION 1')
        #                   self.stopping_1 = (step, train_acc/100, val_acc/100)
        #                   if (self.stopping_1 and self.stopping_2):
        #                     return
          #
        #       # STOPPING 2 average patience values
        #       if val_acc/100 <= sum(self.validation_accuracies[-PATIENCE_2:])/PATIENCE_2:
        #           if (self.stopping_1 and self.stopping_2):
        #               return
        #           else:
        #               if self.stopping_2 == None:
        #                   print('STOPPING EARLY DUE TO STOPPING CRITERION 2')
        #                   self.stopping_2 = (step, train_acc/100, val_acc/100)
        #                   if (self.stopping_1 and self.stopping_2):
        #                     return

        #   if val_acc/100 <= self.validation_accuracies[-1] and len(self.validation_accuracies)>0:
        #       if (self.stopping_1 and self.stopping_2 and self.stopping_3):
        #           return
        #       else:
        #           if self.stopping_3 == None:
        #               print('STOPPING EARLY DUE TO STOPPING CRITERION 3')
        #               self.stopping_3 = (step, train_acc/100, val_acc/100)
        #               if (self.stopping_1 and self.stopping_2 and self.stopping_3):
        #                 return



      # This code is for the final question
      if self.invariance:
        print("\nObtaining final results on invariance sets!")
        sets = [self.val_X, self.translated_val_X, self.bright_val_X, self.dark_val_X,
            self.high_contrast_val_X, self.low_contrast_val_X, self.flipped_val_X,
            self.inverted_val_X,]
        set_names = ['normal validation', 'translated', 'brightened', 'darkened',
               'high contrast', 'low contrast', 'flipped', 'inverted']

        for i in range(len(sets)):
          acc = session.run(self.acc, feed_dict={self.images: sets[i], self.labels: self.val_Y, self.training: False})
          print('Accuracy on', set_names[i], 'data: %.1f%%' % acc)


  def load_pickled_dataset(self, pickle_file):
    print("Loading datasets...")
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      self.train_X = save['train_data']
      self.train_Y = save['train_labels']
      self.val_X = save['val_data']
      self.val_Y = save['val_labels']

      if INCLUDE_TEST_SET:
        self.test_X = save['test_data']
        self.test_Y = save['test_labels']
      del save  # hint to help gc free up memory
    print('Training set', self.train_X.shape, self.train_Y.shape)
    print('Validation set', self.val_X.shape, self.val_Y.shape)
    if INCLUDE_TEST_SET:
      print('Test set', self.test_X.shape, self.test_Y.shape)


  def plot_accuracy(self):
      max_val = max(self.validation_accuracies)
      max_train = max(self.validation_accuracies)
    #   stopping_1_ind = self.validation_accuracies.index(self.stopping_1[1])
    #   stopping_2_ind = self.validation_accuracies.index(self.stopping_2[1])
    #   stopping_3_ind = self.validation_accuracies.index(self.stopping_3[1])
      plt.plot(self.epochs, self.training_accuracies, label='train') #end(' + ("%.3f" % self.training_accuracies[-1]) + ')')
      plt.plot(self.epochs, self.validation_accuracies, label='validation max(' + ("%.3f" % max_val) + ')')
    #   plt.plot(self.epochs, [self.stopping_1[2]]*len(self.epochs), label='stopping_1 end('+ ("%.3f" % self.stopping_1[2]) + ') epoch=' + str(self.stopping_1[0]), linestyle='--')
    #   plt.plot(self.epochs, [self.stopping_2[2]]*len(self.epochs), label='stopping_2 end('+ ("%.3f" % self.stopping_2[2]) + ') epoch=' + str(self.stopping_2[0]), linestyle=':')
    #   plt.plot(self.epochs, [self.stopping_3[2]]*len(self.epochs), label='stopping_3 end('+ ("%.3f" % self.stopping_3[2]) + ') epoch=' + str(self.stopping_3[0]), linestyle='-.')
    #   plt.axvline(x=self.stopping_1[0], linewidth=1.0, linestyle='--')
    #   plt.axvline(x=self.stopping_2[0], linewidth=1.0, linestyle=':')
    #   plt.axvline(x=self.stopping_3[0], linewidth=1.0, linestyle='-.')
    #   plt.plot(self.epochs, [max_val]*len(self.epochs))
      x1,x2,y1,y2 = plt.axis()
      plt.axis=((x1, x2, 0, 1.0))
    #   plt.axis([0, NUM_TRAINING_STEPS-1, 0, 1.0])
      plt.xlabel('number of epochs')
      plt.ylabel('accuracy')
      plt.title('Test and validation accuracy')
      plt.legend(bbox_to_anchor=(0.5, 0.25), loc=0, borderaxespad=0.01, fontsize='small')
      plt.show()


  def load_invariance_datasets(self):
    with open(DATA_PATH + 'invariance_art_data.pickle', 'rb') as f:
      save = pickle.load(f)
      self.translated_val_X = save['translated_val_data']
      self.flipped_val_X = save['flipped_val_data']
      self.inverted_val_X = save['inverted_val_data']
      self.bright_val_X = save['bright_val_data']
      self.dark_val_X = save['dark_val_data']
      self.high_contrast_val_X = save['high_contrast_val_data']
      self.low_contrast_val_X = save['low_contrast_val_data']
      del save


if __name__ == '__main__':
  invariance = False
  if len(sys.argv) > 1 and sys.argv[1] == 'invariance':
    print("Testing finished model on invariance datasets!")
    invariance = True


  # filee = open(str(POOL1) + '_' + str(POOL2) + '.txt', "w")
  # for stride in [1,2,3,4]:
  #     for f in [2,3,4,5]:
  #         STRIDE1SIZE = stride
  #         POOL1SIZE = f
  #         STRIDE2SIZE = stride
  #         POOL2SIZE = f
  #
  #         t1 = time.time()
  #         conv_net = ArtistConvNet(invariance=invariance)
  #         conv_net.train_model()
  #         t2 = time.time()
  #         filee.write('stride size = ' + str(stride) + ' filter size = ' + str(f) + ' validation_acc = ' + ("%.3f" % conv_net.accuracy[1]) + "\n")
  #         filee.write("Finished training. Total time taken: " + str(t2-t1) + "\n")
  # # conv_net.plot_accuracy()
  #
  #         print('stride size = ', stride, ', pool size = ', f, ', validation_acc = ', conv_net.accuracy[1])
  #         print("Finished training. Total time taken:", t2-t1)
  # filee.close()
  t1 = time.time()
  conv_net = ArtistConvNet(invariance=invariance)
  conv_net.train_model()
  conv_net.plot_accuracy()
  t2 = time.time()
  print("Finished training. Total time taken:", t2-t1)
