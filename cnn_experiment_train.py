#!/usr/bin/env python3.5


from datetime import datetime
import time
import os
import cnn_db_loader

import tf_utils
import cnn_tf_graphs

import tensorflow as tf
from tensorflow.contrib import learn


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('experiment_folder', '80',
													 """Directory where to write event logs """
													 """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
														"""Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
														"""Whether to log device placement.""")
tf.app.flags.DEFINE_integer('batch_size', 100,
														"""Size of a batch.""")

cnn_db_loader.NUMBER_ALPHAS = 0
cnn_db_loader.NUMBER_IMAGES = 1
cnn_db_loader.NUMBER_XYZ = 1


os.environ['CUDA_VISIBLE_DEVICES']=''

MOMENTUM = 0.9
LEARNING_RATE_DECAY_FACTOR = 0.5   # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.


PaSC_still_BASE = '/user/HS204/m09113/my_project_folder/PaSC/still/multi_fit_CCR_iter75_reg30_256/'
PaSC_video_BASE = '/user/HS204/m09113/my_project_folder/PaSC/video/multi_fit_CCR_iter75_reg30_256/'
CASIA_BASE = '/user/HS204/m09113/my_project_folder/CASIA_webface/multi_fit_CCR_iter75_reg30/'
Experint_BASE = '/user/HS204/m09113/my_project_folder/cnn_experiments/'
experiment_dir = Experint_BASE+FLAGS.experiment_folder
db_dir = experiment_dir+'/db_input/'
train_dir = experiment_dir+'/train'

tf.logging.set_verbosity(tf.logging.DEBUG)

def train():
	"""Train CIFAR-10 for a number of steps."""
	with tf.Graph().as_default():
		global_step = tf.contrib.framework.get_or_create_global_step()

		pasc_still = cnn_db_loader.PaSC_still_loader(outputfolder=db_dir, db_base=PaSC_still_BASE)
		#pasc_video = cnn_db_loader.PaSC_video_loader(outputfolder=db_dir, db_base=PaSC_video_BASE)
		#casia      = cnn_db_loader.CASIA_webface_loader(outputfolder=db_dir, db_base=CASIA_BASE)
		pasc_still.set_all_as_train()
		#casia.set_all_as_train()
		#pasc_video.split_train_eval(train_proportion=0.8)
		#db_loader = cnn_db_loader.Aggregator(pasc_video, pasc_still, casia)
		#db_loader = cnn_db_loader.Aggregator(pasc_video, pasc_still)
		db_loader = cnn_db_loader.Aggregator(pasc_still)

		num_batches_per_epoch = len(db_loader.examples_train) / FLAGS.batch_size

		if cnn_db_loader.NUMBER_ALPHAS > 0:
			image_list, alphas_list, labels_list = db_loader.get_training_image_alphas_and_label_lists()

			images, alphas, labels = tf_utils.inputs_with_alphas(image_list, alphas_list, labels_list, FLAGS.batch_size, db_loader.get_mean_image_path())

			# Build a Graph that computes the logits predictions from the inference model.
			logits, _ = cnn_tf_graphs.inference(network="dcnn_with_alpha", mode=learn.ModeKeys.TRAIN, batch_size=FLAGS.batch_size, num_classes=db_loader.number_ids, input_image_tensor=images, input_alpha_tensor=alphas)			

		elif cnn_db_loader.NUMBER_ALPHAS==0 and cnn_db_loader.NUMBER_IMAGES==1 and cnn_db_loader.NUMBER_XYZ==0:
			image_list, labels_list = db_loader.get_training_image_and_label_lists()
		
			images, labels = tf_utils.inputs(image_list, labels_list, FLAGS.batch_size, db_loader.get_mean_image_path())

			# Build a Graph that computes the logits predictions from the inference model.
			logits, _ = cnn_tf_graphs.inference(network="dcnn", mode=learn.ModeKeys.TRAIN, batch_size=FLAGS.batch_size, num_classes=db_loader.number_ids, input_image_tensor=images)

		elif cnn_db_loader.NUMBER_ALPHAS==0 and cnn_db_loader.NUMBER_IMAGES==1 and cnn_db_loader.NUMBER_XYZ==1:
			image_list, xyz_list, labels_list = db_loader.get_training_image_xyz_and_label_lists()

			isomap_stacks, labels = tf_utils.inputs_stack_image_and_xyz(image_list, xyz_list, labels_list, FLAGS.batch_size, db_loader.get_mean_image_path(), db_loader.get_mean_xyz_path())

			# Build a Graph that computes the logits predictions from the inference model.
			logits, _ = cnn_tf_graphs.inference(network="dcnn", mode=learn.ModeKeys.TRAIN, batch_size=FLAGS.batch_size, num_classes=db_loader.number_ids, input_image_tensor=isomap_stacks)


		# Calculate loss.
		#loss = cnn_tf_graphs.l2_loss(logits, labels)
		loss = cnn_tf_graphs.softmax_loss(logits, labels, db_loader.number_ids)

		top_k_op = tf.nn.in_top_k(logits, labels, 1)
		sum_correct = tf.reduce_sum(tf.cast(top_k_op, tf.float32))
		accuracy = tf.divide(tf.multiply(sum_correct,tf.constant(100.0)),tf.constant(float(FLAGS.batch_size)))
		#accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(tf.argmax(logits,1), tf.argmax(labels, 1))

		lr = tf.constant(INITIAL_LEARNING_RATE, tf.float32)
		tf.summary.scalar('learning_rate', lr)
		tf.summary.scalar('momentum', MOMENTUM)
		tf.summary.scalar('batch_size', FLAGS.batch_size)
		tf.summary.scalar('accuracy', accuracy)

		optimizer=tf.train.MomentumOptimizer(learning_rate=lr, momentum=MOMENTUM)
		#optimizer=tf.train.AdadeltaOptimizer(learning_rate=lr)

		train_op = tf.contrib.layers.optimize_loss(
					loss=loss,
					global_step=tf.contrib.framework.get_global_step(),
					learning_rate=lr,
					optimizer=optimizer)
	

		logging_hook = tf.train.LoggingTensorHook(
						tensors={'step': tf.contrib.framework.get_global_step(),
								 'loss': loss,
								 'lr':   lr,
								 'acc':  accuracy},
						every_n_iter=100)

		#saver = tf.train.Saver(var_list=None, keep_checkpoint_every_n_hours=1)
		saver = tf.train.Saver(var_list=None, max_to_keep=None)

		class _LearningRateSetterHook(tf.train.SessionRunHook):
			"""Sets learning_rate based on global step."""

			def begin(self):
				self._lrn_rate = INITIAL_LEARNING_RATE * LEARNING_RATE_DECAY_FACTOR**10
				#print(self.num_batches_per_epoch)
	
			def before_run(self, run_context):
				return tf.train.SessionRunArgs(
					tf.contrib.framework.get_global_step(),  # Asks for global step value.
					feed_dict={lr: self._lrn_rate})  # Sets learning rate
	
			def after_run(self, run_context, run_values):
				train_step = run_values.results
				#self._lrn_rate = INITIAL_LEARNING_RATE
				#training_epoch = int(train_step/num_batches_per_epoch)
				self._lrn_rate = INITIAL_LEARNING_RATE * LEARNING_RATE_DECAY_FACTOR**int(train_step/num_batches_per_epoch/2.7)
#				if train_step < 2*num_batches_per_epoch:
#					self._lrn_rate = INITIAL_LEARNING_RATE
#				elif train_step < 4*num_batches_per_epoch:
#					self._lrn_rate = INITIAL_LEARNING_RATE * LEARNING_RATE_DECAY_FACTOR**1
#				elif train_step < 6*num_batches_per_epoch:
#					self._lrn_rate = INITIAL_LEARNING_RATE * LEARNING_RATE_DECAY_FACTOR**2
#				elif train_step < 9*num_batches_per_epoch:
#					self._lrn_rate = INITIAL_LEARNING_RATE * LEARNING_RATE_DECAY_FACTOR**3
#				elif train_step < 12*num_batches_per_epoch:
#					self._lrn_rate = INITIAL_LEARNING_RATE * LEARNING_RATE_DECAY_FACTOR**4
#				else:
#					self._lrn_rate = INITIAL_LEARNING_RATE * LEARNING_RATE_DECAY_FACTOR**5

							
		config = tf.ConfigProto( allow_soft_placement=False, log_device_placement=FLAGS.log_device_placement)
		config.gpu_options.allow_growth = True
	
		with tf.train.MonitoredTrainingSession(
				checkpoint_dir=train_dir,
				hooks=[ tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
						tf.train.NanTensorHook(loss),
						tf.train.CheckpointSaverHook(checkpoint_dir=train_dir, save_steps=num_batches_per_epoch, saver=saver),
						logging_hook,
						_LearningRateSetterHook()],
				config=config,
				save_checkpoint_secs=3600
				) as mon_sess:
			#saver.restore(mon_sess,'/vol/vssp/facer2vm/people/Philipp/cnn_experiments/03/train/model.ckpt-21575')
			while not mon_sess.should_stop():
				mon_sess.run(train_op)
				#mon_sess.run(train_op)
		#my_summary_op = tf.summary.merge_all()
		#sv = tf.train.Supervisor(logdir="/my/training/directory", summary_op=None) # Do not run the summary service



def main(argv=None):  # pylint: disable=unused-argument

	if not os.path.exists(experiment_dir):
		os.mkdir(experiment_dir)


	if not os.path.exists(train_dir):
		os.mkdir(train_dir)

	if not os.path.exists(db_dir):
		os.mkdir(db_dir)

	with tf.device('/gpu:0'):
		train()


if __name__ == '__main__':
	tf.app.run()
