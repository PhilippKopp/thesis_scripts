#!/usr/bin/env python3.5


from datetime import datetime
import time
import os
import math
import sys
import numpy as np
import cnn_db_loader

import tf_utils
import cnn_tf_graphs

import tensorflow as tf
from tensorflow.contrib import learn


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('experiment_folder', '21',
													 """Directory where to write event logs """
													 """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
														"""Whether to log device placement.""")
tf.app.flags.DEFINE_integer('batch_size', 100,
														"""Size of a batch.""")

os.environ['CUDA_VISIBLE_DEVICES']='0'


cnn_db_loader.NUMBER_ALPHAS = 0
cnn_db_loader.NUMBER_IMAGES = 1
cnn_db_loader.NUMBER_XYZ = 0


Experint_BASE = '/user/HS204/m09113/my_project_folder/cnn_experiments/'
experiment_dir = Experint_BASE+FLAGS.experiment_folder
db_dir = experiment_dir+'/db_input/'
train_dir = experiment_dir+'/train'
eval_dir = experiment_dir+'/eval'
eval_log = eval_dir+'/eval.log'

tf.logging.set_verbosity(tf.logging.DEBUG)

def eval(saved_model_path, db_loader):
	"""Train CIFAR-10 for a number of steps."""
	with tf.Graph().as_default():
		#global_step = tf.contrib.framework.get_or_create_global_step()

		if cnn_db_loader.NUMBER_ALPHAS>0:
			image_list, alphas_list, labels_list = db_loader.get_eval_image_alphas_and_label_lists()

			images, alphas, labels = tf_utils.inputs_with_alphas(image_list, alphas_list, labels_list, FLAGS.batch_size, db_loader.get_mean_image_path())

			# Build a Graph that computes the logits predictions from the inference model.
			logits, _ = cnn_tf_graphs.inference(network="alex_with_alpha", mode=learn.ModeKeys.EVAL, batch_size=FLAGS.batch_size, num_classes=db_loader.number_ids, input_image_tensor=images, input_alpha_tensor=alphas)
			
		elif cnn_db_loader.NUMBER_IMAGES==1 and cnn_db_loader.NUMBER_ALPHAS==0 and cnn_db_loader.NUMBER_XYZ==0:
			image_list, labels_list = db_loader.get_eval_image_and_label_lists()

			images, labels = tf_utils.inputs(image_list, labels_list, FLAGS.batch_size, db_loader.get_mean_image_path(), image_size=256)

			# Build a Graph that computes the logits predictions from the inference model.
			logits, _ = cnn_tf_graphs.inference(network="alex", mode=learn.ModeKeys.EVAL, batch_size=FLAGS.batch_size, num_classes=db_loader.number_ids, input_image_tensor=images, image_size=256)

		elif cnn_db_loader.NUMBER_IMAGES==0 and cnn_db_loader.NUMBER_ALPHAS==0 and cnn_db_loader.NUMBER_XYZ==1:
			image_list, labels_list = db_loader.get_eval_xyz_and_label_lists()

			images, labels = tf_utils.inputs(image_list, labels_list, FLAGS.batch_size, db_loader.get_mean_xyz_path())

			# Build a Graph that computes the logits predictions from the inference model.
			logits, _ = cnn_tf_graphs.inference(network="alex", mode=learn.ModeKeys.EVAL, batch_size=FLAGS.batch_size, num_classes=db_loader.number_ids, input_image_tensor=images)

		elif cnn_db_loader.NUMBER_ALPHAS == 0 and cnn_db_loader.NUMBER_IMAGES == 1 and cnn_db_loader.NUMBER_XYZ == 1:
			image_list, xyz_list, labels_list = db_loader.get_eval_image_xyz_and_label_lists()

			isomap_stacks, labels = tf_utils.inputs_stack_image_and_xyz(image_list, xyz_list, labels_list, FLAGS.batch_size, db_loader.get_mean_image_path(), db_loader.get_mean_xyz_path())

			# Build a Graph that computes the logits predictions from the inference model.
			logits, _ = cnn_tf_graphs.inference(network="dcnn", mode=learn.ModeKeys.EVAL, batch_size=FLAGS.batch_size, num_classes=db_loader.number_ids, input_image_tensor=isomap_stacks)

		elif cnn_db_loader.NUMBER_ALPHAS == 0 and cnn_db_loader.NUMBER_IMAGES > 1 and cnn_db_loader.NUMBER_XYZ == 0:
			images_list, labels_list = db_loader.get_eval_multi_image_and_label_lists()

			#images = [0]*cnn_db_loader.NUMBER_IMAGES
			output = tf_utils.inputs_multi(images_list, labels_list, FLAGS.batch_size, db_loader.get_mean_image_path(), png_with_alpha=True, image_size=256)
			#output = tf_utils.inputs_multi(images_list, labels_list, FLAGS.batch_size, db_loader.get_mean_image_path(), png_with_alpha=False, image_size=512)
			images = output[:cnn_db_loader.NUMBER_IMAGES]
			labels  = output[-1]

			confs  = [0]*cnn_db_loader.NUMBER_IMAGES
			with tf.variable_scope("confidence_estimation") as scope:
				for i in range(cnn_db_loader.NUMBER_IMAGES):
					confs[i] = cnn_tf_graphs.confidence_cnn23(images[i], input_size=256)
					scope.reuse_variables()

			merging_input_list = [[images[i], confs[i]] for i in range(cnn_db_loader.NUMBER_IMAGES)]
			merged_image = cnn_tf_graphs.merge_isomaps_softmax(merging_input_list)

			merged_image = tf.slice(merged_image,[0,0,0,0],[-1,-1,-1,3])

			# Build a Graph that computes the logits predictions from the inference model.
			logits, _ = cnn_tf_graphs.inference(network="alex", mode=learn.ModeKeys.EVAL, batch_size=FLAGS.batch_size, num_classes=db_loader.number_ids, input_image_tensor=merged_image, image_size=256)

		
		#correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
		#batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		top_k_op = tf.nn.in_top_k(logits, labels, 1)



		saver = tf.train.Saver()

		config = tf.ConfigProto( allow_soft_placement=False, log_device_placement=FLAGS.log_device_placement)
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:
			print('restore model')
			saver.restore(sess, saved_model_path)

			print('we have',len(db_loader.examples_eval), 'images to evaluate')

			# Start the queue runners.
			coord = tf.train.Coordinator()
			try:
				threads = []
				for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
					threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

				num_iter = int(math.ceil(len(db_loader.examples_eval) / FLAGS.batch_size))
				true_count = 0  # Counts the number of correct predictions.
				total_sample_count = num_iter * FLAGS.batch_size
				step = 0
				print('so have to run this',num_iter, 'times. Let\'s start! iter: ', end=' ')
				sys.stdout.flush()
				while step < num_iter and not coord.should_stop():
					predictions = sess.run([top_k_op])
					true_count += np.sum(predictions)
					step += 1
					print(step, end=' ')
					sys.stdout.flush()

				# Compute precision @ 1.
				precision = true_count / total_sample_count
			except Exception as e:  # pylint: disable=broad-except
				coord.request_stop(e)
			
			coord.request_stop()
			coord.join(threads, stop_grace_period_secs=10)
	return precision
		


def main(argv=None):  # pylint: disable=unused-argument

	if not os.path.exists(experiment_dir):
		print('no experiment dir found!')
		exit()

	if not os.path.exists(train_dir):
		print('no training dir found!')
		exit()

	if not os.path.exists(eval_dir):
		os.mkdir(eval_dir)

	# see what checkpoints we can find in the training folder
	ckpt_files = [f[:-6] for f in os.listdir(train_dir) if os.path.isfile( os.path.join(train_dir, f)) and 'ckpt' in f and 'index' in f]
	ckpt_files.sort()
	ckpt_steps_found = [int(ckpt_file.split('/')[-1].split('-')[-1]) for ckpt_file in ckpt_files]
	#print('found:',ckpt_steps_found)

	ckpt_steps_done=[]
	if os.path.exists(eval_log):
		with open(eval_log,'r') as log:
			for line in log:
				ckpt_steps_done.append(int(line.split()[0]))
	#print('done:',ckpt_steps_done)

	ckpt_files_todo=[]
	for i in range(len(ckpt_files)):
		if not ckpt_steps_found[i] in ckpt_steps_done:
			ckpt_files_todo.append(ckpt_files[i])
	print('checkpoint files todo:',ckpt_files_todo)


	db_loader = cnn_db_loader.lazy_dummy(db_dir)


	with tf.device('/gpu:0'):
		for ckpt_file in ckpt_files_todo:
			# first check if it still there
			if os.path.exists(os.path.join(train_dir, ckpt_file)+'.meta'):
				print('evaluating checkpoint file',ckpt_file)
				precision = eval(os.path.join(train_dir, ckpt_file), db_loader)
				print ('\ntrain iter',ckpt_file,'has precision',precision)
				with open(eval_log,'a') as log:
					log.write(ckpt_file.split('/')[-1].split('-')[-1]+' '+str(precision)+'\n')



if __name__ == '__main__':
	tf.app.run()
