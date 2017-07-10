#!/usr/bin/env python3.5


from datetime import datetime
import time
import os
import math
import sys
import numpy as np
import glob
import obj_analysis_lib as oal

import tf_utils
import cnn_tf_graphs

import tensorflow as tf
from tensorflow.contrib import learn


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('experiment_folder', '40',
													 """Directory where to write event logs """
													 """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
														"""Whether to log device placement.""")
tf.app.flags.DEFINE_integer('batch_size', 100,
														"""Size of a batch.""")

NUMBER_ALPHAS = 0
NUMBER_IMAGES = 1
NUMBER_XYZ = 0

os.environ['CUDA_VISIBLE_DEVICES']='0'


print('globing all files together first')

if NUMBER_XYZ>0 and NUMBER_IMAGES==0:
	file_ending='xyzmap.png'
	mean_name = 'total_mean_xyz.png'
elif NUMBER_XYZ==0 and NUMBER_IMAGES>0:
	file_ending='isomap.png'
	mean_name = 'total_mean.png'
else:
	file_ending='isomap.png'
	file_ending_xyz = 'xyzmap.png'
	mean_name = 'total_mean.png'
	mean_name_xyz =  'total_mean_xyz.png'


#image_list=glob.glob('/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30/verification_templates/split1/10*/*'+file_ending)
#image_list=glob.glob('/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256/verification_templates/*/*/*'+file_ending)
#image_list=glob.glob('/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30/verification_templates/*/*/*'+file_ending)
#image_list=glob.glob('/user/HS204/m09113/my_project_folder/IJB_A/face_boxes/*/*')
#image_list=glob.glob('/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256_conf13_sm/verification_templates_merged/*/*')
image_list=glob.glob('/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256_conf13_sm/verification_templates_take_best1/*/*')

print('found all files! Let\'s do the work now')
Experint_BASE = '/user/HS204/m09113/my_project_folder/cnn_experiments/'
experiment_dir = Experint_BASE+FLAGS.experiment_folder
db_dir = experiment_dir+'/db_input/'
train_dir = experiment_dir+'/train'
eval_dir = experiment_dir+'/eval'
eval_log = eval_dir+'/eval.log'
#test_log = experiment_dir+'/merged_ijba_vectors.csv'
test_log = experiment_dir+'/best1_ijba_vectors.csv'

#take_iter = 42469
take_iter = 88477
#take_iter= None

tf.logging.set_verbosity(tf.logging.DEBUG)

def test(saved_model_path, images, alphas=[]):
	with tf.Graph().as_default():
		
		image_path_tensor = tf.placeholder(tf.string)
		#image_tf = tf_utils.single_input_image(image_path_tensor, db_dir+mean_name, image_size=256)
		#image_tf = tf_utils.single_input_image(image_path_tensor, '/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256_conf13_sm/templates_merged_mean.png', image_size=256)
		image_tf = tf_utils.single_input_image(image_path_tensor, '/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256_conf13_sm/take_best1_merge_mean.png', image_size=256)
		image_tf = tf.expand_dims(image_tf,0)

		# Build a Graph that computes the logits predictions from the inference model.
		if NUMBER_ALPHAS == 0 and NUMBER_IMAGES == 1 and NUMBER_XYZ == 0:
			_, feature_vector_tensor = cnn_tf_graphs.inference(network="alex", mode=learn.ModeKeys.EVAL, batch_size=1, num_classes=10868, input_image_tensor=image_tf, image_size=256)

		elif NUMBER_ALPHAS == 1 and NUMBER_IMAGES == 1 and NUMBER_XYZ == 0:
			alphas_tf = tf.placeholder(tf.float32, shape=(63))
			alphas_tf = tf.expand_dims(alphas_tf,0)
			_, feature_vector_tensor = cnn_tf_graphs.inference(network="alex_with_alpha", mode=learn.ModeKeys.EVAL, batch_size=1, num_classes=10868, input_image_tensor=image_tf, input_alpha_tensor=alphas_tf)

		elif NUMBER_ALPHAS == 0 and NUMBER_IMAGES == 1 and NUMBER_XYZ == 1:
			xyz_path_tensor = tf.placeholder(tf.string)
			xyz_tf = tf_utils.single_input_image(image_path_tensor, db_dir+mean_name_xyz)
			xyz_tf = tf.expand_dims(xyz_tf,0)
			stack_tf = tf.concat([image_tf, xyz_tf], axis=3)
			_, feature_vector_tensor = cnn_tf_graphs.inference(network="dcnn", mode=learn.ModeKeys.EVAL, batch_size=1, num_classes=10868, input_image_tensor=stack_tf)


		saver = tf.train.Saver()

		vectors = np.empty([len(images), feature_vector_tensor.shape[1]])

		config = tf.ConfigProto( allow_soft_placement=False, log_device_placement=FLAGS.log_device_placement)
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:
			print('restore model')
			saver.restore(sess, saved_model_path)
			print ('restoring done')

			#print('we have',db_loader.num_examples_eval, 'images to evaluate')
			for idx, image_path in enumerate(images):
				if idx%1000==0:
					print (idx,'of',len(images))
				if NUMBER_ALPHAS == 0 and NUMBER_IMAGES == 1 and NUMBER_XYZ == 0:
					vector = sess.run(feature_vector_tensor, feed_dict={image_path_tensor: image_path})
				elif NUMBER_ALPHAS == 1 and NUMBER_IMAGES == 1 and NUMBER_XYZ == 0:
					vector = sess.run(feature_vector_tensor, feed_dict={image_path_tensor: image_path, alphas_tf: np.expand_dims(np.array(alphas[idx]),axis=0)})
				elif NUMBER_ALPHAS == 0 and NUMBER_IMAGES == 1 and NUMBER_XYZ == 1:
					xyz_path = image_path.replace(file_ending, file_ending_xyz)
					vector = sess.run(feature_vector_tensor, feed_dict={image_path_tensor: image_path, xyz_path_tensor: xyz_path})
				vectors[idx,:] = vector[0]
				#print ('got vector of length',len(vector[0]),'and sum',sum(vector[0]))
	return vectors

	


def main(argv=None):  # pylint: disable=unused-argument

	if not os.path.exists(experiment_dir):
		print('no experiment dir found!')
		exit()

	if not os.path.exists(train_dir):
		print('no training dir found!')
		exit()

	if not os.path.exists(eval_dir):
		print('no eval dir found!')
		exit()

	if not os.path.exists(eval_log):
		print('no log file found!')
		exit()

	if not take_iter:
		# find best network
		best_accuracy=0.0
		best_iter=0
		with open(eval_log,'r') as log:
			for line in log:
				iter_, accuracy = [float(x) for x in line.split()]
				if accuracy>best_accuracy:
					best_accuracy = accuracy
					best_iter = int(iter_)

		best_net_checkpoint = train_dir+'/model.ckpt-'+str(best_iter)
		print('best network is',best_net_checkpoint)
		# double check if we have this network checkpoint
		if not os.path.exists(best_net_checkpoint+'.meta'):
			print('shit! this checkpoint went missing... exiting ...')
			exit(0)
	else:
		best_net_checkpoint = train_dir+'/model.ckpt-'+str(take_iter)


	#db_loader = cnn_db_loader.PaSC_db_loader(db_base=PaSC_BASE, outputfolder=experiment_dir)
	

	with tf.device('/gpu:0'):

		if NUMBER_ALPHAS==0:
			vectors = test(best_net_checkpoint, image_list)
		else:
			print('reading fitting logs for alphas now')
			all_alphas = []
			old_alphas = []
			old_folder = ''
			for image in image_list:
				folder = os.path.dirname(image)
				if folder==old_folder:
					alphas = old_alphas
				else:
					alphas, _  = oal.read_fitting_log(folder+'/fitting.log')
				old_alphas = alphas
				old_folder = folder
				all_alphas.append(alphas)
			vectors = test(best_net_checkpoint, image_list, alphas=all_alphas)


		with open(test_log, 'w') as log:
			for i in range(len(image_list)):
				log.write(image_list[i]+' ')
				for x in range(vectors.shape[1]):
					log.write(str(vectors[i,x])+' ')
				log.write('\n')


		
		#print ('\ntrain iter',ckpt_file,'has precision',precision)
		#with open(eval_log,'a') as log:
		#	log.write(ckpt_file.split('/')[-1].split('-')[-1]+' '+str(precision)+'\n')



if __name__ == '__main__':
	tf.app.run()
