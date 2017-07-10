#!/usr/bin/env python3.5

import os
import sys
import numpy as np
import glob, random
import merge_isomaps

#import tf_utils
#import cnn_tf_graphs
#import cnn_db_loader
import cv2

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


ISOMAP_SIZE = 256


#print('globing all files together first')
os.environ['CUDA_VISIBLE_DEVICES']='0'

file_ending='isomap.png'
mean_name = 'total_mean.png'

#INPUT_ISOMAP_BASE = '/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256/verification_templates/'
#image_list=glob.glob(INPUT_ISOMAP_BASE+'split1/10*/*'+file_ending)
#INPUT_ISOMAP_BASE = '/user/HS204/m09113/my_project_folder/CASIA_webface/multi_fit_CCR_iter75_reg30_256/'
INPUT_ISOMAP_BASE = '/user/HS204/m09113/my_project_folder/PaSC/video/multi_fit_CCR_iter75_reg30_256/'
#image_list=glob.glob(INPUT_ISOMAP_BASE+'*/*/*'+file_ending) #IJBA
image_list=glob.glob(INPUT_ISOMAP_BASE+'*/*'+file_ending) #CASIA, PASC

print('found all files! Let\'s do the work now')

Experint_BASE = '/user/HS204/m09113/my_project_folder/cnn_experiments/'
experiment_dir = Experint_BASE+FLAGS.experiment_folder
db_dir = experiment_dir+'/db_input/'
train_dir = experiment_dir+'/train'
eval_dir = experiment_dir+'/eval'
eval_log = eval_dir+'/eval.log'

#OUTPUT_FOLDER_CONF = '/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256_conf13_sm/verification_templates/'
#OUTPUT_FOLDER_CONF = '/user/HS204/m09113/my_project_folder/CASIA_webface/multi_fit_CCR_iter75_reg30_256_conf13/'
OUTPUT_FOLDER_CONF = '/user/HS204/m09113/my_project_folder/PaSC/video/multi_fit_CCR_iter75_reg30_256_conf13/'

if not os.path.exists(OUTPUT_FOLDER_CONF):
	os.mkdir(OUTPUT_FOLDER_CONF)

take_iter = 88477
#take_iter = None

tf.logging.set_verbosity(tf.logging.DEBUG)


def main(argv=None):  # pylint: disable=unused-argument

	if not os.path.exists(experiment_dir):
		print('no experiment dir found!')
		exit()

	if not os.path.exists(train_dir):
		print('no training dir found!')
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

	#create split folders
	#for split in range(1,11):
	#	if not os.path.exists(OUTPUT_FOLDER_CONF+'split'+str(split)):
	#		os.mkdir(OUTPUT_FOLDER_CONF+'split'+str(split))
	

	conf_output_paths = []
	for image_path in image_list:
		#create template folder if neccessary
		template_folder = os.path.dirname(image_path).replace(INPUT_ISOMAP_BASE, OUTPUT_FOLDER_CONF)
		if not os.path.exists(template_folder):
			os.mkdir(template_folder)

		conf_output_paths.append(image_path.replace(INPUT_ISOMAP_BASE, OUTPUT_FOLDER_CONF).split('.')[0]+'.isomap_conf.npy')

	#print (image_list[:10])
	#print (conf_output_paths[:10])
	merge_isomaps.write_cnn_confidences(best_net_checkpoint, image_list, conf_output_paths)


if __name__ == '__main__':
	tf.app.run()