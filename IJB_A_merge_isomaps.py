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
#from shutil import copyfile


import tensorflow as tf
from tensorflow.contrib import learn


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
														"""Whether to log device placement.""")

INPUT_ISOMAP_BASE = '/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256/verification_templates/'
INPUT_CONF_BASE = '/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256_conf13_sm/verification_templates/'

OUTPUT_MERGE_BASE = '/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256_conf13_sm/verification_templates_merge_best3/'


#print('globing all files together first')
os.environ['CUDA_VISIBLE_DEVICES']='0'

isomap_file_ending = '.isomap.png'
confidence_file_ending = '.isomap_conf.npy'
mean_name = 'total_mean.png'


template_list=glob.glob(INPUT_ISOMAP_BASE+'*/*')
#template_list=glob.glob(INPUT_ISOMAP_BASE+'split1/*')

print('found all templates')


if not os.path.exists(OUTPUT_MERGE_BASE):
	os.mkdir(OUTPUT_MERGE_BASE)

#create split folders
for split in range(1,11):
	if not os.path.exists(OUTPUT_MERGE_BASE+'split'+str(split)):
		os.mkdir(OUTPUT_MERGE_BASE+'split'+str(split))

tf.logging.set_verbosity(tf.logging.DEBUG)


merging_lists = []
confidence_lists=[]
merged_image_output_paths = []


for template_path in template_list:
	isomaps = glob.glob(template_path+'/*'+isomap_file_ending)
	confidence_maps = [i.replace(INPUT_ISOMAP_BASE, INPUT_CONF_BASE) for i in isomaps]
	confidence_maps = [c.replace(isomap_file_ending, confidence_file_ending) for c in confidence_maps]
	#template_num = os.path.basename(os.path.normpath(template_path))
	image_output_path = template_path.replace(INPUT_ISOMAP_BASE, OUTPUT_MERGE_BASE)+'.png'

	#print (isomaps)
	#print (confidence_maps)
	#print (image_output_path)

	merging_lists.append(isomaps)
	confidence_lists.append(confidence_maps)
	merged_image_output_paths.append(image_output_path)

print('found all files! Let\'s do the work now')	

#1) merge all images in a template together to one image
#merge_isomaps.merge_sm_with_tf(merging_lists, confidence_lists, merged_image_output_paths)


#2) calc mean for each confidence and take highest
#for i in range(len(merging_lists)):
#
#	mean_conf=[]
#	if len(confidence_lists[i])>0:
#		for j in range(len(confidence_lists[i])):
#			mean_conf.append( np.mean(np.load(confidence_lists[i][j])) )
#			#print ('image',merging_lists[i][j],'has mean conf',mean_conf[-1])
#		index_heighest_mean = mean_conf.index(max(mean_conf))
#		#print('copying',merging_lists[i][index_heighest_mean])
#
#		os.symlink(merging_lists[i][index_heighest_mean], merged_image_output_paths[i])


#3) calc mean for each confidence, take highest 3 and merge them
for i in range(len(merging_lists)):

	mean_conf=[]
	if len(confidence_lists[i])>0:
		if len(confidence_lists[i])<=3:
			continue
		for j in range(len(confidence_lists[i])):
			mean_conf.append( np.mean(np.load(confidence_lists[i][j])) )
			#print ('image',merging_lists[i][j],'has mean conf',mean_conf[-1])
		
		best_3 = sorted(zip(mean_conf, range(len(mean_conf))), reverse=True)[:3]
		best_3_indices = [x[1] for x in best_3]
		#print ('best 3 indices are', best_3_indices, 'of total',mean_conf)

		new_confidence_list = [confidence_lists[i][best_index] for best_index in best_3_indices]
		new_isomap_list = [merging_lists[i][best_index] for best_index in best_3_indices]

		#print ('orig', confidence_lists[i], 'new', new_confidence_list)
		#print ('orig', merging_lists[i], 'new', new_isomap_list)
		confidence_lists[i] = new_confidence_list
		merging_lists[i] = new_isomap_list

		#index_heighest_mean = mean_conf.index(max(mean_conf))
		#print('copying',merging_lists[i][index_heighest_mean])

		#os.symlink(merging_lists[i][index_heighest_mean], merged_image_output_paths[i])

merge_isomaps.merge_sm_with_tf(merging_lists, confidence_lists, merged_image_output_paths)
