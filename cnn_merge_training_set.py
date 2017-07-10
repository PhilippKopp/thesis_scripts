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

#INPUT_ISOMAP_BASE = '/user/HS204/m09113/my_project_folder/CASIA_webface/multi_fit_CCR_iter75_reg30_256/'
INPUT_ISOMAP_BASE = '/user/HS204/m09113/my_project_folder/PaSC/video/multi_fit_CCR_iter75_reg30_256/'

#INPUT_CONF_BASE = '/user/HS204/m09113/my_project_folder/CASIA_webface/multi_fit_CCR_iter75_reg30_256_conf13/'
INPUT_CONF_BASE = '/user/HS204/m09113/my_project_folder/PaSC/video/multi_fit_CCR_iter75_reg30_256_conf13/'

#OUTPUT_MERGE_BASE = '/user/HS204/m09113/my_project_folder/CASIA_webface/random_merges_256_conf13/'
OUTPUT_MERGE_BASE = '/user/HS204/m09113/my_project_folder/PaSC/video/random_merges_256_conf13/'


#print('globing all files together first')
os.environ['CUDA_VISIBLE_DEVICES']='0'

isomap_file_ending = '.isomap.png'
confidence_file_ending = '.isomap_conf.npy'
mean_name = 'total_mean.png'


template_list=glob.glob(INPUT_ISOMAP_BASE+'*/')
#template_list=glob.glob(INPUT_ISOMAP_BASE+'split1/*')

print('found all templates')


if not os.path.exists(OUTPUT_MERGE_BASE):
	os.mkdir(OUTPUT_MERGE_BASE)

#create split folders
#for split in range(1,11):
#	if not os.path.exists(OUTPUT_MERGE_BASE+'split'+str(split)):
#		os.mkdir(OUTPUT_MERGE_BASE+'split'+str(split))

tf.logging.set_verbosity(tf.logging.DEBUG)


merging_lists = []
confidence_lists=[]
merged_image_output_paths = []


random.seed(404)
for template_path in template_list:
	isomaps_all = glob.glob(template_path+'/*'+isomap_file_ending)
	if len(isomaps_all)<10:
		continue
	#print(template_path, 'we would have',len(isomaps_all),'isomaps')
	for i in range(10):
		random.shuffle(isomaps_all)
		random_number_to_merge = random.randint(2,int(2*(len(isomaps_all)-1)/3))
		#print('but we take',random_number_to_merge)
		isomaps=isomaps_all[:random_number_to_merge]
		confidence_maps = [i.replace(INPUT_ISOMAP_BASE, INPUT_CONF_BASE) for i in isomaps]
		confidence_maps = [c.replace(isomap_file_ending, confidence_file_ending) for c in confidence_maps]
		#id_num = os.path.basename(os.path.normpath(template_path))
		if not os.path.exists(template_path.replace(INPUT_ISOMAP_BASE, OUTPUT_MERGE_BASE)):
			os.mkdir(template_path.replace(INPUT_ISOMAP_BASE, OUTPUT_MERGE_BASE))
		merge_num=0
		image_output_path = template_path.replace(INPUT_ISOMAP_BASE, OUTPUT_MERGE_BASE)+str(merge_num).zfill(3)+'_'+str(random_number_to_merge)+'.png'
		while(os.path.exists(image_output_path)):
			merge_num+=1
			image_output_path = template_path.replace(INPUT_ISOMAP_BASE, OUTPUT_MERGE_BASE)+str(merge_num).zfill(3)+'_'+str(random_number_to_merge)+'.png'

		#print (isomaps)
		#print (confidence_maps)
		#print (image_output_path)

		merging_lists.append(isomaps)
		confidence_lists.append(confidence_maps)
		merged_image_output_paths.append(image_output_path)

print('found all files! Let\'s do the work now')	

#exit(0)

#1) merge all images in a template together to one image
merge_isomaps.merge_sm_with_tf(merging_lists, confidence_lists, merged_image_output_paths)

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



