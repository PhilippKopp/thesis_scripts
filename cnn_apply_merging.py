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

NUMBER_ALPHAS = 0
NUMBER_IMAGES = 1
NUMBER_XYZ = 0

ISOMAP_SIZE = 256


#print('globing all files together first')
os.environ['CUDA_VISIBLE_DEVICES']='0'

if NUMBER_XYZ>0:
	file_ending='xyzmap.png'
	mean_name = 'total_mean_xyz.png'
else:
	file_ending='isomap.png'
	mean_name = 'total_mean.png'

#image_list=glob.glob('/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30/verification_templates/split1/10*/*'+file_ending)
#image_list=glob.glob('/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30/verification_templates/*/*/*'+file_ending)
#image_list=glob.glob('/user/HS204/m09113/my_project_folder/merging_playground/IJB_A_120/*isomap.png')
image_list=glob.glob('/user/HS204/m09113/my_project_folder/merging_playground/PaSC_still_05663/*isomap.png')
print('found all files! Let\'s do the work now')
Experint_BASE = '/user/HS204/m09113/my_project_folder/cnn_experiments/'
experiment_dir = Experint_BASE+FLAGS.experiment_folder
db_dir = experiment_dir+'/db_input/'
train_dir = experiment_dir+'/train'
eval_dir = experiment_dir+'/eval'
eval_log = eval_dir+'/eval.log'

#OUTPUT_FOLDER_CONF = '/user/HS204/m09113/my_project_folder/merging_playground/IJB_A_120_conf14_sm/'
#OUTPUT_FOLDER_RESULT = '/user/HS204/m09113/my_project_folder/merging_playground/IJB_A_120_conf14_sm_merges/'
OUTPUT_FOLDER_CONF = '/user/HS204/m09113/my_project_folder/merging_playground/PaSC_still_05663_conf13_sm/'
OUTPUT_FOLDER_RESULT = '/user/HS204/m09113/my_project_folder/merging_playground/PaSC_still_05663_conf13_sm_merges/'

if not os.path.exists(OUTPUT_FOLDER_CONF):
	os.mkdir(OUTPUT_FOLDER_CONF)

if not os.path.exists(OUTPUT_FOLDER_RESULT):
	os.mkdir(OUTPUT_FOLDER_RESULT)

#take_iter = 116788
take_iter = None

tf.logging.set_verbosity(tf.logging.DEBUG)

def read_merging_list(file_path):
	merging_list = []
	f = open(file_path, 'r')
	for line in f:
		parts = line[:-1].split(' ')[:-1]
		merging_list.append(parts)
	return merging_list
	


def main(argv=None):  # pylint: disable=unused-argument

	if not os.path.exists(experiment_dir):
		print('no experiment dir found!')
		exit()

	if not os.path.exists(train_dir):
		print('no training dir found!')
		exit()

	#if not os.path.exists(eval_dir):
	#	print('no eval dir found!')
	#	exit()

	#if not os.path.exists(eval_log):
	#	print('no log file found!')
	#	exit()

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
	

	conf_output_paths = []
	for image_path in image_list:
		conf_output_paths.append(OUTPUT_FOLDER_CONF+os.path.basename(image_path).split('.')[0]+'.npy')

	merge_isomaps.write_cnn_confidences(best_net_checkpoint, image_list, conf_output_paths)

	#for image in image_list:
	#	conf_file = OUTPUT_FOLDER_CONF+os.path.basename(image).split('.')[0]+'.npy'
	#	cv2.imwrite(conf_file.replace('.npy','_coloured.png'),merge_isomaps.color_alpha_only(np.load(conf_file), minval=-1.5, maxval=3.0))

	#merging_list = read_merging_list('/user/HS204/m09113/my_project_folder/merging_playground/pasc_still_05671_training_triplets.txt')
	random.seed(404)
	merging_list = []
	confidence_lists=[]
	merged_image_output_path = []
	for merge_legth in range(2,100):
		for i in range(3):

			try:
				merging_list.append(random.sample(image_list,merge_legth))
				merged_image_output_path.append(OUTPUT_FOLDER_RESULT+'/'+str(merge_legth)+'_images/'+str(i)+'/tf_merge_'+str(merge_legth)+'_'+str(i)+'.png')

				#create outputfolders
				if not os.path.exists(OUTPUT_FOLDER_RESULT+'/'+str(merge_legth)+'_images/'):
					os.mkdir(OUTPUT_FOLDER_RESULT+'/'+str(merge_legth)+'_images/')
				if not os.path.exists(OUTPUT_FOLDER_RESULT+'/'+str(merge_legth)+'_images/'+str(i)):
					os.mkdir(OUTPUT_FOLDER_RESULT+'/'+str(merge_legth)+'_images/'+str(i))
				
				tmp_list=[]	
				for isomap_path in merging_list[-1]:
					tmp_list.append(OUTPUT_FOLDER_CONF+os.path.basename(isomap_path).split('.')[0]+'.npy')
					try:
						os.symlink(isomap_path, OUTPUT_FOLDER_RESULT+str(merge_legth)+'_images/'+str(i)+'/'+os.path.basename(isomap_path))
					except FileExistsError:
						pass
				confidence_lists.append(tmp_list)

			except ValueError:
				pass

	# do the debug colouring		
	if True==True:
		for j, confidence_list in enumerate(confidence_lists):
			conf_map_exp=[]
			for conf_map in confidence_list:
				#print ('conf',conf_map[0,100:102,100:102])
				#conf_map_exp.append( np.exp(conf_map[0]) )
				conf_map_exp.append( np.exp(np.load(conf_map)))
				#print ('conf exp',conf_map_exp[-1][100:102,100:102])
				#print (conf_map_exp[-1].shape)
			exp_sum = np.sum(conf_map_exp, axis=0)
			#print (exp_sum[100:102,100:102])
			#print ('sum shape ',exp_sum.shape)
			for i, conf_map in enumerate(conf_map_exp):
				conf_map/=exp_sum
				#conf_colour_file = OUTPUT_FOLDER_RESULT+str(len(confidence_list))+'_images/'+str(idx)+'/'+os.path.basename(isomap_list[i]).split('.')[0]+'.coloured.png'
				conf_colour_file = os.path.dirname(merged_image_output_path[j])+'/'+os.path.basename(merging_list[j][i]).split('.')[0]+'.coloured.png'
				cv2.imwrite(conf_colour_file,merge_isomaps.color_alpha_only(conf_map, minval=0.0, maxval=1.0))

		
	merge_isomaps.merge_sm_with_tf(merging_list, confidence_lists, merged_image_output_path)

	merge_results = glob.glob(OUTPUT_FOLDER_RESULT+'/*/*/tf_merge_*')
	summary_folder = OUTPUT_FOLDER_RESULT +'/summary/'
	if not os.path.exists(summary_folder):
		os.mkdir(summary_folder)

	for merge_result in merge_results:
		try:
			os.symlink(merge_result, summary_folder+os.path.basename(merge_result))
		except FileExistsError:
			pass		



if __name__ == '__main__':
	tf.app.run()