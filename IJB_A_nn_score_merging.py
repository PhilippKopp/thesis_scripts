#!/usr/bin/env python3.5

import sys, os, random
import numpy as np
import IJB_A_template_lib as itl
import obj_analysis_lib as oal
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES']=''

scores_base_HO2D = '/user/HS204/m09113/my_project_folder/IJB_A/fr_verification_experiments/verification_exp_99/'
scores_base_HO3D = '/user/HS204/m09113/my_project_folder/IJB_A/fr_verification_experiments/verification_exp_80/'
scores_base_PH   = '/user/HS204/m09113/my_project_folder/IJB_A/fr_verification_experiments/verification_exp_13/'

verification_exp_base = '/user/HS204/m09113/my_project_folder/IJB_A/fr_verification_experiments/verification_exp_84/'
MATCHING_FOLDER = '/matching_cos/'
if not os.path.exists(verification_exp_base):
	os.mkdir(verification_exp_base)
if not os.path.exists(verification_exp_base+MATCHING_FOLDER):
	os.mkdir(verification_exp_base+MATCHING_FOLDER)

FITTING_BASE = '/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256/verification_templates/'

trained_nn_score_choser = '/user/HS204/m09113/my_project_folder/IJB_A/nn_score_chooser/04/model.ckpt-1164141'

BATCH_SIZE = 1
#assemble nn
input_layer = tf.placeholder(tf.float32, shape=(BATCH_SIZE,15))
fc1 = tf.layers.dense(input_layer, units=110 , activation=tf.nn.relu, name='fc1')
fc2 = tf.layers.dense(fc1, units=110 , activation=tf.nn.relu, name='fc2')
fc3 = tf.layers.dense(fc2, units=3 , activation=tf.nn.relu, name='fc3')

saver = tf.train.Saver()


chosen_score = [0, 0, 0]
for split in range(1,11):
	print ('merging split',split)

	#matches_file = merge_A_base+MATCHING_FOLDER+'split'+str(split)+'.matches'
	#comparisons_A, templates_A = itl.read_matching_output(matches_file)

	comparisons_HO2D, _ = itl.read_matching_output(scores_base_HO2D+MATCHING_FOLDER+'split'+str(split)+'.matches')
	comparisons_HO3D, _ = itl.read_matching_output(scores_base_HO3D+MATCHING_FOLDER+'split'+str(split)+'.matches')
	comparisons_PH,   _ = itl.read_matching_output(scores_base_PH  +MATCHING_FOLDER+'split'+str(split)+'.matches')

	metadata_file_path = '/vol/vssp/datasets/still/IJB_A/11/split'+str(split)+'/verify_metadata_'+str(split)+'.csv'
	templates_dict = itl.read_IJBA_templates_definition(metadata_file_path)


	comparisons_merged = []

	with tf.Session() as sess:
		saver.restore(sess, trained_nn_score_choser)

		for i in range(len(comparisons_PH)):
			if comparisons_HO2D[i][2] != 'fte' and comparisons_HO3D[i][2] != 'fte' and comparisons_PH[i][2] != 'fte':
				fitting_log_template_A = FITTING_BASE+'split'+str(split)+'/'+str(comparisons_PH[i][0])+'/fitting.log'
				poses_mean_A, poses_std_A = oal.read_pose_from_log(fitting_log_template_A)

				fitting_log_template_B = FITTING_BASE+'split'+str(split)+'/'+str(comparisons_PH[i][1])+'/fitting.log'
				poses_mean_B, poses_std_B = oal.read_pose_from_log(fitting_log_template_B)

				scores = [comparisons_HO2D[i][2], comparisons_HO3D[i][2], comparisons_PH[i][2]]
				input_vector = np.array(poses_mean_A + poses_std_A + poses_mean_B + poses_std_B + scores)

				output = sess.run(fc3, feed_dict={input_layer: input_vector[None, :]})
				better_score = scores[np.argmax(output)]
				chosen_score[np.argmax(output)]+=1

				comparisons_merged.append([comparisons_PH[i][0], comparisons_PH[i][1], better_score])
			else:
				comparisons_merged.append([comparisons_PH[i][0], comparisons_PH[i][1], comparisons_HO2D[i][2]])
				chosen_score[0]+=1

	itl.write_matching_output(comparisons_merged, templates_dict, verification_exp_base+MATCHING_FOLDER+'split'+str(split)+'.matches')
	itl.write_sim_matrix(comparisons_merged, templates_dict, verification_exp_base+MATCHING_FOLDER+'split'+str(split)+'.simmmatrix')
print (chosen_score)
