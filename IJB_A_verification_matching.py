#!/usr/bin/env python3.5
import sys, os
import numpy as np
import IJB_A_template_lib as itl
import obj_analysis_lib as oal
from scipy.spatial import distance


verification_exp_base = '/user/HS204/m09113/my_project_folder/IJB_A/fr_verification_experiments/verification_exp_89/'
MATCHING_FOLDER = 'matching_cos/'
if not os.path.exists(verification_exp_base+MATCHING_FOLDER):
	os.mkdir(verification_exp_base+MATCHING_FOLDER)


for split in range(1,11):
	print ('matching split',split)
	#metadata_file_path = '/vol/vssp/datasets/still/IJB_A/11/split1/verify_metadata_1.csv'
	features_file_path = verification_exp_base+'/features/split'+str(split)+'_features.txt'
	templates_dict = itl.read_template_features(features_file_path)
	comparisons = itl.read_comparisons('/vol/vssp/datasets/still/IJB_A/11/split'+str(split)+'/verify_comparisons_'+str(split)+'.csv')
	
	#templates_dict = itl.read_IJBA_templates_definition(metadata_file_path)
	for comparison in comparisons:
		feature1 = np.array(templates_dict[comparison[0]].features)
		feature2 = np.array(templates_dict[comparison[1]].features)

		if feature1.shape[0]==0 or feature2.shape[0]==0:
			comparison.append('fte')
		else:
			score = 1 - distance.cosine(feature1, feature2)

			#score = 1/np.linalg.norm(feature1-feature2)
			comparison.append(score)
			#score_cnn = 1 - distance.cosine(feature1[:512], feature2[:512])
			#score_alp = 1 - distance.cosine(feature1[513:], feature2[513:])
			#comparison.append((score_cnn+score_alp)/2)

	itl.write_matching_output(comparisons, templates_dict, verification_exp_base+MATCHING_FOLDER+'split'+str(split)+'.matches')
	itl.write_sim_matrix(comparisons, templates_dict, verification_exp_base+MATCHING_FOLDER+'split'+str(split)+'.simmmatrix')
	#also write similarity matrix

