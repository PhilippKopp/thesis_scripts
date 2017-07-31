#!/usr/bin/env python3.5
import sys, os
import numpy as np
import IJB_A_template_lib as itl
import obj_analysis_lib as oal

#from scipy.spatial import distance


verification_exp_folder = '/user/HS204/m09113/my_project_folder/IJB_A/fr_verification_experiments/verification_exp_89/'
feature_output_base = verification_exp_folder + 'features/'
if not os.path.exists(verification_exp_folder):
	os.mkdir(verification_exp_folder)
if not os.path.exists(feature_output_base):
	os.mkdir(feature_output_base)

cnn_features_file = '/user/HS204/m09113/my_project_folder/cnn_experiments/99_ho/ijba_vectors.csv'
#cnn_features_file = '/user/HS204/m09113/my_project_folder/cnn_experiments/21/best1_ijba_vectors.csv'
#cnn_features_file = '/user/HS204/m09113/my_project_folder/cnn_experiments/21/merged_ijba_vectors.csv'
#cnn_features_file = '/user/HS204/m09113/my_project_folder/cnn_experiments/21/merge3_ijba_vectors.csv'


#template_fitting_base = '/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30/verification_templates/'

#cnn_features = itl.read_cnn_features_file(cnn_features_file)
cnn_features = itl.read_ho_features_file(cnn_features_file)
#cnn_features = itl.read_fb_cnn_features_file(cnn_features_file)
#cnn_features = itl.read_merged_cnn_features_file(cnn_features_file)
#print (len(cnn_features))
#print(list(cnn_features.keys())[10])

CONF_BASE = '/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256_conf13_sm/verification_templates/'

for split in range(1,11):
		print('gathering features of split', split)
		#metadata_file_path = '/vol/vssp/datasets/still/IJB_A/11/split1/verify_metadata_1.csv'
		metadata_file_path = '/vol/vssp/datasets/still/IJB_A/11/split'+str(split)+'/verify_metadata_'+str(split)+'.csv'
		templates_dict = itl.read_IJBA_templates_definition(metadata_file_path)

		###################################either option 1)2) or 3)

		####################1) averaging of all images in a template
		CONF_BASED_WEIGHTED_AVRG=True
		EXP_AVERAGING=False
		FEATURES_FROM_HO=True
		np.seterr(invalid='raise')
		for template_key in templates_dict.keys():
			#print (templates_dict[template_key])
			template_features = []
			template_confidences = []
			for img in templates_dict[template_key].images:
				#print (img)
				img_without_extension = img.split('/')[1].split('.')[0]  # for own feature vectors based on isomap
				#print (img_without_extension, 'in template', template_key)
				if FEATURES_FROM_HO:
					img_without_extension = img.split('.')[0]    # for ho feature vectors
				#print (img_without_extension)
				try:
					template_features.append(cnn_features[img_without_extension])
					if CONF_BASED_WEIGHTED_AVRG:
						confidence_file_path = CONF_BASE+'split'+str(split)+'/'+str(template_key)+'/'+img_without_extension+'.isomap_conf.npy'
						if FEATURES_FROM_HO:
							confidence_file_path = CONF_BASE+'split'+str(split)+'/'+str(template_key)+'/'+img_without_extension.split('/')[1]+'.isomap_conf.npy'
						#print(confidence_file_path)
						if os.path.exists(confidence_file_path):
							if EXP_AVERAGING:
								template_confidences.append(np.exp(np.mean(np.load(confidence_file_path))))
							else:
								template_confidences.append(np.mean(np.load(confidence_file_path)))
						else:
							template_confidences.append(0.0)
				except KeyError:
					#print(img)
					#print(img_without_extension)
					#exit(0)
					pass
			if len(template_features)>0:
				template_features_np = np.array(template_features)
				if CONF_BASED_WEIGHTED_AVRG:
					try:
						template_confidences_np = np.array(template_confidences)
						template_features_np = template_features_np*template_confidences_np[:, None]
						template_features_np/= np.sum(template_confidences_np)
						templates_dict[template_key].features = np.sum(template_features_np, axis=0).tolist()
					except FloatingPointError: #if confidence is zero
						templates_dict[template_key].features = None

				#print (template_features_np.shape)
				else:
					templates_dict[template_key].features = np.mean(template_features_np, axis=0).tolist()
			
				#alphas,_ = oal.read_fitting_log(template_fitting_base+'split'+str(split)+'/'+str(template_key)+'/fitting.log')
				#templates_dict[template_key].features += alphas
				#print (len(templates_dict[template_key].features))
			else:
				templates_dict[template_key].features = None



		########################2) just take the alphas of each template		
#		for template_key in templates_dict.keys():
#			try:
#				templates_dict[template_key].features,_ = oal.read_fitting_log(template_fitting_base+'split'+str(split)+'/'+str(template_key)+'/fitting.log')
#			except oal.OalException as e:
#				templates_dict[template_key].features = None

		#exit(0)

		########################3) images already merged and have template id names
#		for template_key in templates_dict.keys():
#			try:
#			#print(template_key)
#				features = cnn_features['split'+str(split)+'/'+str(template_key)]
#			except KeyError:
#				features = None
#			templates_dict[template_key].features = features


		itl.write_template_features(templates_dict, feature_output_base+'split'+str(split)+'_features.txt')


