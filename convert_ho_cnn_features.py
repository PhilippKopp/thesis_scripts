#!/usr/bin/env python3.5

import scipy.io as sio
import numpy as np

ho_features_file = '/user/HS204/m09113/facer2vm_project_area/Share/IJB-A_feature/cnn_feature/face_train_test_netbuild112s56_17757.mat'

ho_features_phil_style = '/user/HS204/m09113/my_project_folder/cnn_experiments/99_ho/ijba_vectors.csv'



ho_features = sio.loadmat(ho_features_file)

print (len(ho_features['image_path'][0]))
print (ho_features['image_path'][0][0])

print (len(ho_features['features']))
#print (ho_features['features'][0])

vectors = np.empty([len(ho_features['image_path'][0]), ho_features['features'][0].shape[0]])
print (vectors.shape)
image_list = []
for i, image_path in enumerate(ho_features['image_path'][0]):
	image_name = str(image_path[0])
	if image_name[:3] =='img':
		image_name = image_name.split('_')[0]
	else:
		image_name = image_name.split('_')[0]+'_'+image_name.split('_')[1]
	print (image_name)
	image_list.append(image_name)
	vectors[i,:]=ho_features['features'][i]





with open(ho_features_phil_style, 'w') as log:
			for i in range(len(image_list)):
				log.write(image_list[i]+' ')
				for x in range(vectors.shape[1]):
					log.write(str(vectors[i,x])+' ')
				log.write('\n')