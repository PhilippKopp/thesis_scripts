#!/usr/bin/env python3.5

import scipy.io
import os
import numpy as np
from glob import glob

AFLW2000_path = '/user/HS204/m09113/facer2vm_project_area/people/Philipp/3DDFA_Release/Data_KFLW/samples/images/'

mat_files = glob(AFLW2000_path+'*3DDFA.mat')

print ('mat_files',len(mat_files))
for mat_file in mat_files:
	lms = scipy.io.loadmat(mat_file)['pt2d_3d']
	image_name = os.path.basename(mat_file)[:-10]
	pts_file_name = image_name+'_eos.pts'
	print (image_name)
	print (lms.shape)
	with open(AFLW2000_path+pts_file_name, 'w') as pts_file:
		pts_file.write("version: 1\n")
		pts_file.write("n_points: 68\n")
		pts_file.write("{\n")
		for i in range(68):
			pts_file.write(str(lms[0,i])+' '+str(lms[1,i])+'\n')
		pts_file.write("}")





