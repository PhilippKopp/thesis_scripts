#!/usr/bin/env python3.5

import scipy.io
import os
import numpy as np
from glob import glob

_300vw_path = '/user/HS204/m09113/facer2vm_project_area/data/300VW_Dataset_2015_12_14/'

mat_files = glob(_300vw_path+'*/iccr_lms/all_lms.mat')

print ('mat_files',len(mat_files))
for mat_file in mat_files:
	print ('loading',mat_file)
	lms = scipy.io.loadmat(mat_file)['data'][0,0][0]
	
	number_of_lms = lms.shape[2]
	print (number_of_lms)
	for i in range(number_of_lms):
		pts_path = mat_file[:-len('all_lms.mat')] + str(i+1).zfill(6)+'.pts'

		with open(pts_path, 'w') as pts_file:
			pts_file.write("version: 1\n")
			pts_file.write("n_points: 66\n")
			pts_file.write("{\n")
			for j in range(66):
				pts_file.write(str(lms[j,0,i])+' '+str(lms[j,1,i])+'\n')
			pts_file.write("}")






