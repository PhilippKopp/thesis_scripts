#!/usr/bin/env python3.5
import sys, os
import numpy as np

import glob
import obj_analysis_lib as oal


#id_and_expression_dirs = glob.glob('/user/HS204/m09113/my_project_folder/KF-ITW-prerelease_alpha_experiments/*/expression_mix')
id_and_expression_dirs = glob.glob('/user/HS204/m09113/my_project_folder/KF-ITW-prerelease_alpha_experiments/*/*')
if len(id_and_expression_dirs)==0:
	print ("ERROR: no videos found!!")
	exit(0)

DB = []
DB_FILE = '/user/HS204/m09113/facer2vm_project_area/people/Philipp/KF-ITW_pose_DB.csv'
for id_and_expression_dir in id_and_expression_dirs:
	print ('id and expression dir: ',id_and_expression_dir)

	# assemble all fitting results we find for this video and load the alphas
#	experiment_alphas =[ [] for i in range(len(categories)) ]

	#fitting_dirs = glob.glob(id_and_expression_dir+'/00[3,4,5,7]*')	
	fitting_dirs = glob.glob(id_and_expression_dir+'/*')	
	for fitting_dir in fitting_dirs:
		
		fitting_log_file = fitting_dir+'/fitting.log'
		if not os.path.exists(fitting_log_file):
			print ("ERROR: There is no fitting log file where there should be one!!", fitting_dir)
			exit(0)
	
		lines = []
		with open(fitting_log_file, "r") as fitting_log:
			for line in fitting_log:
				lines.append(line)
	
		for l in range(len(lines)):
			if lines[l].startswith("lms file:"):

				path = 	lines[l].split()[2]
				if not path in [i[0] for i in DB]:
					person = path.split('/')[7]
					expression = path.split('/')[8]
					img_num = path.split('/')[-1][:-4]
					angles = lines[l+1].split()[1::2]
					
					#DB_entry=person+" "+expression+" "+img_num+" "+angles[0]+" "+angles[1]+" "+angles[2]
					DB_entry=[path, person, expression, img_num, angles[0], angles[1], angles[2]]
					DB.append(DB_entry)



print (len(DB))
DB.sort()

with open(DB_FILE, "w") as dbf:
	dbf.write("# id expression imgNumber yaw pitch roll\n")
	for DB_entry in DB:
		line = ''
		for part in DB_entry[1:]:
			line +=part+" "
		dbf.write(line+'\n')

### write DB 