#!/usr/bin/env python3.5

import glob, os, sys
import eos_starter_lib as esl
from concurrent.futures import ThreadPoolExecutor
import random
import shutil

EXE = "/user/HS204/m09113/eos/eos_build14/examples/fit-model-multi-alpha-exp"

LOGNAME = "fitting.log"

OUTPUTBASE = "/vol/vssp/facer2vm/people/Philipp/KF-ITW-prerelease_alpha_experiments/"

message = ""

KF_ITW_POSE_FILE = "/user/HS204/m09113/facer2vm_project_area/data/KF-ITW-prerelease/KF-ITW_pose_DB.csv"

NUMBER_OF_FITTINGS_PER_EXPERIMENT = 10

pose_db=[]

with open(KF_ITW_POSE_FILE, "r") as dbf:
	dbf.readline() # header
	for line in dbf:
		pose_db.append(line.split())




id_folders = glob.glob("/user/HS204/m09113/facer2vm_project_area/data/KF-ITW-prerelease/*")
#id_folders = os.walk("/user/HS204/m09113/facer2vm_project_area/data/KF-ITW-prerelease/").next()[1]

with ThreadPoolExecutor(max_workers=30) as executor:
	for n in range(0,len(id_folders)):
		id_folder = id_folders[n]
	
		# make absolute
		id_folder = os.path.abspath(id_folder)	
	
		# check if it's a folder
		if (not os.path.isdir(id_folder)):
			continue;		

		person_id = id_folder[-2:]
	
		expressions = next(os.walk(id_folder))[1]
	
		for exp in expressions:

			expression = "/"+exp+"/"
			id_and_expr_folder = id_folder + expression
			random.seed(exp+person_id)


			group_20_left =[]
			group_10_left =[]
			group_10_centre =[]
			group_10_right =[]
			group_20_right =[]



			for pose_entry in pose_db:
				if pose_entry[0]==person_id and pose_entry[1]==exp: #same person and expression
					yaw = float(pose_entry[3])
					if yaw < -20:
						group_20_left.append(id_and_expr_folder+pose_entry[2]+".pts")
					elif yaw < -10 and yaw >-20:
						group_10_left.append(id_and_expr_folder+pose_entry[2]+".pts")
					elif yaw > -10 and yaw <10:
						group_10_centre.append(id_and_expr_folder+pose_entry[2]+".pts")
					elif yaw > 10 and yaw <20:
						group_10_right.append(id_and_expr_folder+pose_entry[2]+".pts")
					elif yaw > 20:
						group_20_right.append(id_and_expr_folder+pose_entry[2]+".pts")

			print ("in",id_and_expr_folder,"found",len(group_20_left),len(group_10_left),len(group_10_centre),len(group_10_right),len(group_20_right),)

			for set_iter in range(NUMBER_OF_FITTINGS_PER_EXPERIMENT):

				# create outpurfolders
				outputfolder = OUTPUTBASE + os.path.basename(id_folder)
				if (not os.path.exists(outputfolder)):
					os.mkdir(outputfolder)		

				outputfolder += "/"+exp+"_only/"
				if (not os.path.exists(outputfolder)):
					os.mkdir(outputfolder)	

				for experiment_idx in range(9,10):

					#### for each pose experiment
					outputfolder_experiment = outputfolder + "pose_exp_"+format(experiment_idx, '02d')+"_"+format(set_iter, '02d')+"/"
					if (not os.path.exists(outputfolder_experiment)):
						os.mkdir(outputfolder_experiment)
					else:
						shutil.rmtree(outputfolder_experiment)
						os.mkdir(outputfolder_experiment)
	

					if experiment_idx==0:
						lms  = random.sample(group_20_left,2)+random.sample(group_10_left,2)+random.sample(group_10_centre,2)+random.sample(group_10_right,2)+random.sample(group_20_right,2)
					elif experiment_idx==1:
						lms  = random.sample(group_20_left,5)+random.sample(group_20_right,5)
					elif experiment_idx==2:
						lms  = random.sample(group_20_left,2)+random.sample(group_10_left,3)+random.sample(group_10_right,3)+random.sample(group_20_right,2)
					elif experiment_idx==3:
						lms  = random.sample(group_10_left,4)+random.sample(group_10_centre,2)+random.sample(group_10_right,4)
					elif experiment_idx==4:
						lms = random.sample(group_10_left,2)+random.sample(group_10_centre,6)+random.sample(group_10_right,2)
					elif experiment_idx==5:
						lms = random.sample(group_10_centre,10)
					elif experiment_idx==6:
						lms  = random.sample(group_20_left,5)+random.sample(group_10_left,4)+random.sample(group_10_centre,1)
					elif experiment_idx==7:
						lms  = random.sample(group_10_centre,1)+random.sample(group_10_right,4)+random.sample(group_20_right,5)
					elif experiment_idx==8:
						lms  = random.sample(group_20_left,3)+random.sample(group_10_centre,4)+random.sample(group_20_right,3)
					elif experiment_idx==9:
						lms  = random.sample(group_20_left,1)+random.sample(group_10_left,4)+random.sample(group_10_right,4)+random.sample(group_20_right,1)
					imgs = esl.find_imgs_to_lms (lms, "*.png")
	
	
					message = "id "+os.path.basename(id_folder) + " expression " + exp + " pose experiment "+format(experiment_idx, '02d')+", set "+str(set_iter)
					# prepare multi image fit command
					cmd = esl.assemble_command(EXE, lms, imgs, outputfolder_experiment, regularisation=30.0, iterations=75)
				
					# print id and start cmd
					executor.submit(esl.start_and_log,"multiframe fitting on "+message, cmd, None, log=outputfolder_experiment+LOGNAME)




	
