#!/usr/bin/env python3.5

import glob, os, sys
import datetime
import eos_starter_lib as esl

EXE = "/user/HS204/m09113/eos/eos_build/examples/fit-model-shape-exp"

LOGNAME = "fitting.log"

OUTPUTBASE = "/vol/vssp/facer2vm/people/Philipp/KF-ITW-prerelease/"

EXPERIMENT = "single_iter400_reg30/"

message = ""

OVERWRITE = False




id_folders = glob.glob("/user/HS204/m09113/facer2vm_project_area/data/KF-ITW-prerelease/*")
#id_folders = os.walk("/user/HS204/m09113/facer2vm_project_area/data/KF-ITW-prerelease/").next()[1]
for n in range(0,len(id_folders)):
	id_folder = id_folders[n]

	# make absolute
	id_folder = os.path.abspath(id_folder)  

	# check if it's a folder
	if (not os.path.isdir(id_folder)):
		continue;       

	if (not os.path.isdir(id_folder)):
		continue

	expressions = next(os.walk(id_folder))[1]

	for exp in expressions:

		expression = "/"+exp+"/"

		# create outputfolder
		outputfolder = OUTPUTBASE + os.path.basename(id_folder)
		if (not os.path.exists(outputfolder)):
			os.mkdir(outputfolder)  

		outputfolder += expression
		if (not os.path.exists(outputfolder)):
			os.mkdir(outputfolder)  

		outputfolder += EXPERIMENT
		if (not os.path.exists(outputfolder)):
			os.mkdir(outputfolder)

		id_and_expr_folder = id_folder + expression     

		# gather lm and img files
		lms  = glob.glob(id_and_expr_folder+"/*.pts")
		imgs = esl.find_imgs_to_lms (lms, "*.png")  

		for i in range(len(lms)):
			try:
		
				# prepare multi image fit command
				img_num = os.path.splitext(os.path.basename(lms[i]))[0]
				cmd = esl.assemble_command(EXE, lms[i], imgs[i], outputfolder+img_num)
	
				#print (cmd)
		
				# print id and start cmd
				message = "id "+os.path.basename(id_folder) + " expression " + exp + " img num "+ img_num
				if (os.path.exists(outputfolder+img_num+'.png') and not OVERWRITE):
					print ("already done", message)
					continue

				print ("single frame fitting on",message)
				with open(outputfolder+img_num+"."+LOGNAME, "w") as logfile:
					logfile.write(cmd+"\n \n")
					logfile.write(str(datetime.datetime.now())+"\n \n")
					stdout, stderr = esl.run(cmd, 600) # 60sec/min * 40 min = 2400 sec 
					logfile.write(stdout + "\n \n")
					logfile.write(stderr + "\n \n")
					logfile.write(str(datetime.datetime.now()))
	
			except Exception as e:
				print("ERROR on " + message + ": " + str(e))

	
