#!/usr/bin/env python3.5

import glob, os, sys
import eos_starter_lib as esl
from concurrent.futures import ThreadPoolExecutor

EXE = "/user/HS204/m09113/eos/eos_build14/examples/fit-model-multi-shape-exp"

LOGNAME = "fitting.log"

OUTPUTBASE = "/vol/vssp/facer2vm/people/Philipp/KF-ITW-prerelease/"

EXPERIMENT = "multi_iter400_reg45/"

message = ""

OVERWRITE = False


id_folders = glob.glob("/user/HS204/m09113/facer2vm_project_area/data/KF-ITW-prerelease/*")
#id_folders = os.walk("/user/HS204/m09113/facer2vm_project_area/data/KF-ITW-prerelease/").next()[1]

print (EXPERIMENT)

with ThreadPoolExecutor(max_workers=20) as executor:
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
			try:
		
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
	
				message = "id "+os.path.basename(id_folder) + " expression " + exp
				if (os.path.exists(outputfolder+'merged.obj') and not OVERWRITE):
					print ("already done", message)
					continue
		
		
				id_and_expr_folder = id_folder + expression
		
		
				# gather lm and img files
				lms  = glob.glob(id_and_expr_folder+"/*.pts")
				imgs = esl.find_imgs_to_lms (lms, "*.png")	
		
		
				# prepare multi image fit command
				cmd = esl.assemble_command(EXE, lms, imgs, outputfolder, regularisation=45.0)
		
				# print id and start cmd
				executor.submit(esl.start_and_log,"multiframe fitting on "+message, cmd, None, log=outputfolder+LOGNAME) #21600
				
			except Exception as e:
				print("ERROR on " + message + ": " + str(e))



	
