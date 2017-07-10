#!/usr/bin/env python3.5

import glob, os, sys
import eos_starter_lib as esl
from concurrent.futures import ThreadPoolExecutor

EXE = "/user/HS204/m09113/eos/eos_build14/examples/fit-model-multi-cnn-exp"

LOGNAME = "fitting.log"

OUTPUTBASE = "/user/HS204/m09113/my_project_folder/PaSC/still/multi_fit_CCR_iter75_reg30_256/"
#OUTPUTBASE = "/user/HS204/m09113/my_project_folder/PaSC/video/multi_fit_CCR_iter75_reg30_256/"

message = ""

OVERWRITE = True




vid_folders = glob.glob("/user/HS204/m09113/my_project_folder/PaSC/still/data_CCR/*")
#vid_folders = glob.glob("/user/HS204/m09113/my_project_folder/PaSC/video/data_CCR/*")

with ThreadPoolExecutor(max_workers=30) as executor:
	for n in range(0,len(vid_folders)):
		vid_folder = vid_folders[n]
	
		# make absolute
		vid_folder = os.path.abspath(vid_folder)	
	
		# check if it's a folder
		if (not os.path.isdir(vid_folder)):
			continue;		
	
		try:
	
			# create outputfolder
			outputfolder = OUTPUTBASE + os.path.basename(vid_folder)+"/"
			if (not os.path.exists(outputfolder)):
				os.mkdir(outputfolder)	


			message = "video "+os.path.basename(vid_folder) +" ("+str(n)+" of "+str(len(vid_folders))+" )"
			if (os.path.exists(outputfolder+'merged.obj') and not OVERWRITE):
				print ("already done", message)
				continue
	
			# gather lm and img files
			lms  = glob.glob(vid_folder+"/*.pts")
			imgs = esl.find_imgs_to_lms (lms, ".jpg")	
	
	
			# prepare multi image fit command
			cmd = esl.assemble_command(EXE, lms, imgs, outputfolder, regularisation=30.0, iterations=75)
	
			# print id and start cmd
			executor.submit(esl.start_and_log,"multiframe fitting on "+message, cmd, None, log=outputfolder+LOGNAME) #21600
			
		except Exception as e:
			print("ERROR on " + message + ": " + str(e))	