#!/usr/bin/env python3.5

import glob, os, sys
import eos_starter_lib as esl
from concurrent.futures import ThreadPoolExecutor

EXE = "/user/HS204/m09113/eos/eos_build14/examples/fit-model-multi-cnn-exp"

LOGNAME = "fitting.log"

OUTPUTBASE = "/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30/"

message = None



with ThreadPoolExecutor(max_workers=10) as executor:
	id_folders = glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/input/*")
	for n in range(0,len(id_folders)):
		id_folder = id_folders[n]
		try:
			# make absolute
			id_folder = os.path.abspath(id_folder)	
			id_num = os.path.basename(id_folder)
			message = "id "+id_num + "   ("+str(n)+" of "+ str(len(id_folders)) +" )"
	
			# check if it's a folder
			#if (not os.path.isdir(id_folder)):
			#	continue;		
	
			# gather lm and img files
			lms  = glob.glob(id_folder+"/*.pts")
			imgs = esl.find_imgs_to_lms (lms, ".*[!pts]")
	
			# create outputfolder
			outputfolder = OUTPUTBASE+id_num+"/"
			if (not os.path.exists(outputfolder)):
				os.mkdir(outputfolder)	
	
			# prepare multi image fit command
			cmd = esl.assemble_command(EXE, lms, imgs, outputfolder, regularisation=30.0, iterations=75)
	
			# print id and start cmd
			
			executor.submit(esl.start_and_log,"multiframe fitting on "+message, cmd, None, log=outputfolder+LOGNAME) #21600
			#esl.start_and_log("multiframe fitting on "+message, cmd, None, log=outputfolder+LOGNAME) #21600
			
		except Exception as e:
			print("ERROR on " + message + ": " + str(e))
	
		
