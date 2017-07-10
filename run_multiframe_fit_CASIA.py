#!/usr/bin/env python3.5

import glob, os, sys
import eos_starter_lib as esl
from concurrent.futures import ThreadPoolExecutor

EXE = "/user/HS204/m09113/eos/eos_build14/examples/fit-model-multi-cnn-exp"

LOGNAME = "fitting.log"

OUTPUTBASE = "/user/HS204/m09113/my_project_folder/CASIA_webface/multi_fit_CCR_iter75_reg30_256/"

message = ""

OVERWRITE = True




id_folders = glob.glob("/user/HS204/m09113/my_project_folder/CASIA_webface/landmarks/*")
images_base = '/vol/vssp/datasets/still/CASIA-WebFace/CASIA-WebFace/'

with ThreadPoolExecutor(max_workers=30) as executor:
	for n in range(0,len(id_folders)):
		id_folder = id_folders[n]
	
		# make absolute
		id_folder = os.path.abspath(id_folder)	
	
		# check if it's a folder
		if (not os.path.isdir(id_folder)):
			continue;		
	
		try:
	
			# create outputfolder
			outputfolder = OUTPUTBASE + os.path.basename(id_folder)+"/"
			if (not os.path.exists(outputfolder)):
				os.mkdir(outputfolder)	


			message = "video "+os.path.basename(id_folder) +" ("+str(n)+" of "+str(len(id_folders))+" )"
			if (os.path.exists(outputfolder+'merged.obj') and not OVERWRITE):
				print ("already done", message)
				continue
	
			# gather lm and img files
			lms  = glob.glob(id_folder+"/*.pts")
			imgs = [] #esl.find_imgs_to_lms (lms, ".jpg")	
			for i in range(len(lms)):
				imgs.append(images_base+lms[i][-15:-3]+'jpg')
			
			#print (lms[:3])
			#print (imgs[:3])
			#exit()
	
			# prepare multi image fit command
			cmd = esl.assemble_command(EXE, lms, imgs, outputfolder, regularisation=30.0, iterations=75)
	
			# print id and start cmd
			executor.submit(esl.start_and_log,"multiframe fitting on "+message, cmd, None, log=outputfolder+LOGNAME) #21600
			
		except Exception as e:
			print("ERROR on " + message + ": " + str(e))	