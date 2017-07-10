#!/usr/bin/env python3.5

import glob, os, sys
import eos_starter_lib as esl
from concurrent.futures import ThreadPoolExecutor

EXE = "/user/HS204/m09113/eos/eos_build14/examples/fit-model-multi-shape-exp"

LOGNAME = "fitting.log"

OUTPUTBASE = "/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter10_reg30_mesh_renderings/"

message = ""

OVERWRITE = True

if (not os.path.exists(OUTPUTBASE)):
	os.mkdir(OUTPUTBASE)


lm_files = glob.glob("/user/HS204/m09113/facer2vm_project_area/data/AFLW2000-3D/images/*pts")

with ThreadPoolExecutor(max_workers=40) as executor:
	for n in range(0,len(lm_files)):
		lm_file = lm_files[n]
		#vid_folder = vid_folders[n]
	
		# make absolute
		lm_file = os.path.abspath(lm_file)	
	
		# check if it's a folder
		if (os.path.isdir(lm_file)):
			continue;		
	
		try:
	
			# create outputfolder
			outputfolder = OUTPUTBASE + os.path.basename(lm_file)[:-8]+"/"
			if (not os.path.exists(outputfolder)):
				os.mkdir(outputfolder)	


			message = "image "+os.path.basename(lm_file) +" ("+str(n)+" of "+str(len(lm_files))+" )"
			#if (os.path.exists(outputfolder+'merged.obj') and not OVERWRITE):
			#	print ("already done", message)
			#	continue
	
			# gather lm and img files
			#lms  = glob.glob(vid_folder+"/*.pts")
			img_file = esl.find_imgs_to_lms ([lm_file[:-8]+'.pts'], "*.jpg")[0]

			#print (lm_file)
			#print (img_file)
	
			# prepare multi image fit command
			cmd = esl.assemble_command(EXE, [lm_file], [img_file], outputfolder, regularisation=30.0, iterations=10, model="3.4k")
			#cmd.replace("/user/HS204/m09113/eos/install/share/ibug_to_sfm.txt", "/user/HS204/m09113/my_project_folder/AFLW2000_fittings/ibug_to_sfm_static_contour.txt")
			#print (cmd)
			#exit()
	
			# print id and start cmd
			executor.submit(esl.start_and_log,"multiframe fitting on "+message, cmd, None, log=outputfolder+LOGNAME) #21600
			
		except Exception as e:
			print("ERROR on " + message + ": " + str(e))	
