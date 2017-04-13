#!/usr/bin/env python3.5

import glob, os, sys
import eos_starter_lib as esl
from concurrent.futures import ThreadPoolExecutor
import random

EXE = "/user/HS204/m09113/eos/eos_build14/examples/fit-model-multi-alpha-exp"

LOGNAME = "fitting.log"

OUTPUTBASE = "/vol/vssp/facer2vm/people/Philipp/KF-ITW-prerelease_alpha_experiments/"

message = ""

OVERWRITE = True


id_folders = glob.glob("/user/HS204/m09113/facer2vm_project_area/data/KF-ITW-prerelease/*")
#id_folders = os.walk("/user/HS204/m09113/facer2vm_project_area/data/KF-ITW-prerelease/").next()[1]

#list of experiments to do: for example [20, 1] means 20 fittings with 1 randomly selected image each
experiments = [ [40, 1], [40, 2], [40, 3], [30, 4], [30, 5] , [20, 7], [20, 10], [20, 20], [15, 30], [10, 50], [10, 70], [10, 90] ]


with ThreadPoolExecutor(max_workers=20) as executor:
	for n in range(0,len(id_folders)):
		id_folder = id_folders[n]
	
		# make absolute
		id_folder = os.path.abspath(id_folder)	
	
		# check if it's a folder
		if (not os.path.isdir(id_folder)):
			continue;		
	
		expressions = next(os.walk(id_folder))[1]
	
		for exp in expressions:

		
			expression = "/"+exp+"/"
			id_and_expr_folder = id_folder + expression

			# gather lm and img files
			lms  = glob.glob(id_and_expr_folder+"/*.pts")
			#imgs = esl.find_imgs_to_lms (lms, "*.png")

			for experiment in experiments: # each specific number of image

				#check if we have enough images with lms for this experiment
				if experiment[1]>len(lms):
					continue
					
				random.seed(experiment[1]*223)

				for set_iter in range(experiment[0]): #each time same number of images but different images

					lms_exp = random.sample(lms, experiment[1])
					#print (lms_exp)
					imgs_exp = esl.find_imgs_to_lms (lms_exp, "*.png")
		
					# create outputfolder
					outputfolder = OUTPUTBASE + os.path.basename(id_folder)
					if (not os.path.exists(outputfolder)):
						os.mkdir(outputfolder)	
		
					outputfolder += "/"+exp+"_only/"
					if (not os.path.exists(outputfolder)):
						os.mkdir(outputfolder)	
		
					outputfolder += format(experiment[1], '03d') +"_images_"+format(set_iter, '02d')+"/"
					if (not os.path.exists(outputfolder)):
						os.mkdir(outputfolder)
	
					message = "id "+os.path.basename(id_folder) + " expression " + exp + " with " + str(experiment[1])+" images, set "+str(set_iter)
					if (os.path.exists(outputfolder+'merged.obj') and not OVERWRITE):
						print ("already done", message)
						continue
			
					# prepare multi image fit command
					cmd = esl.assemble_command(EXE, lms_exp, imgs_exp, outputfolder, regularisation=30.0, iterations=75)
			
					# print id and start cmd
					executor.submit(esl.start_and_log,"multiframe fitting on "+message, cmd, None, log=outputfolder+LOGNAME) #21600

			
				



	
