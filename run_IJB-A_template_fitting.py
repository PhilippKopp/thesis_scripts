#!/usr/bin/env python3.5
import sys, os
import numpy as np
import eos_starter_lib as esl
import IJB_A_template_lib as itl
from concurrent.futures import ThreadPoolExecutor
#from scipy.spatial import distance



EXE = "/user/HS204/m09113/eos/eos_build14/examples/fit-model-multi-cnn-exp"

LOGNAME = "fitting.log"

INPUT_BASE='/user/HS204/m09113/my_project_folder/IJB_A/input_org/'


message = None



with ThreadPoolExecutor(max_workers=39) as executor:

	for split in range(1,11):
		#metadata_file_path = '/vol/vssp/datasets/still/IJB_A/11/split1/verify_metadata_1.csv'
		metadata_file_path = '/vol/vssp/datasets/still/IJB_A/11/split'+str(split)+'/verify_metadata_'+str(split)+'.csv'
		output_base = '/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256/verification_templates/split'+str(split)+'/'
		templates_dict = itl.read_IJBA_templates_definition(metadata_file_path)

		os.mkdir(output_base)

		for n, template in enumerate(templates_dict.items()):
			#try:
				
			message = "template "+str(template[1].template_id) + "   ("+str(n)+" of "+ str(len(templates_dict)) +") in split "+str(split)
	
			# check if it's a folder
			#if (not os.path.isdir(id_folder)):
			#	continue;		

			# create outputfolder
			outputfolder = output_base+str(template[1].template_id)+"/"
			if (not os.path.exists(outputfolder)):
				os.mkdir(outputfolder)	
			else:
				print("something is going wrong here!!")

	
			# gather lm and img files
			imgs = [INPUT_BASE+x for x in template[1].images]
			imgs = [img for img in imgs if os.path.exists(img)]

			if len(imgs)==0:
				print('fuck!!! ohohohohoh', template[1].images)
				with open(outputfolder+LOGNAME, 'w') as out:
					out.write('no images with lms found')
				continue

			lms = [x.split('.')[0]+'.pts' for x in imgs]

			#lms  = glob.glob(id_folder+"/*.pts")
			#imgs = esl.find_imgs_to_lms (lms, ".*[!pts]")
	
	
			# prepare multi image fit command
			cmd = esl.assemble_command(EXE, lms, imgs, outputfolder, regularisation=30.0, iterations=75)
	
			# print id and start cmd
			
			executor.submit(esl.start_and_log,"multiframe fitting on "+message, cmd, None, log=outputfolder+LOGNAME) #21600
			#esl.start_and_log("multiframe fitting on "+message, cmd, None, log=outputfolder+LOGNAME) #21600
			
			#except Exception as e:
			#	print("ERROR on " + message + ": " + str(e))
