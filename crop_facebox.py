#!/usr/bin/env python3.5

import glob, os
import eos_starter_lib as esl
from concurrent.futures import ThreadPoolExecutor
import subprocess, shlex

#id_folders = glob.glob("/user/HS204/m09113/my_project_folder/CASIA_webface/landmarks/*")
#id_folders = glob.glob("/user/HS204/m09113/my_project_folder/PaSC/video/data_CCR/*")
#id_folders = glob.glob("/user/HS204/m09113/my_project_folder/PaSC/still/data_CCR/0246*")
#id_folders = glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/input_org/*")
#id_folders = glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/input_org/frame/*.pts")

#id_folders = glob.glob("/user/HS204/m09113/my_project_folder/CASIA_webface/landmarks/180180*")
#images_base = '/vol/vssp/datasets/still/CASIA-WebFace/CASIA-WebFace/'

#OUTPUTBASE = "/user/HS204/m09113/my_project_folder/CASIA_webface/face_boxes/"
#OUTPUTBASE = "/user/HS204/m09113/my_project_folder/PaSC/video/face_boxes/"
#OUTPUTBASE = "/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30/"

#print (id_folders)


#def read_pts(filename):
#	"""A helper function to read ibug .pts landmarks from a file."""
#	lines = open(filename).read().splitlines()
#	lines = lines[3:71]
#
#	landmarks = []
#	for l in lines:
#		coords = l.split()
#		landmarks.append([float(coords[0]), float(coords[1])])
#
#	return landmarks
#
#def run (message, exe_list):
#	if not message=='':
#		print (message)
#	#subprocess.run(exe_list, timeout=timeout_sec, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#	completed = subprocess.run(exe_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#	err = completed.stderr.decode('utf-8')
#	std = completed.stdout.decode('utf-8')
#	if not err == '':
#		print ('err',err)
#
#
#
#
#with ThreadPoolExecutor(max_workers=30) as executor:
#	for n in range(0,len(id_folders)):
#		id_folder = id_folders[n]
	#
#		# make absolute
#		id_folder = os.path.abspath(id_folder)	
#
#		#print (id_folder)
		#
	#
#		# check if it's a folder
#		if (not os.path.isdir(id_folder)):
#			continue;		
	#
#		try:
		#
#			# create outputfolder
#			#outputfolder = OUTPUTBASE + os.path.basename(id_folder)+"/"
#			#if (not os.path.exists(outputfolder)):
#			#	print('ohohoh', outputfolder)
#				#os.mkdir(outputfolder)	
#
#			#exit(0)
#			message = "video "+os.path.basename(id_folder) +" ("+str(n)+" of "+str(len(id_folders))+" )"
#
#			# gather lm and img files
#			lms  = glob.glob(id_folder+"/*.pts")
#			#lms = [id_folder]
#			#imgs = [] #esl.find_imgs_to_lms (lms, ".jpg")	
#			imgs = esl.find_imgs_to_lms (lms, ".jpg")	
#			#imgs = esl.find_imgs_to_lms (lms, ".*[!pts]")
#			for i in range(len(lms)):
#				#img_source = images_base+lms[i][-15:-3]+'jpg'
#				#img_dest = OUTPUTBASE+lms[i][-15:-3]+'jpg'
#				img_source = imgs[i]
#				#img_dest =  img_source.replace('data_CCR', 'face_boxes')
#				img_dest =  img_source.replace('input_org', 'face_boxes')
#
#				#print (img_source)
#				#print (img_dest)
				#
#				landmarks = read_pts(lms[i])
#				height_delta = abs(landmarks[28-1][1]-landmarks[9-1][1])*2.5
#				height_offset = landmarks[28-1][1]-height_delta/2.5*1.2
#				width_delta = height_delta
#				width_offset = (landmarks[28-1][0]+landmarks[9-1][0])/2-width_delta/2
#
#				exe_list = ['convert', img_source, '-crop', str(width_delta)+'x'+str(height_delta)+'+'+str(width_offset)+'+'+str(height_offset), '-resize', '512x512', img_dest]
#				executor.submit(run,message, exe_list)
#				message = ''
#				#exit(0)
#		except Exception as e:
#			print("ERROR on " + message + ": " + str(e))	
				#
import cv2
import numpy as np

def fix_img (img_path):
	img = cv2.imread(img_path)
	width, height, _ = img.shape
	if width != 512 and height==512:
		print ('adding rows to', img_path)
		zeros = np.zeros((512-width, 512,3))
		#print (img.shape)
		#print (zeros.shape)
		new = np.concatenate((zeros, img), axis=0)
		#print (new.shape)
		cv2.imwrite(img_path, new)
	elif height != 512 and width==512:
		print ('adding cols to', img_path)
		zeros = np.zeros((512, 512-height,3))
		#print (img.shape)
		#print (zeros.shape)
		new = np.concatenate((zeros, img), axis=1)
		#print (new.shape)
		cv2.imwrite(img_path, new)
	elif width != 512 and height != 512:
		print ('adding rows and cols to', img_path)
		zeros1 = np.zeros((512-width, height,3))
		tmp = np.concatenate((zeros1, img), axis=0)
		zeros2 = np.zeros((512, 512-height,3))
		new = np.concatenate((zeros2, tmp), axis=1)
		#cv2.imwrite(img_path.replace('.jpg','_new.jpg'), new)
		#exit(0)
		cv2.imwrite(img_path, new)






#images = glob.glob("/user/HS204/m09113/my_project_folder/PaSC/video/face_boxes/*/*")
images = glob.glob("/user/HS204/m09113/my_project_folder/CASIA_webface/face_boxes/*/*")
#images = glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/face_boxes/*/*")
print ('number of found files', len(images))
#index = images.index('/user/HS204/m09113/my_project_folder/PaSC/video/face_boxes/06051d661/06051d661-217.jpg')
#print ('index:',index)
#exit(0)

with ThreadPoolExecutor(max_workers=35) as executor:
	for img_path in images:
		executor.submit(fix_img,img_path)
	











