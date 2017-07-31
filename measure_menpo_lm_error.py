#!/usr/bin/env python3.5

import sys
import numpy as np
import cv2
import glob
import json
from math import sqrt




def read_lms_from_fitting_log(log_path):
	lines=[]
	with open(log_path, "r") as fitting_log:
		for line in fitting_log:
			lines.append(line)

	all_lms={}
	parse=False
	for line in lines:
		if line.startswith('Finished fitting') or len(line)<=1:
			parse=False

		if parse:
			parts=line.split(', ')
			lm_file = parts[0]
			lms = [float(x) for x in parts[22:-1]]
			lm_file_id = lm_file.split('/')[-3]+' '+lm_file.split('/')[-1][:-4]
			#print (lm_file_id)
			#print(len(lms))
			lms2 = [[lms[x+1], lms[x]] for x in range(0,len(lms),2)]
			mapped_lms = [lms2[x] for x in fitting_lm_mapping ]
			all_lms[lm_file_id]= mapped_lms

		if line.startswith('lm_file,'):
			parse=True

	#if len(all_lms)==0:
	#	print('no poses found in fitting log file', fitting_log_file)
	#	raise OalException

	return all_lms

def read_lms_from_pts(path):
	"""A helper function to read ibug .pts landmarks from a file."""
	#print (path)
	lines = open(path).read().splitlines()
	if ICCR_LMS_USED:
		lines = lines[3:69]
	else:
		lines = lines[3:71]

	landmarks = []
	for l in lines:
		coords = l.split()
		landmarks.append([float(coords[1]), float(coords[0])])
		#landmarks.append([float(coords[0]), float(coords[1])])
	#print (landmarks)
	return landmarks

def read_all_pts_of_video(path):
	all_files = glob.glob(path+'*.pts')
	all_lms={}
	for file in all_files:
		lms = read_lms_from_pts(file)
		mapped_lms = [lms[x] for x in fitting_lm_mapping ]
		lm_file_id = file.split('/')[-3]+' '+file.split('/')[-1][:-4]
		all_lms[lm_file_id]= mapped_lms
	return all_lms

def read_gt_lm_file(path):
	with open(path, 'r') as json_data:
		d = json.load(json_data)
		#print (type(d))
		#print (d.keys())
		lms = d["landmarks"]["points"]
		#print (lms)
		return lms


ICCR_LMS_USED = True

GT_BASE = '/user/HS204/m09113/facer2vm_project_area/data/Menpo_3d_challenge/Menpo_3d_challenge_trainset_videos/final_published/projected_image_space/'
FITTING_BASE = '/user/HS204/m09113/my_project_folder/menpo_challenge/300vw_trainingsset_fittings/multi_iter75_reg30/'
#FITTING_BASE = '/user/HS204/m09113/my_project_folder/menpo_challenge/300vw_trainingsset_fittings/CSR_lm_tracking_multi_iter75_reg30/'
#FITTING_BASE = '/user/HS204/m09113/my_project_folder/menpo_challenge/300vw_trainingsset_fittings/CSR_bb_tracking_multi_iter75_reg30/'
DETECTED_LMS_BASE = '/user/HS204/m09113/facer2vm_project_area/data/300VW_Dataset_2015_12_14/'

if ICCR_LMS_USED:
	fitting_lm_mapping = list(range(17,66))
else:
	fitting_lm_mapping = list(range(17,68))

#print (fitting_lm_mapping)
if ICCR_LMS_USED:
	gt_lm_mapping = list(range(33,76))+list(range(77,80))+list(range(81,84))
else:
	gt_lm_mapping = list(range(33,84))
OUTER_EYE_INDICES_GT_AFTER_MAPPING = [19, 28]
#print (gt_lm_mapping)
#print (len(fitting_lm_mapping), len(gt_lm_mapping))

videos = glob.glob(GT_BASE + '*')

all_images_mean_dist = []
for v, video in enumerate(videos):
	print ('video',v,'of',len(videos))

	# get ground truth landmarks for this video
	all_gt_lms = {}
	if True==True:
		all_gt_lm_paths = glob.glob(video+'/*.ljson')
		for gt_lm_path in all_gt_lm_paths:
			lm_file_id = gt_lm_path.split('/')[-2]+' '+gt_lm_path[-12:-6]
			lms = read_gt_lm_file(gt_lm_path)
			all_gt_lms[lm_file_id] = [lms[x] for x in gt_lm_mapping]
	elif True==False:
		all_gt_lms = read_all_pts_of_video(DETECTED_LMS_BASE+video.split('/')[-1]+'/annot/')


	#get our reprojected lms for this video
	if True==False:
		fitting_log = FITTING_BASE+video[-3:]+'/fitting.log'
		our_reprojected_lms = read_lms_from_fitting_log(fitting_log)
	elif True==True:
		#our_reprojected_lms = read_all_pts_of_video(DETECTED_LMS_BASE+video.split('/')[-1]+'/annot/')
		#our_reprojected_lms = read_all_pts_of_video(DETECTED_LMS_BASE+video.split('/')[-1]+'/CSR_lms/')
		#our_reprojected_lms = read_all_pts_of_video(DETECTED_LMS_BASE+video.split('/')[-1]+'/CSR_lms_lm_tracking/')
		#our_reprojected_lms = read_all_pts_of_video(DETECTED_LMS_BASE+video.split('/')[-1]+'/CSR_lms_rcnnBB/')
		our_reprojected_lms = read_all_pts_of_video(DETECTED_LMS_BASE+video.split('/')[-1]+'/iccr_lms/')



	video_frames_mean_dist=[]
	for key in all_gt_lms.keys():
		#print (key)
		#print ('gt', len(all_gt_lms[key]), all_gt_lms[key])
		#print ('ours', len(our_reprojected_lms[key]), our_reprojected_lms[key])
		#try:
		distance = 0
		for lmi in range(len(all_gt_lms[key])):
			#print (sqrt( (all_gt_lms[key][lmi][0]-our_reprojected_lms[key][lmi][0])**2 + (all_gt_lms[key][lmi][1]-our_reprojected_lms[key][lmi][1])**2  ))
			distance += sqrt( (all_gt_lms[key][lmi][0]-our_reprojected_lms[key][lmi][0])**2 + (all_gt_lms[key][lmi][1]-our_reprojected_lms[key][lmi][1])**2  )
		interocular_distance = sqrt( (all_gt_lms[key][OUTER_EYE_INDICES_GT_AFTER_MAPPING[0]][0] - all_gt_lms[key][OUTER_EYE_INDICES_GT_AFTER_MAPPING[1]][0])**2 + (all_gt_lms[key][OUTER_EYE_INDICES_GT_AFTER_MAPPING[0]][1] - all_gt_lms[key][OUTER_EYE_INDICES_GT_AFTER_MAPPING[1]][1])**2 )

		mean_dist = distance/( len(all_gt_lms[key]) * interocular_distance)
		#if mean_dist>0.1:
		#	print (video, key, mean_dist)
		#print ('mean distance of inner face landmarks in video',video,'is',mean_dist)
		all_images_mean_dist.append(mean_dist)	

		#except:
		#	pass

	video_error_mean = np.mean(all_images_mean_dist[-len(all_gt_lms):])
	#if video_error_mean > 0.15:
	print ('error on video', video, 'is', video_error_mean)

	#print (len(all_gt_lms))
	#print (len(our_meareprojected_lms))
	#exit(0)
print ('overall mean distance of inner face landmarks and fittings', FITTING_BASE, 'is', np.mean(all_images_mean_dist))


