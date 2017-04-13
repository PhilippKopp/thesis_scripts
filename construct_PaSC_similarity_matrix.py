#!/usr/bin/env python3.5
import sys, os
import numpy as np
import xml.etree.ElementTree as ET
import obj_analysis_lib as oal
from scipy.spatial import distance



query_file = '/vol/vssp/datasets/multiview02/PaSC/Protocol/PaSC_20130611/PaSC/metadata/sigsets/pasc_video_control.xml'
#query_file = '/vol/vssp/datasets/multiview02/PaSC/Protocol/PaSC_20130611/PaSC/metadata/sigsets/pasc_video_handheld.xml'

#target_file = '/vol/vssp/datasets/multiview02/PaSC/Protocol/PaSC_20130611/PaSC/metadata/sigsets/pasc_video_control.xml'

FITTING_RESULTS_BASE = '/user/HS204/m09113/my_project_folder/PaSC/multi_fit_CCR_iter75_reg30_only_10_alphas/'

OUTPUT_FILE = '/user/HS204/m09113/my_project_folder/PaSC/multi_fit_CCR_iter75_reg30_only_10_alphas_control_without_fte.csv'

QUERY_AND_TARGET_SAME = True

query = ET.parse(query_file)
query_root = query.getroot()


#print (query_root[0][0])
query_db = []
for i, query_video in enumerate(query_root):
	query_video_name = query_video[0].attrib['file-name']
	alphas = None
	try:
		query_alphas, _ = oal.read_fitting_log(FITTING_RESULTS_BASE+query_video_name[:-4]+'/fitting.log')
		query_alphas = np.array(query_alphas)
		#query_alphas = query_alphas/np.linalg.norm(query_alphas)
	except oal.OalException:
		print ("No alphas found in",query_video_name)
	except FileNotFoundError:
		print ("Video not found",query_video_name)
	query_id_name = query_video.attrib['name']
	query_db.append([query_id_name, query_alphas])
	if i%100==0:
		print ('loaded',i,'of',len(query_root))


print ('measuring distance...')
if QUERY_AND_TARGET_SAME:
	target_db = query_db

same_id =[]
different_id=[]
for query_idx, query_element in enumerate(query_db):
	for target_idx, target_element in enumerate(target_db):
		if query_element[1] is None or target_element[1] is None:
			#score = 0.5
			continue
		elif QUERY_AND_TARGET_SAME and query_idx==target_idx:
			continue
		else:
			score = 1 - distance.cosine(query_element[1], target_element[1])
			#score = 1 - math.acos(score)/math.pi

		if query_element[0]==target_element[0]:
			same_id.append(score)
		else:
			different_id.append(score)
		#print (score)
	#print (query_db)
#print ('max',max(score))
#print ('min',min(score))

print ('writing roc plot file',OUTPUT_FILE,'...')
with open(OUTPUT_FILE, "w") as results:
	for i in same_id:
		results.write(str(i)+" ")
	results.write("\n")
	results.write("\n")
	for i in different_id:
		results.write(str(i)+" ")
	results.write("\n")

