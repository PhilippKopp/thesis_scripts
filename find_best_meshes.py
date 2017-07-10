#!/usr/bin/env python3.5

import sys
from glob import glob
import numpy as np
#from cycler import cycler

OUTPUTPATH="/user/HS204/m09113/my_project_folder/Results/mesh_distances_KF-ITW_iterations_BMVC"
SAVE4PRES =None
SAVE4LATEX=None

#SAVE4LATEX=OUTPUTPATH
#SAVE4PRES =OUTPUTPATH


if SAVE4LATEX:
	import texfig
import matplotlib.pyplot as plt

if SAVE4LATEX:
	#fig = texfig.figure(width=8.268) #entire page
	fig = texfig.figure(width=4.8)

if SAVE4PRES:
	#plt.rcParams["font.family"] ="monospace"
	plt.figure(figsize=(10, 8/3*2))

# each curve has: label, marker, [log files]
#comparing every multifit
#distance_files = [ ['02 neutral','*',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/02/neutral/distances.log']],
#				   ['02 surprised','o',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/02/surprised/distances.log']],
#				   ['02 happy','+',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/02/happy/distances.log']],
#				   ['08 neutral','*',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/08/neutral/distances.log']],
#				   ['08 surprised','o',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/08/surprised/distances.log']],
#				   ['11 neutral','*',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/11/neutral/distances.log']],
#				   ['11 surprised','o',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/11/surprised/distances.log']],
#				   ['11 happy','+',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/11/happy/distances.log']],
#				   ['13 neutral','*',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/13/neutral/distances.log']],
#				   ['13 surprised','o',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/13/surprised/distances.log']],
#				   ['13 happy','+',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/13/happy/distances.log']],
#				   ['16 neutral','*',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/16/neutral/distances.log']],
#				   ['16 happy','+',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/16/happy/distances.log']],
#		]

#plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'gold', 'm', 'k', 'slategray', 'peru'])))
# (0/255, 0/255, 0/255), (000/255, 70/255, 160/255), (201/255, 169/255, 000/255), 

DB_BASE = '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/'
ID = '*/'
EXPRESSION = '*/'
#EXPRESSION = 'neutral/'
#EXPRESSION = 'surprised/'
#EXPERIMENT = '*/'
EXPERIMENT = 'multi_iter400_reg30/'
DISTANCE_FILE_NAME = 'distances_v3.log'


# comparing experiments
#plt.title("Error between 3D Scan and Fitting Result on KF-ITW")
#distance_files = [ #['single mean all','*',glob(DB_BASE+ID+EXPRESSION+'single_iter400_reg30/mean.'+DISTANCE_FILE_NAME)],
#				   ['Mean Face','*',glob(DB_BASE+ID+EXPRESSION+'mean_face/'+DISTANCE_FILE_NAME)],
#				   ['Single Image Fitting','*',glob(DB_BASE+ID+EXPRESSION+'single_iter400_reg30/???.'+DISTANCE_FILE_NAME)],
#				   ['Multi Image Fitting','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg30/'+DISTANCE_FILE_NAME)],
#				  ]
# comparing regularisations
#distance_files = [ ['multi all reg5','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg5/'+DISTANCE_FILE_NAME)],
#				   ['multi all reg15','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg15/'+DISTANCE_FILE_NAME)],
#				   ['multi all reg25','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg25/'+DISTANCE_FILE_NAME)],
#				   ['multi all reg30','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg30/'+DISTANCE_FILE_NAME)],
#				   ['multi all reg35','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg35/'+DISTANCE_FILE_NAME)],
#				   ['multi all reg45','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg45/'+DISTANCE_FILE_NAME)],
#				  ]

# comparing number of iterations
#distance_files = [  ['400','+',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg30/'+DISTANCE_FILE_NAME)],
#					['100','+',glob(DB_BASE+ID+EXPRESSION+'multi_iter100_reg30/'+DISTANCE_FILE_NAME)],
#					['50','+',glob(DB_BASE+ID+EXPRESSION+'multi_iter50_reg30/'+DISTANCE_FILE_NAME)],
#					['10','+',glob(DB_BASE+ID+EXPRESSION+'multi_iter10_reg30/'+DISTANCE_FILE_NAME)],
#					['5','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter5_reg30/'+DISTANCE_FILE_NAME)],
#					['4','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter4_reg30/'+DISTANCE_FILE_NAME)],
#					['3','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter3_reg30/'+DISTANCE_FILE_NAME)],
#					['2','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter2_reg30/'+DISTANCE_FILE_NAME)],
#					['1','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter1_reg30/'+DISTANCE_FILE_NAME)],
#					#['mean face','o',glob(DB_BASE+ID+EXPRESSION+'mean_face/'+DISTANCE_FILE_NAME)],
#				 ]


# comparing expressions
#distance_files = [ ['neutral','*',glob(DB_BASE+ID+'neutral/'+EXPERIMENT+DISTANCE_FILE_NAME)],
#				   ['happy','*',glob(DB_BASE+ID+'happy/'+EXPERIMENT+DISTANCE_FILE_NAME)],
#				   ['surprised','*',glob(DB_BASE+ID+'surprised/'+EXPERIMENT+DISTANCE_FILE_NAME)],
#				  ]
# comparing ids
#distance_files = [ ['02','*',glob(DB_BASE+'02/'+EXPRESSION+EXPERIMENT+DISTANCE_FILE_NAME)],
#				   ['08','*',glob(DB_BASE+'08/'+EXPRESSION+EXPERIMENT+DISTANCE_FILE_NAME)],
#				   ['11','*',glob(DB_BASE+'11/'+EXPRESSION+EXPERIMENT+DISTANCE_FILE_NAME)],
#				   ['13','*',glob(DB_BASE+'13/'+EXPRESSION+EXPERIMENT+DISTANCE_FILE_NAME)],
#				   ['16','*',glob(DB_BASE+'16/'+EXPRESSION+EXPERIMENT+DISTANCE_FILE_NAME)],
#		]

#distance_files = [ ['single 02 happy 29k','*',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/02/happy/single_iter400_reg300_m29k/001.distances.log']],
#					['single 02 happy 3500','*',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/02/happy/single_iter400_reg30/001.distances.log']]
#]

#distance_files = [ ['single 08 neutral','*',['/user/HS204/m09113/my_project_folder/KF-ITW-single_test_08_neutral/08/neutral/distances.log']],
#					['single 08 neutral with mask','*',['/user/HS204/m09113/my_project_folder/KF-ITW-single_test_08_neutral_mask/08/neutral/merged.distances.log']]
#]
#distance_files = [ ['multi 02 neutral pymesh imp','*',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/02/neutral/multi_iter400_reg30/distances.log']],
#			       ['multi 02 neutral own','*',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/02/neutral/multi_iter400_reg30/distances_own.log']]
#				 ]

NORMALIZE = True
# read diffs from files

def read_distance_file(file):
	#diffs = []
	with open(file, "r") as f:
		f.readline() # header for distances
		diffs = [float(i) for i in f.readline().split()]
		f.readline() # header for inter eye distance
		inter_eye_distance = float(f.readline())
	return diffs, inter_eye_distance


IDs=['02/', '08/', '11/', '13/', '16/']
Expressions=['neutral/', 'happy/','surprised/']

for ID in IDs:
	for EXPRESSION in Expressions:
		distance_files = glob(DB_BASE+ID+EXPRESSION+'single_iter400_reg30/???.'+DISTANCE_FILE_NAME)
		if len(distance_files)==0:
			continue

		print ("video ",DB_BASE+ID+EXPRESSION+"single_iter400_reg30/ consistes of",len(distance_files),"distance files.")
		all_mean_diffs = []
		for distance_file in distance_files:
			diffs, inter_eye_distance = read_distance_file(distance_file)
			if (NORMALIZE):
				diffs = [d / inter_eye_distance for d in diffs]
			mean_diff = np.mean(diffs)
			all_mean_diffs.append(mean_diff)
		#all_diffs.sort()	

		print ("The smallest mean error is",min(all_mean_diffs), "from distance file", distance_files[all_mean_diffs.index(min(all_mean_diffs))])
		print ("")