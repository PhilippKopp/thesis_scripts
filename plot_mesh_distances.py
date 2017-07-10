#!/usr/bin/env python3.5

import sys
from glob import glob
import numpy as np
from cycler import cycler

OUTPUTPATH="/user/HS204/m09113/my_project_folder/Results/mesh_distances_KF-ITW_patrik_thesis"
#OUTPUTPATH="/user/HS204/m09113/my_project_folder/Results/mesh_distances_KF-ITW_iterations_thesis"
#OUTPUTPATH='mesh_distances_KF-ITW_fitting_types_BMVC'
SAVE4PRES =None
SAVE4LATEX=None

SAVE4LATEX=OUTPUTPATH
#SAVE4PRES =OUTPUTPATH


if SAVE4LATEX:
	import texfig
import matplotlib.pyplot as plt

if SAVE4LATEX:
	#fig = texfig.figure(width=8.268) #entire page
	fig = texfig.figure(width=4.8) #philipp thesis 5.8

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

#plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'slategray', 'gold', 'peru'])))
plt.rc('axes', prop_cycle=(cycler('color', ['b', 'c', 'y', 'm', 'c', 'm', 'k', 'slategray', 'gold', 'peru'])))
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
distance_files = [ #['single mean all','*',glob(DB_BASE+ID+EXPRESSION+'single_iter400_reg30/mean.'+DISTANCE_FILE_NAME)],
				   #['Mean Face','*',glob(DB_BASE+ID+EXPRESSION+'mean_face/'+DISTANCE_FILE_NAME)],
				   #['Single Image Fitting','*',glob(DB_BASE+ID+EXPRESSION+'single_iter400_reg30/???.'+DISTANCE_FILE_NAME)],
				   ['Multi Image Fitting','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg30/'+DISTANCE_FILE_NAME)],
#				   ['Best single Fittings','*',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/02/neutral/single_iter400_reg30/058.distances_v3.log', '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/02/happy/single_iter400_reg30/127.distances_v3.log', '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/02/surprised/single_iter400_reg30/010.distances_v3.log', '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/08/neutral/single_iter400_reg30/053.distances_v3.log', '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/08/surprised/single_iter400_reg30/001.distances_v3.log', '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/11/neutral/single_iter400_reg30/076.distances_v3.log', '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/11/happy/single_iter400_reg30/091.distances_v3.log', '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/11/surprised/single_iter400_reg30/068.distances_v3.log', '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/13/neutral/single_iter400_reg30/025.distances_v3.log', '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/13/happy/single_iter400_reg30/026.distances_v3.log', '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/13/surprised/single_iter400_reg30/070.distances_v3.log', '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/16/neutral/single_iter400_reg30/163.distances_v3.log', '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/16/happy/single_iter400_reg30/050.distances_v3.log']],
				  ]
# comparing regularisations
#distance_files = [ ['multi all reg5','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg5/'+DISTANCE_FILE_NAME)],
#				   ['multi all reg15','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg15/'+DISTANCE_FILE_NAME)],
#				   ['multi all reg25','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg25/'+DISTANCE_FILE_NAME)],
#				   ['multi all reg30','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg30/'+DISTANCE_FILE_NAME)],
#				   ['multi all reg35','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg35/'+DISTANCE_FILE_NAME)],
#				   ['multi all reg45','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg45/'+DISTANCE_FILE_NAME)],
#				  ]

# comparing number of iterations
#distance_files = [  #['400','+',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg30/'+DISTANCE_FILE_NAME)],
					#['100','+',glob(DB_BASE+ID+EXPRESSION+'multi_iter100_reg30/'+DISTANCE_FILE_NAME)],
					#['50','+',glob(DB_BASE+ID+EXPRESSION+'multi_iter50_reg30/'+DISTANCE_FILE_NAME)],
					#['10 iter.','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter10_reg30/'+DISTANCE_FILE_NAME)],
					#['5 iter.','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter5_reg30/'+DISTANCE_FILE_NAME)],
					#['4 iterations','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter4_reg30/'+DISTANCE_FILE_NAME)],
					#['3 iter.','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter3_reg30/'+DISTANCE_FILE_NAME)],
					#['2','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter2_reg30/'+DISTANCE_FILE_NAME)],
					#['1 iter.','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter1_reg30/'+DISTANCE_FILE_NAME)],
					#['mean face','*',glob(DB_BASE+ID+EXPRESSION+'mean_face/'+DISTANCE_FILE_NAME)],
				 #]


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

# AFLW2000 experiment
#distance_files = [  #['3DDFA with Surrey registration','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/3DDFA/_eval_with_surrey_mean/*distances_v3.log')],
					#['3DDFA','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/3DDFA/_eval_without_surrey/*.distances_v3.log')],
					#['eos iter200 reg30 with fitting registration','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter200_reg30/_eval_with_own_fitting_registration/*.distances_v3.log')],
					#['eos iter50 reg30 with fitting registration','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter50_reg30/_eval_with_own_fitting_registration/*.distances_v3.log')],
					#['eos','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter10_reg30/_eval_with_own_fitting_registration/*.distances_v3.log')],
					#['eos iter5 reg30 with fitting registration','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter5_reg30/_eval_with_own_fitting_registration/*.distances_v3.log')],
					#['eos iter2 reg30 with fitting registration','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter2_reg30/_eval_with_own_fitting_registration/*.distances_v3.log')],
					#['eos iter1 reg30 with fitting registration','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter1_reg30/_eval_with_own_fitting_registration/*.distances_v3.log')],
					#['eos iter10 reg30 with surrey mean registration','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter10_reg30/_eval_with_surrey_mean_reg/*.distances_v3.log')],
					#['eos iter5 reg30 845model with surrey mean registration','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter5_reg30_845model/_eval_with_surrey_mean_registration/*.distances_v3.log')],
					#['eos iter5 reg30 845model with surrey fitting registration','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter5_reg30_845model/_eval_with_surrey_fitting_registration/*.distances_v3.log')],
					#['eos iter5 reg30 845model with own fitting registration','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter5_reg30_845model/_eval_with_own_fitting_registration/*.distances_v3.log')],
					#['eos iter5  reg30 with surrey mean registration','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter5_reg30/_eval_with_surrey_mean_registration/*.distances_v3.log')],
					#['eos iter5  reg30 front contour with surrey mean registration','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter5_reg30_front_contour/_eval_with_surrey_mean_registration/*distances_v3.log')],
					#['eos iter5  reg30 without contour with surrey mean registration','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter5_reg30_without_contour/_eval_with_surrey_mean_registration/*distances_v3.log')],
					#['eos iter10 reg30 static contour with surrey mean registration','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter10_reg30_static_contour/_eval_with_surrey_mean_registration/*distances_v3.log')],
					#['eos iter10 reg30 on basel directly','*',glob('/user/HS204/m09113/my_project_folder/AFLW2000_fittings/eos_single_iter10_reg30/_eval_directly_basel_gt/*distances_v3.log')],
				#]

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



for curve_idx in range(len(distance_files)):
	print ("curve",distance_files[curve_idx][0],"consistes of",len(distance_files[curve_idx][2]),"distance files.")
	all_diffs = []
	for distance_file in distance_files[curve_idx][2]:
		diffs, inter_eye_distance = read_distance_file(distance_file)
		if (NORMALIZE):
			diffs = [d / inter_eye_distance for d in diffs]
		all_diffs.extend(diffs)
	all_diffs.sort()
	
	# assemble x and y coordinates
	delta = 0.005
	x_max = 0.06
	x_coordinates = [i*delta for i in range(int(x_max/delta)+1)]
	y_coordinates = []
	for x in x_coordinates:
		for j, diff in enumerate(all_diffs):	
			if (diff>x):
				y_coordinates.append(j/len(all_diffs))
				#print str(diff)+"  "+str(x)+"  "+str(j)
				break
	# make sure y is as long as x, if not fill with 1
	if (len(y_coordinates)<len(x_coordinates)):
		y_coordinates.extend([1]*(len(x_coordinates)-len(y_coordinates)))
	#print (len(x_coordinates))
	#print (len(y_coordinates))
	plt.plot(x_coordinates, y_coordinates, label=distance_files[curve_idx][0], marker=distance_files[curve_idx][1])

#print (x_coordinates)
plt.plot([0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06], [0.0, 0.13, 0.26, 0.38, 0.49, 0.585, 0.675, 0.76, 0.82, 0.87, 0.905, 0.93, 0.95], label="Earlier proposed*", marker='*')
plt.plot([0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06], [0.0, 0.14, 0.30, 0.45, 0.59, 0.69, 0.79, 0.86, 0.91, 0.93, 0.95, 0.97, 0.98], label="ITW", marker='*')
plt.plot([0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06], [0.0, 0.11, 0.22, 0.32, 0.42, 0.51, 0.59, 0.65, 0.71, 0.76, 0.80, 0.83, 0.86], label="Classic*", marker='*')
# save or show figure

plt.xlabel("Normalised vertex error")
plt.ylabel("Vertices proportion")
plt.ylim([0,1])
plt.xlim([0,x_max])
#plt.xlim([0.0275, 0.031])
#plt.ylim([0.72,0.795])
#plt.legend(loc=2, ncol=2)
plt.legend(loc=4)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #place to the right
plt.grid(True)




if SAVE4PRES:
	plt.savefig(SAVE4PRES+".png")

if SAVE4LATEX:
	texfig.savefig(SAVE4LATEX)

plt.show()
