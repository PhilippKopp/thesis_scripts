#!/usr/bin/env python3.5

import sys
from glob import glob
import numpy as np
SAVE="/user/HS204/m09113/my_project_folder/Results/KF-ITW_take_mean_of_vertex_error"
SAVE=None

if SAVE:
	import texfig
import matplotlib.pyplot as plt

if SAVE:
	fig = texfig.figure(width=8.268) #entire page



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




DB_BASE = '/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/'
ID = '*/'
EXPRESSION = '*/'
#EXPRESSION = 'neutral/'
#EXPRESSION = 'surprised/'
#EXPERIMENT = '*/'
EXPERIMENT = 'multi_iter400_reg30/'
DISTANCE_FILE_NAME = 'distances_v3.log'


# comparing experiments
distance_files = [ ['single mean all','*',glob(DB_BASE+ID+EXPRESSION+'single_iter400_reg30/mean.'+DISTANCE_FILE_NAME)],
				   ['single all','*',glob(DB_BASE+ID+EXPRESSION+'single_iter400_reg30/???.'+DISTANCE_FILE_NAME)],
				   ['multi all','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg30/'+DISTANCE_FILE_NAME)],
				   ['mean face','*',glob(DB_BASE+ID+EXPRESSION+'mean_face/'+DISTANCE_FILE_NAME)],
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
distance_files = [  ['multi 400 iterations','+',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg30/'+DISTANCE_FILE_NAME)],
					['multi 100 iterations','+',glob(DB_BASE+ID+EXPRESSION+'multi_iter100_reg30/'+DISTANCE_FILE_NAME)],
					['multi 50  iterations','+',glob(DB_BASE+ID+EXPRESSION+'multi_iter50_reg30/'+DISTANCE_FILE_NAME)],
					['multi 10  iterations','+',glob(DB_BASE+ID+EXPRESSION+'multi_iter10_reg30/'+DISTANCE_FILE_NAME)],
					['multi 5   iterations','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter5_reg30/'+DISTANCE_FILE_NAME)],
					['multi 4   iterations','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter4_reg30/'+DISTANCE_FILE_NAME)],
					['multi 3   iterations','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter3_reg30/'+DISTANCE_FILE_NAME)],
					['multi 2   iterations','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter2_reg30/'+DISTANCE_FILE_NAME)],
					['multi 1   iterations','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter1_reg30/'+DISTANCE_FILE_NAME)],
					['mean face','o',glob(DB_BASE+ID+EXPRESSION+'mean_face/'+DISTANCE_FILE_NAME)],
				 ]


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
plt.plot([0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06], [0.0, 0.13, 0.26, 0.38, 0.49, 0.58, 0.67, 0.76, 0.82, 0.88, 0.91, 0.93, 0.94], label="imperial results eos without bs (entire db)")
plt.plot([0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06], [0.0, 0.14, 0.30, 0.45, 0.59, 0.69, 0.79, 0.86, 0.91, 0.93, 0.95, 0.97, 0.98], label="imperial results their ITW (entire db)")
# save or show figure

plt.xlabel("vertex error")
plt.ylabel("Vertexes proportion")
plt.ylim([0,1])
plt.xlim([0,x_max])
plt.legend(loc=4)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #place to the right
plt.grid(True)
plt.show()

if SAVE:
	texfig.savefig(SAVE)



