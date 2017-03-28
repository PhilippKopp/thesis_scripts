#!/usr/bin/env python3.5

import sys
from glob import glob
SAVE="/user/HS204/m09113/my_project_folder/Results/KF-ITW_comp_expressions_multi"
SAVE=None

if SAVE:
	import texfig
import matplotlib.pyplot as plt

if SAVE:
	fig = texfig.figure(width=8.268) #entire page



# each curve has: label, marker, [log files]
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

# comparing experiments
#distance_files = [ ['single mean all','*',glob(DB_BASE+ID+EXPRESSION+'single_iter400_reg30/mean.distances.log')],
#				   ['single all','*',glob(DB_BASE+ID+EXPRESSION+'single_iter400_reg30/???.distances.log')],
#				   ['multi all','*',glob(DB_BASE+ID+EXPRESSION+'multi_iter400_reg30/distances.log')],
#				  ]
# comparing expressions
#distance_files = [ ['neutral','*',glob(DB_BASE+ID+'neutral/'+EXPERIMENT+'distances.log')],
#				   ['happy','*',glob(DB_BASE+ID+'happy/'+EXPERIMENT+'distances.log')],
#				   ['surprised','*',glob(DB_BASE+ID+'surprised/'+EXPERIMENT+'distances.log')],
#				  ]
# comparing ids
#distance_files = [ ['02','*',glob(DB_BASE+'02/'+EXPRESSION+EXPERIMENT+'distances.log')],
#				   ['08','*',glob(DB_BASE+'08/'+EXPRESSION+EXPERIMENT+'distances.log')],
#				   ['11','*',glob(DB_BASE+'11/'+EXPRESSION+EXPERIMENT+'distances.log')],
#				   ['13','*',glob(DB_BASE+'13/'+EXPRESSION+EXPERIMENT+'distances.log')],
#				   ['16','*',glob(DB_BASE+'16/'+EXPRESSION+EXPERIMENT+'distances.log')],
#		]

distance_files = [ ['single 02 happy 29k','*',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/02/happy/single_iter400_reg300_m29k/001.distances.log']],
					['single 02 happy 3500','*',['/user/HS204/m09113/my_project_folder/KF-ITW-prerelease/02/happy/single_iter400_reg30/001.distances.log']]
]


# read diffs from files

for curve_idx in range(len(distance_files)):
	diffs = []
	for distance_file in distance_files[curve_idx][2]:
		with open(distance_file, "r") as diff_file:
		
			for line in diff_file:
				diffs.extend([float(i) for i in line.split()])
	diffs.sort()
	#diffs = diffs[0:int(len(diffs)*2/3)]
	#print len(diffs)
	# assemble x and y coordinates
	delta = 0.005
	x_max = 0.06
	x_coordinates = [i*delta for i in range(int(x_max/delta)+1)]
	y_coordinates = []
	for x in x_coordinates:
		for j, diff in enumerate(diffs):	
			if (diff>x):
				y_coordinates.append(j/len(diffs))
				#print str(diff)+"  "+str(x)+"  "+str(j)
				break
	# make sure y is as long as x, if not fill with 1
	if (len(y_coordinates)<len(x_coordinates)):
		y_coordinates.extend([1]*(len(x_coordinates)-len(y_coordinates)))
	#print (len(x_coordinates))
	#print (len(y_coordinates))
	plt.plot(x_coordinates, y_coordinates, label=distance_files[curve_idx][0], marker=distance_files[curve_idx][1])

#print (x_coordinates)
plt.plot([0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06], [0.0, 0.13, 0.26, 0.38, 0.49, 0.58, 0.67, 0.76, 0.82, 0.88, 0.91, 0.93, 0.94], label="imperial results")
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



