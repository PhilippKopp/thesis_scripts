#!/usr/bin/env python3.5
import sys, os
import numpy as np
from cycler import cycler
import glob
import obj_analysis_lib as oal

SAVE="/user/HS204/m09113/my_project_folder/Results/alpha_deviations_KF-ITW_over_pose_prepared"
#SAVE=None

if SAVE:
	import texfig
import matplotlib.pyplot as plt

if SAVE:
	fig = texfig.figure(width=8.268) #entire page
	fig = texfig.figure(width=4.8) #entire page

plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'gold', 'm', 'k', 'slategray', 'peru'])))
NUMBER_OF_ALPHAS_TO_PLOT = 10



#id_and_expression_dirs = glob.glob('/user/HS204/m09113/my_project_folder/KF-ITW-prerelease_alpha_experiments/*/expression_mix')
id_and_expression_dirs = glob.glob('/user/HS204/m09113/my_project_folder/KF-ITW-prerelease_alpha_experiments/*/*_only')
if len(id_and_expression_dirs)==0:
	print ("ERROR: no videos found!!")
	exit(0)

#categories = ['yaw<20', 'yaw>-20', '-30<yaw<30', '-40<yaw<40', 'yaw<-10 or yaw>10', 'all', '<-20 and +-20 and >20']
categories = ['exp_'+format(i, '01d') for i in range(10)]

all_std_devs = []

for id_and_expression_dir in id_and_expression_dirs:
	print ('id and expression dir: ',id_and_expression_dir)

	# assemble all fitting results we find for this video and load the alphas
	experiment_alphas =[ [] for i in range(len(categories)) ]

	#fitting_dirs = glob.glob(id_and_expression_dir+'/00[3,4,5,7]*')	
	fitting_dirs = glob.glob(id_and_expression_dir+'/pose_exp_*')	
	for fitting_dir in fitting_dirs:
		pose_exp = int(fitting_dir.split('/')[-1][-5:-3])
		iteration = int(fitting_dir.split('/')[-1][-2:])
		fitting_log = fitting_dir+'/fitting.log'
		if not os.path.exists(fitting_log):
			print ("ERROR: There is no fitting log file where there should be one!!", fitting_dir)
			exit(0)
	
		alphas, angles = oal.read_fitting_log(fitting_log)
		angles = np.array(angles)

		# add the alphas to the specific experiment_alphas
		experiment_alphas[pose_exp].append(alphas)

	# print some information
	for i in range(len(categories)):
		print ('for experiment with', categories[i], len(experiment_alphas[i]), 'fittings have been found that match the criteria')
	
	# now calculate the std deviation 
	experiment_std_devs=[]
	for i in range(len(experiment_alphas)):
		alphas = np.array(experiment_alphas[i])
		std_devs = np.std(alphas, axis=0)
		experiment_std_devs.append(std_devs)
	all_std_devs.append(experiment_std_devs)

all_std_devs = np.array(all_std_devs)


x_coordinates = np.arange(len(categories))

for alpha_idx in range(NUMBER_OF_ALPHAS_TO_PLOT):
	y_coordinates = np.mean(all_std_devs[:,:,alpha_idx], axis=0)  # [i[alpha_idx] for i in all_std_devs]
	y_deviation   = np.std(all_std_devs[:,:,alpha_idx], axis=0)
	#plt.plot(x_coordinates, y_coordinates, label="alpha "+str(alpha_idx) , marker="s")
	plt.errorbar(x_coordinates, y_coordinates, y_deviation, label="alpha "+str(alpha_idx) , marker="s",linestyle='None')

plt.xticks(x_coordinates,categories)
plt.ylabel("std deviation of alphas")
#plt.ylim([0,1])
plt.xlim([np.min(x_coordinates)-x_coordinates.shape[0]/(2*len(categories)),np.max(x_coordinates)+x_coordinates.shape[0]/(2*len(categories))])
plt.legend(loc=9, ncol=NUMBER_OF_ALPHAS_TO_PLOT, numpoints=1, handletextpad=0, columnspacing=0.5)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #place to the right
plt.grid(True)

plt.show()

if SAVE:
	texfig.savefig(SAVE)


