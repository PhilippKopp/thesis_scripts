#!/usr/bin/env python3.5
import sys, os
import numpy as np
import glob
import obj_analysis_lib as oal

SAVE="/user/HS204/m09113/my_project_folder/Results/KF-ITW_take_mean_of_vertex_error"
SAVE=None

if SAVE:
	import texfig
import matplotlib.pyplot as plt

if SAVE:
	fig = texfig.figure(width=8.268) #entire page


fitting_dirs = glob.glob('/user/HS204/m09113/my_project_folder/KF-ITW-prerelease_alpha_experiments/*/*_only/*')
experiments =[]
for fitting_dir in fitting_dirs:
	num_imgs = int(fitting_dir.split('/')[-1][:3])
	iteration = fitting_dir.split('/')[-1][-2:]
	fitting_log = fitting_dir+'/fitting.log'
	if not os.path.exists(fitting_log):
		print ("ERROR: There is no fitting log file where there should be one!!", fitting_dir)
		exit(0)
	
	# add this number of images if not there yet	
	if len(experiments)==0 or (not True in [num_imgs==i[0] for i in experiments]):
		experiments.append([num_imgs])

	# add the fitting logs to the experiments
	for experiment in experiments:
		if experiment[0]==num_imgs:
			experiment.append(fitting_log)

#sort experiments by number of images
experiments.sort(key=lambda experiment: experiment[0])

for experiment in experiments:
	print ('for experiment with', experiment[0], 'images', len(experiment)-1, 'fittings have been found')
#print (experiments[0])


all_std_devs=[]
for experiment in experiments:
	alphas = np.empty(((len(experiment)-1),63))
	for idx, fitting in enumerate(experiment[1:]):
		alphas[idx,:] = oal.read_fitting_log(fitting)
	std_devs = np.std(alphas, axis=0)
	all_std_devs.append(std_devs)

x_coordinates = [i[0] for i in experiments]
for alpha_idx in range(10):
	y_coordinates = [i[alpha_idx] for i in all_std_devs]
	plt.plot(x_coordinates, y_coordinates, label="alpha "+str(alpha_idx) , marker="s")

plt.xlabel("number of images used for multi image fitting")
plt.ylabel("std deviation of alphas")
#plt.ylim([0,1])
#plt.xlim([0,x_max])
plt.legend(loc=1)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #place to the right
plt.grid(True)

plt.show()

if SAVE:
	texfig.savefig(SAVE)


