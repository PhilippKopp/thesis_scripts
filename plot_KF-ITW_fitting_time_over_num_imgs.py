#!/usr/bin/env python3.5
import sys, os
import numpy as np
from cycler import cycler
import glob
import obj_analysis_lib as oal

SAVE="/user/HS204/m09113/my_project_folder/Results/alpha_deviations_KF-ITW_mix_of_expressions"
SAVE=None

if SAVE:
	import texfig
import matplotlib.pyplot as plt

if SAVE:
	#fig = texfig.figure(width=8.268) #entire page
	fig = texfig.figure(width=4.8)

plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'gold', 'm', 'k', 'slategray', 'peru'])))

#id_and_expression_dirs = glob.glob('/user/HS204/m09113/my_project_folder/KF-ITW-prerelease_alpha_experiments/multi_model3k_iter10_reg30/*/*')
id_and_expression_dirs = glob.glob('/user/HS204/m09113/my_project_folder/KF-ITW-prerelease_alpha_experiments/multi_model3k_iter10_reg30/*/*_only')
if len(id_and_expression_dirs)==0:
	print ("ERROR: no videos found!!")
	exit(0)

all_std_devs = []

number_of_images_in_sets = []

experiments =[]

for id_and_expression_dir in id_and_expression_dirs:
	print ('id and expression dir: ',id_and_expression_dir)

	# assemble all fitting results we find for this video and load the alphas

	fitting_dirs = glob.glob(id_and_expression_dir+'/*')	
	for fitting_dir in fitting_dirs:
		try:
			num_imgs = int(fitting_dir.split('/')[-1][:3])
		except ValueError:
			continue
		iteration = fitting_dir.split('/')[-1][-2:]
		fitting_log = fitting_dir+'/fitting.log'
		if not os.path.exists(fitting_log):
			print ("ERROR: There is no fitting log file where there should be one!!", fitting_dir)
			exit(0)
		
		# add this number of images if not there yet	
		if len(experiments)==0 or (not True in [num_imgs==i[0] for i in experiments]):
			experiments.append([num_imgs])
			number_of_images_in_sets.append(num_imgs)
	
		# add the fitting logs to the experiments
		for experiment in experiments:
			if experiment[0]==num_imgs:
				experiment.append(fitting_log)
	
	#sort experiments by number of images
	experiments.sort(key=lambda experiment: experiment[0])
	
for experiment in experiments:
	print ('for experiment with', experiment[0], 'images', len(experiment)-1, 'fittings have been found')

	
# now get the fitting times for each number of images
	#experiment_std_devs=[]
times = []
for experiment in experiments:
	experiment_times =[]
	for idx, fitting in enumerate(experiment[1:]):
		#print ("fitting log",fitting,"took",oal.read_fitting_time_from_log(fitting),"seconds")
		#exit(0)
		fitting_time = oal.read_fitting_time_from_log(fitting)
		experiment_times.append(fitting_time)
	times.append(experiment_times)

#print (times)
#exit(0)
#print (len(all_std_devs), ",", len(all_std_devs[0]), ",", all_std_devs[0][0].shape )

x_coordinates = [i[0] for i in experiments]

#print (x_coordinates)
#for idx, experiment_times in enumerate(times):
y_coordinates = []
y_deviation = []
for i in range(len(experiments)):
	y_coordinates.append(np.mean(times[i]))
	y_deviation.append(np.std(times[i]))
#y_coordinates = np.mean(times, axis=0)  # [i[alpha_idx] for i in all_std_devs]
#y_deviation   = np.std(times, axis=0)
	#plt.plot(x_coordinates, y_coordinates, label="alpha "+str(alpha_idx) , marker="s")
plt.errorbar(x_coordinates, y_coordinates, y_deviation, label="10 iterations only fitting", marker="s")
#plt.errorbar(x_coordinates, np.array(y_coordinates)/7.5, np.array(y_deviation)/7.5, label="approx. 10 iterations", marker="s")

plt.xlabel("number of images used for multi image fitting")
plt.ylabel("time for fitting in sec")
#plt.ylim([0,1])
#plt.xlim([0,x_max])
plt.legend(loc=1)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #place to the right
plt.grid(True)

plt.show()

if SAVE:
	texfig.savefig(SAVE)


