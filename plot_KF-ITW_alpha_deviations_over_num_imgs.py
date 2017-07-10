#!/usr/bin/env python3.5
import sys, os
import numpy as np
from cycler import cycler
import glob
import obj_analysis_lib as oal

#SAVE="/user/HS204/m09113/my_project_folder/Results/alpha_deviations_KF-ITW_mix_of_expressions"
SAVE="/user/HS204/m09113/my_project_folder/Results/alpha_deviations_KF-ITW_all_videos_thesis"
#SAVE=None

if SAVE:
	import texfig
import matplotlib.pyplot as plt

if SAVE:
	#fig = texfig.figure(width=8.268) #entire page
	fig = texfig.figure(width=5.8)

plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'gold', 'm', 'k', 'slategray', 'peru'])))

#id_and_expression_dirs = glob.glob('/user/HS204/m09113/my_project_folder/KF-ITW-prerelease_alpha_experiments/*/expression_mix')
#id_and_expression_dirs = glob.glob('/user/HS204/m09113/my_project_folder/KF-ITW-prerelease_alpha_experiments/02/happy_only')
id_and_expression_dirs = glob.glob('/user/HS204/m09113/my_project_folder/KF-ITW-prerelease_alpha_experiments/multi_iter75_reg30/*/*_only')
if len(id_and_expression_dirs)==0:
	print ("ERROR: no videos found!!")
	exit(0)

all_std_devs = []

number_of_images_in_sets = []

for id_and_expression_dir in id_and_expression_dirs:
	print ('id and expression dir: ',id_and_expression_dir)

	# assemble all fitting results we find for this video and load the alphas
	experiments =[]
	fitting_dirs = glob.glob(id_and_expression_dir+'/*images*')	
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
			number_of_images_in_sets.append(num_imgs)
	
		# add the fitting logs to the experiments
		for experiment in experiments:
			if experiment[0]==num_imgs:
				experiment.append(fitting_log)
	
	#sort experiments by number of images
	experiments.sort(key=lambda experiment: experiment[0])
	
	for experiment in experiments:
		print ('for experiment with', experiment[0], 'images', len(experiment)-1, 'fittings have been found')
	
	# now calculate the std deviation 
	experiment_std_devs=[]
	for experiment in experiments:
		alphas = np.empty(((len(experiment)-1),63))
		for idx, fitting in enumerate(experiment[1:]):
			alphas[idx,:], _ = oal.read_fitting_log(fitting)
		std_devs = np.std(alphas, axis=0)
		experiment_std_devs.append(std_devs)
	all_std_devs.append(experiment_std_devs)
	#break

#print (len(all_std_devs), ",", len(all_std_devs[0]), ",", all_std_devs[0][0].shape )
#x_coordinates = [i[0] for i in experiments]

# make sure number of image sets is equal for all videos
min_num_sets = min([len(i) for i in all_std_devs]) #if len(all_std_devs)>1 else len(all_std_devs[0])
#print([len(i) for i in all_std_devs])
#print (min_num_sets)
for video_idx in range(len(all_std_devs)):
	all_std_devs[video_idx] = all_std_devs[video_idx][:min_num_sets]

all_std_devs = np.array(all_std_devs)
#print ('all std shape',all_std_devs.shape)

# assemble x coordinates with correct number of image sets
x_coordinates = sorted(list(set(number_of_images_in_sets)))[:min_num_sets]

#print (x_coordinates)
for alpha_idx in range(5):
	y_coordinates = np.mean(all_std_devs[:,:,alpha_idx], axis=0)  # [i[alpha_idx] for i in all_std_devs]
	y_deviation   = np.std(all_std_devs[:,:,alpha_idx], axis=0)
	#plt.plot(x_coordinates, y_coordinates, label="alpha "+str(alpha_idx) , marker="s")
	plt.errorbar(x_coordinates, y_coordinates, y_deviation, label="alpha "+str(alpha_idx) , marker="s")

plt.xlabel("Number of images")
plt.ylabel("Standard deviation of alphas")
#plt.ylim([0,1])
#plt.xlim([0,x_max])
plt.legend(loc=1)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #place to the right
plt.grid(True)

plt.show()

if SAVE:
	texfig.savefig(SAVE)


