#!/usr/bin/env python3.5
import sys, os
import numpy as np
from cycler import cycler
import obj_analysis_lib as oal

OUTPUTPATH="/user/HS204/m09113/my_project_folder/Results/alex_cnn_eval"
SAVE4PRES =None
SAVE4LATEX=None

#SAVE4LATEX=OUTPUTPATH
#SAVE4PRES =OUTPUTPATH


if SAVE4LATEX:
	import texfig
import matplotlib.pyplot as plt

if SAVE4LATEX:
	fig = texfig.figure(width=8.268) #entire page

if SAVE4PRES:
	#plt.rcParams["font.family"] ="monospace"
	plt.figure(figsize=(10, 8/3*2))

FR_EXPERIMENT = False

if FR_EXPERIMENT:
	experiment_paths = [ #['01 ', '/user/HS204/m09113/my_project_folder/cnn_experiments/01/'],
						 ['02 alex', '/user/HS204/m09113/my_project_folder/cnn_experiments/02/'],
						 ['03 dcnn', '/user/HS204/m09113/my_project_folder/cnn_experiments/03/'],
						 ['04 alex on xyz', '/user/HS204/m09113/my_project_folder/cnn_experiments/04/'],
						 ['05 alex 256', '/user/HS204/m09113/my_project_folder/cnn_experiments/05/'],
						 ['09 alex facebox', '/user/HS204/m09113/my_project_folder/cnn_experiments/09/'],
						 ['11 alex with alpha', '/user/HS204/m09113/my_project_folder/cnn_experiments/11/'],
						 ['12 alex with rgb+xyz (on trainingset)', '/user/HS204/m09113/my_project_folder/cnn_experiments/12/'],
						 ['13 dcnn with rgb+xyz', '/user/HS204/m09113/my_project_folder/cnn_experiments/13/'],
						 ['20 alex only good isomaps', '/user/HS204/m09113/my_project_folder/cnn_experiments/20/'],
						 ['21 alex with merged isomaps', '/user/HS204/m09113/my_project_folder/cnn_experiments/21/'],
						 ['40 alex 3 merging', '/user/HS204/m09113/my_project_folder/cnn_experiments/40/'],
						 ['41 alex 3 merging', '/user/HS204/m09113/my_project_folder/cnn_experiments/41/'],
						 ['45 alex 3 merging', '/user/HS204/m09113/my_project_folder/cnn_experiments/45/'],
						]
else:
	experiment_paths = [ ['01 loss', '/user/HS204/m09113/my_project_folder/IJB_A/nn_score_chooser/01/training_loss.csv'],
						 ['01 accuracy', '/user/HS204/m09113/my_project_folder/IJB_A/nn_score_chooser/01/training_acc.csv'],
						 ['02 loss', '/user/HS204/m09113/my_project_folder/IJB_A/nn_score_chooser/02/training_loss.csv'],
						 ['02 accuracy', '/user/HS204/m09113/my_project_folder/IJB_A/nn_score_chooser/02/training_acc.csv'],
						 ['03 loss', '/user/HS204/m09113/my_project_folder/IJB_A/nn_score_chooser/03/training_loss.csv'],
						 ['03 accuracy', '/user/HS204/m09113/my_project_folder/IJB_A/nn_score_chooser/03/training_acc.csv'],
						 ['04 loss', '/user/HS204/m09113/my_project_folder/IJB_A/nn_score_chooser/04/training_loss.csv'],
						 ['04 accuracy', '/user/HS204/m09113/my_project_folder/IJB_A/nn_score_chooser/04/training_acc.csv'],
						]




for i in range(len(experiment_paths)):
	experiment_path = experiment_paths[i][1]
	if FR_EXPERIMENT:
		eval_log = experiment_path+'eval/eval.log'
	else:
		eval_log = experiment_path

	x_coordinates = []
	y_coordinates = []
	with open(eval_log, 'r') as log:
		for line in log:
			iteration = int(line.split()[0])
			performance = float(line.split()[1])
			x_coordinates.append(iteration)
			y_coordinates.append(performance)


	y_coordinates = [y for (x,y) in sorted(zip(x_coordinates,y_coordinates), key=lambda pair: pair[0])]
	x_coordinates.sort()
	
	#plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'gold', 'm', 'k', 'slategray', 'peru'])))
	
	#x_coordinates = np.arange(len(categories))
	
	#y_coordinates = np.mean(all_errors[:,:], axis=0)  # [i[alpha_idx] for i in all_errors]
	#y_deviation   = np.std(all_errors[:,:], axis=0)
	#plt.errorbar(x_coordinates, y_coordinates, y_deviation, label="L2 error", marker="s",linestyle='None')
	plt.plot(x_coordinates, y_coordinates, marker="*", label=experiment_paths[i][0])
	
	#plt.xticks(x_coordinates,categories)
plt.ylabel("top 1 accuracy")
plt.xlabel("training iterations")
#plt.ylim([0,1])
#plt.xlim([np.min(x_coordinates)-x_coordinates.shape[0]/(2*len(categories)),np.max(x_coordinates)+x_coordinates.shape[0]/(2*len(categories))])
#plt.legend(loc=9, ncol=NUMBER_OF_ALPHAS_TO_PLOT, numpoints=1, handletextpad=0, columnspacing=0.5)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #place to the right
#plt.legend(loc=1, handles=[traditional_patch, deep_learning_patch, human_patch])
plt.legend(loc=1)
#plt.title("Top-5 Error at ImageNet Classification Challenge")

plt.grid(True)

if SAVE4PRES:
	plt.savefig(SAVE4PRES+".png")

if SAVE4LATEX:
	texfig.savefig(SAVE4LATEX)

plt.show()
