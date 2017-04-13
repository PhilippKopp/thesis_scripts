#!/usr/bin/env python3.5

import sys
import numpy as np
from cycler import cycler
from math import sqrt
import sklearn.metrics

#SAVE="/user/HS204/m09113/my_project_folder/Results/alphas_all_imgs"
SAVE="/user/HS204/m09113/my_project_folder/Results/temnp"
SAVE=None

if SAVE:
	import texfig
import matplotlib.pyplot as plt

#if SAVE:
#	fig = texfig.figure(width=8.268, ratio=1/sqrt(2)) #ration of A4 page is sqrt(2)


plots = [ ["PaSC control all alphas without fte", "/user/HS204/m09113/my_project_folder/PaSC/multi_fit_CCR_iter75_reg30_control_without_fte.csv" ],
		  ["PaSC handheld all alphas without fte", "/user/HS204/m09113/my_project_folder/PaSC/multi_fit_CCR_iter75_reg30_handheld_without_fte.csv" ],
		  ["PaSC control 10 alphas without fte", "/user/HS204/m09113/my_project_folder/PaSC/multi_fit_CCR_iter75_reg30_only_10_alphas_control_without_fte.csv" ],
		  ["PaSC handheld 10 alphas without fte", "/user/HS204/m09113/my_project_folder/PaSC/multi_fit_CCR_iter75_reg30_only_10_alphas_handheld_without_fte.csv" ],
		]

def load_matrix( csv_file):
	with open(csv_file, "r") as file:
		same_id = file.readline()
		same_id = [float(i) for i in same_id.split()]
		file.readline()
		different_id = file.readline()
		different_id = [float(i) for i in different_id.split()]
	return same_id, different_id
#exit()




plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'gold', 'm', 'k', 'slategray', 'peru'])))

for plot_idx, plot in enumerate(plots):
	same_id, different_id = load_matrix( plot[1])

	# assemble gt label vector
	gt_labels = [0]*len(same_id)
	gt_labels.extend([1]*len(different_id))
	gt_labels =np.array(gt_labels)
	#print (np.unique(gt_labels))

	# assemble score vector
	scores = []
	scores.extend(same_id)
	scores.extend(different_id)
	scores = np.array(scores)

	fpr, tpr, _ = sklearn.metrics.roc_curve(gt_labels, scores, pos_label=0)

	plt.plot(fpr, tpr, label=plot[0])

plt.plot([x / 1000.0 for x in range(0, 1000, 1)], [x / 1000.0 for x in range(0, 1000, 1)], color='navy', linestyle='--', label='random')
plt.plot([0.001, 0.01, 0.1, 1],[0.05, 0.2, 0.5, 0.96], '*-', label='PaSC Control Surrey')
plt.plot([0.001, 0.01, 0.1, 1],[0.03, 0.13, 0.36, 0.96], '*-', label='PaSC Handheld Surrey')
plt.xscale('log')
plt.xlim([0.001,1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc=2)


plt.show()

if SAVE:
	texfig.savefig(SAVE)