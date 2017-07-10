#!/usr/bin/env python3.5
import sys, os
import numpy as np
from cycler import cycler
import obj_analysis_lib as oal

OUTPUTPATH="/user/HS204/m09113/my_project_folder/Results/imageNet_performance"
SAVE4PRES =None
SAVE4LATEX=None

#SAVE4LATEX=OUTPUTPATH
SAVE4PRES =OUTPUTPATH



if SAVE4LATEX:
	import texfig
import matplotlib.pyplot as plt

if SAVE4LATEX:
	fig = texfig.figure(width=8.268) #entire page

if SAVE4PRES:
	#plt.rcParams["font.family"] ="monospace"
	plt.figure(figsize=(10, 8/3*2))

plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'gold', 'm', 'k', 'slategray', 'peru'])))

x_coordinates = np.arange(len(categories))

y_coordinates = np.mean(all_errors[:,:], axis=0)  # [i[alpha_idx] for i in all_errors]
y_deviation   = np.std(all_errors[:,:], axis=0)
plt.errorbar(x_coordinates, y_coordinates, y_deviation, label="L2 error", marker="s",linestyle='None')

plt.xticks(x_coordinates,categories)
plt.ylabel("L2 error between mean alphas and \"gt\" alphas calculated over entire video")
#plt.ylim([0,1])
plt.xlim([np.min(x_coordinates)-x_coordinates.shape[0]/(2*len(categories)),np.max(x_coordinates)+x_coordinates.shape[0]/(2*len(categories))])
#plt.legend(loc=9, ncol=NUMBER_OF_ALPHAS_TO_PLOT, numpoints=1, handletextpad=0, columnspacing=0.5)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #place to the right
plt.legend(loc=1, handles=[traditional_patch, deep_learning_patch, human_patch])
plt.title("Top-5 Error at ImageNet Classification Challenge")

plt.grid(True)

if SAVE4PRES:
	plt.savefig(SAVE4PRES+".png")

if SAVE4LATEX:
	texfig.savefig(SAVE4LATEX)

plt.show()
