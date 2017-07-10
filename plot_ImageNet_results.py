#!/usr/bin/env python3.5
import sys, os
import numpy as np
from cycler import cycler
import glob
import obj_analysis_lib as oal

OUTPUTPATH="/user/HS204/m09113/my_project_folder/Results/imageNet_performance"
SAVE4PRES =None
SAVE4LATEX=None

SAVE4LATEX=OUTPUTPATH
#SAVE4PRES =OUTPUTPATH


if SAVE4LATEX:
	import texfig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if SAVE4LATEX:
	fig = texfig.figure(width=5.8) #entire page

if SAVE4PRES:
	#plt.rcParams["font.family"] ="monospace"
	plt.figure(figsize=(10, 5))

plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'gold', 'm', 'k', 'slategray', 'peru'])))

years = ["2010", "2011", "2012", "2013", "2014", "Human", "2015", "2016"]

bar_width = 0.6
x_coordinates = np.arange(len(years))
y_coordinates = [28, 25.7, 16.4, 11.7, 7.4, 5.0, 3.6, 2.9]
#plt.errorbar(x_coordinates, y_coordinates, y_deviation, label="L2 error", marker="s",linestyle='None')
barlist = plt.bar(x_coordinates,y_coordinates,width=bar_width)

#barlist[0].set_color((242/255, 242/255, 242/255))
barlist[0].set_facecolor((242/255, 242/255, 242/255))
barlist[1].set_facecolor((242/255, 242/255, 242/255))
barlist[2].set_facecolor((000/255, 70/255, 160/255))
barlist[3].set_facecolor((000/255, 70/255, 160/255))
barlist[4].set_facecolor((000/255, 70/255, 160/255))
barlist[5].set_facecolor((201/255, 169/255, 000/255))
barlist[6].set_facecolor((000/255, 70/255, 160/255))
barlist[7].set_facecolor((000/255, 70/255, 160/255))





plt.xticks(x_coordinates+bar_width/2,years)
plt.ylabel("Top-5 error rate (%)")
#plt.ylim([0,1])
x_coordinates = x_coordinates+bar_width/2
plt.xlim([np.min(x_coordinates)-x_coordinates.shape[0]/(len(years)),np.max(x_coordinates)+x_coordinates.shape[0]/(len(years))])
#plt.legend(loc=9, ncol=NUMBER_OF_ALPHAS_TO_PLOT, numpoints=1, handletextpad=0, columnspacing=0.5)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #place to the right
#plt.grid(True)

traditional_patch = mpatches.Patch(color=(242/255, 242/255, 242/255), label='Traditional')
deep_learning_patch = mpatches.Patch(color=(000/255, 70/255, 160/255), label='Deep Learning')
human_patch = mpatches.Patch(color=(201/255, 169/255, 000/255), label='Human')
plt.legend(loc=1, handles=[traditional_patch, deep_learning_patch, human_patch])
#plt.title("Top-5 Error at ImageNet Classification Challenge")


plt.gca().yaxis.grid(True)
plt.gca().set_axisbelow(True)

if SAVE4PRES:
	plt.savefig(SAVE4PRES+".png")

if SAVE4LATEX:
	texfig.savefig(SAVE4LATEX)

plt.show()

