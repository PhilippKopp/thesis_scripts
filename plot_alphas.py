#!/usr/bin/env python3.5


import sys
import numpy as np
from cycler import cycler
from math import sqrt

#SAVE="/user/HS204/m09113/my_project_folder/Results/alphas_all_imgs"
SAVE="/user/HS204/m09113/my_project_folder/Results/alphas_convergence_BMVC"
#SAVE=None

if SAVE:
	import texfig
import matplotlib.pyplot as plt

#if SAVE:
#	fig = texfig.figure(width=8.268, ratio=1/sqrt(2)) #ration of A4 page is sqrt(2)




### single images
#plots = [ [ "only img 3", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_happy_conv_oneImg03.csv"],
#		  [ "only img 19", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_happy_conv_oneImg19.csv"],
#		  [ "only img 30", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_happy_conv_oneImg30.csv"],
#		  [ "only img 142", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_happy_conv_oneImg142.csv"],
#		  [ "only img 158", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_happy_conv_oneImg158.csv"]
#		]
#plots = [ [ "4 imgs (without 19)", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_happy_conv_4Imgs_19.csv"],
#		  [ "4 imgs (without 30)", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_happy_conv_4Imgs_30.csv"],
#		  [ "5 imgs", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_happy_conv_5Imgs.csv"]
#		 ]

plots = [ #[" half of all images", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_happy_conv_halfImgs.csv" ],
#		  [" all images without some bad fittings (manualy excluded)", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_happy_without_bad_conv.csv" ],
#		  [" all images without 30", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_happy_conv_without_30.csv" ],
		  [" all images", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_happy_conv.csv" ]
		]

#plots = [ ["02 happy", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_happy_conv.csv" ],
#		  ["02 neutral", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_neutral_conv.csv" ],
#		  ["02 surprised", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_02_surprised_conv.csv" ],
#		  ["08 neutral", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_08_neutral_conv.csv" ],
#		  ["13 neutral", "/user/HS204/m09113/my_project_folder/multi_fitting_convergence_tests/KF-ITW_multi_13_neutral_conv.csv" ]
#		]






def load_alphas( file):
	data = np.loadtxt(file, delimiter=", ")
	alphas = data[1::3]
	return alphas


plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'gold', 'm', 'k', 'slategray', 'peru'])))
#if SAVE:
	#f, axarr = texfig.subplots_philipp(len(plots),width=8.268, ratio=sqrt(2), sharex=True, squeeze=False)
#else:
	#f, axarr = plt.subplots(len(plots), sharex=True, squeeze=False)
#
#for plot_idx, plot in enumerate(plots):
	#alphas = load_alphas( plot[1])
	#x_coordinates = range(len(alphas))
#
	#for num_alpha in range(len(alphas[0])):
		#axarr[plot_idx,0].plot(x_coordinates, alphas[:,num_alpha])
	#axarr[plot_idx,0].set_title(plot[0])
	#axarr[plot_idx,0].set_ylim([-1.2,1])

##### Plot without subfigures: 
if SAVE:
	#fig = texfig.figure(width=8.268) #entire page
	fig = texfig.figure(width=4.8)

alphas = load_alphas( plots[0][1])
x_coordinates = range(len(alphas))

for alpha_idx in range(len(alphas[0])):
	plt.plot(x_coordinates, alphas[:,alpha_idx], label="alpha "+str(alpha_idx+1))
	#axarr[plot_idx,0].set_title(plot[0])
	#axarr[plot_idx,0].set_ylim([-1.2,1])

plt.xlabel("Iterations")
plt.ylabel("Coefficient value")
plt.ylim([-1.0,0.6])
plt.xlim([0,100])
#plt.legend(loc=3)
#plt.legend(loc=8, ncol=int(len(alphas[0])/2), handletextpad=0, columnspacing=0)
plt.legend(loc=7, ncol=1, handletextpad=0, columnspacing=0, labelspacing=0.2)
#plt.setp(plt.gca().get_legend().get_texts(), fontsize='12') #legend 'list' fontsize
#plt.legend(loc=9, ncol=len(alphas[0])/2, numpoints=1, handletextpad=0, columnspacing=0.5)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #place to the right
#plt.grid(True)


plt.show()

if SAVE:
	texfig.savefig(SAVE)
