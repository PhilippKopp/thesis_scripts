#!/usr/bin/env python3.5

import sys
import numpy as np
from cycler import cycler
from math import sqrt
import glob
import sklearn.metrics

OUTPUTPATH="/user/HS204/m09113/my_project_folder/Results/roc_IJB_A_verification_input"
#OUTPUTPATH="/user/HS204/m09113/my_project_folder/Results/roc_IJB_A_verification"
SAVE4PRES =None
SAVE4LATEX=None

#SAVE4LATEX=OUTPUTPATH
#SAVE4PRES =OUTPUTPATH


if SAVE4LATEX:
	import texfig
import matplotlib.pyplot as plt

if SAVE4LATEX:
	#fig = texfig.figure(width=10.268) #entire page
	fig = texfig.figure(width=5.8) #entire page

if SAVE4PRES:
	#plt.rcParams["font.family"] ="monospace"
	plt.figure(figsize=(10, 8/3*2))


plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'gold', 'm', 'k', 'slategray', 'peru'])))
#(201/255, 169/255, 000/255), (000/255, 70/255, 160/255), (0/255, 0/255, 0/255),

#plt.title("Face Recognition PaSC Videos")
#plots = [ ["PaSC control all alphas without fte", "/user/HS204/m09113/my_project_folder/PaSC/multi_fit_CCR_iter75_reg30_control_without_fte.csv" ],
#		  ["PaSC handheld all alphas without fte", "/user/HS204/m09113/my_project_folder/PaSC/multi_fit_CCR_iter75_reg30_handheld_without_fte.csv" ],
#		  ["PaSC control 10 alphas without fte", "/user/HS204/m09113/my_project_folder/PaSC/multi_fit_CCR_iter75_reg30_only_10_alphas_control_without_fte.csv" ],
#		  ["PaSC handheld 10 alphas without fte", "/user/HS204/m09113/my_project_folder/PaSC/multi_fit_CCR_iter75_reg30_only_10_alphas_handheld_without_fte.csv" ],
#		]
#plots = [ ["PaSC control set", "/user/HS204/m09113/my_project_folder/PaSC/multi_fit_CCR_iter75_reg30_control_without_fte.csv" ],
#		  ["PaSC handheld set", "/user/HS204/m09113/my_project_folder/PaSC/multi_fit_CCR_iter75_reg30_handheld_without_fte.csv" ]
#		]

#plt.title("ROC IJB-A verification")

#plots_single = [ ["alpha only1", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_00/matching/split1.simmmatrix" ],
#		  ["alpha only2", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_00/matching/split2.simmmatrix" ],
#		  ["alpha only3", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_00/matching/split3.simmmatrix" ],
#		  ["alpha only4", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_00/matching/split4.simmmatrix" ],
#		  ["alpha only5", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_00/matching/split5.simmmatrix" ],
#		  ["alpha only6", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_00/matching/split6.simmmatrix" ],
#		  ["alpha only7", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_00/matching/split7.simmmatrix" ],
#		  ["alpha only8", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_00/matching/split8.simmmatrix" ],
#		  ["alpha only9", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_00/matching/split9.simmmatrix" ],
#		  ["alpha only10", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_01/matching/split10.simmmatrix" ],
#		  #["alpha only1", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_01/matching/split1.simmmatrix" ],
#		  #["alpha only2", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_01/matching/split2.simmmatrix" ],
#		  #["alpha only3", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_01/matching/split3.simmmatrix" ],
#		  #["alpha only4", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_01/matching/split4.simmmatrix" ],
#		  #["alpha only5", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_01/matching/split5.simmmatrix" ],
#		  #["alpha only6", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_01/matching/split6.simmmatrix" ],
#		  #["alpha only7", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_01/matching/split7.simmmatrix" ],
#		  #["alpha only8", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_01/matching/split8.simmmatrix" ],
#		  #["alpha only9", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_01/matching/split9.simmmatrix" ],
#		  #["alpha only10", "/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_01/matching/split10.simmmatrix" ],
#		  #["PaSC handheld set", "/user/HS204/m09113/my_project_folder/PaSC/multi_fit_CCR_iter75_reg30_handheld_without_fte.csv" ]
#		]

plots = [ #["alphas", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_00/matching_cos/split*.simmmatrix") ], 
		  #["alphas only l2", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_00/matching_l2/split?.simmmatrix") ], 
		  #["cnn exp 1: alexNetv1 cos", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_01/matching_cos/split?.simmmatrix" )],
		  #["xyz alexNet", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_04/matching_cos/split*.simmmatrix" )],
		  #["rgb alexNet", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_02/matching_cos/split*.simmmatrix" )],
		  #["facebox alexNet", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_09/matching_cos/split*.simmmatrix" )],
		  #["rgb alexNet 256 cos", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_05/matching_cos/split*.simmmatrix" )],
		  #["rgb+alphas alexNet", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_11/matching_cos/split*.simmmatrix" )],
		  #["rgb+xyz alexNet", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_12/matching_cos/split*.simmmatrix" )],
		  #["cnn + alphas", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_02/matching/split?.simmmatrix" )],
		  #["rgb dcnn", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_03/matching_cos/split*.simmmatrix" )],
		  ["rgb+xyz dcnn", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_13/matching_cos/split*.simmmatrix" )],
		  ["Ho", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_99/matching_cos/split*.simmmatrix" )],
		  ["ideal Ho PH merge (36142)", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_98/matching_cos/split*.simmmatrix" )],
		  ["Ho PH merge, taking higher (64770)", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_97/matching_cos/split*.simmmatrix" )],
		  ["Ho PH merge, taking lower (52464)", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_96/matching_cos/split*.simmmatrix" )],
		  ["Ho PH merge, taking fancy 25 (29263)", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_95/matching_cos/split*.simmmatrix" )],
		  ["Ho PH merge, taking fancy 45 (15787)", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_94/matching_cos/split*.simmmatrix" )],
		  ["Ho PH merge, taking fancy 65 (1700)", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_93/matching_cos/split*.simmmatrix" )],
		  ["Ho PH merge, more confident -0.1 <> 0.8 (1652)", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_92/matching_cos/split*.simmmatrix" )],
		  ["Ho PH merge, more confident 0.0 <> 0.7 (5227)", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_91/matching_cos/split*.simmmatrix" )],
		  #["score mean xyz alex+rgb dcnn", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_80/matching_cos/split*.simmmatrix" )],
		  #["conf13sm alex merge 50", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_50/matching_cos/split*.simmmatrix" )],
		  #["conf13sm alex merge 51", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_51/matching_cos/split*.simmmatrix" )],
		  #["conf13sm best1 alex 55", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_55/matching_cos/split*.simmmatrix" )],
		  #["conf13sm best1 alex 56", glob.glob("/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_56/matching_cos/split*.simmmatrix" )],
		]


def load_matrix( csv_file):
	with open(csv_file, "r") as file:
		matches = int(file.readline())
		same_id = file.readline()
		same_id = [float(i) for i in same_id.split()]
		#file.readline()
		different_id = file.readline()
		different_id = [float(i) for i in different_id.split()]
	return matches, same_id, different_id
#exit()


def quantise(x_vals, y_vals, rate):
	#x_quant = []
	y_quant = []
	for r in rate:
		idx_after = next(i for i in enumerate(x_vals) if i[1]>r)[0]
		idx_before = idx_after-1
		y_quant.append((y_vals[idx_before]+y_vals[idx_after])/2)
	return y_quant


steps = [1/(1.01**x) for x in range(1,800)]

#for plot_idx, plot in enumerate(plots_single):
#	matches, same_id, different_id = load_matrix( plot[1])
#
#	# assemble gt label vector
#	gt_labels = [0]*len(same_id)
#	gt_labels.extend([1]*len(different_id))
#	gt_labels =np.array(gt_labels)
#	#print (np.unique(gt_labels))
#
#	# assemble score vector
#	scores = []
#	scores.extend(same_id)
#	scores.extend(different_id)
#	scores = np.array(scores)
#
#	fpr, tpr, _ = sklearn.metrics.roc_curve(gt_labels, scores, pos_label=0)
#	tpr = tpr*(len(same_id)+len(different_id))/matches # IS THIS CORRECT FOR FAILED TO ENROL HANDLING???
#	#print (fpr[:20])
#
#	plt.plot(fpr, tpr, label=plot[0])


for plot in plots:
	curve_parts_y_quant = []
	for split_curve in plot[1]:
		matches, same_id, different_id = load_matrix( split_curve)

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
		tpr = tpr*(len(same_id)+len(different_id))/matches # IS THIS CORRECT FOR FAILED TO ENROL HANDLING???

		y_quant = quantise(fpr,tpr,steps)
		curve_parts_y_quant.append(y_quant)

	mean_curve = np.mean(curve_parts_y_quant, axis = 0)
	std_mean_curve = np.std(curve_parts_y_quant, axis = 0)

	# plot all points of curve
	#plt.plot(steps, mean_curve, label=plot[0]+'_quant')

	#plot std deviation for some points
	plt.errorbar(steps[::20], mean_curve[::20], std_mean_curve[::20], label=plot[0])

#plt.plot(steps, [0.5]*len(std_steps), marker='s', label='test')

plt.plot([x / 1000.0 for x in range(0, 1000, 1)], [x / 1000.0 for x in range(0, 1000, 1)], linestyle='--', label='random')
#plt.plot([0.001, 0.01, 0.1, 1],[0.05, 0.2, 0.5, 0.96], '*-', label='PaSC Control Surrey')
#plt.plot([0.001, 0.01, 0.1, 1],[0.03, 0.13, 0.36, 0.96], '*-', label='PaSC Handheld Surrey')
plt.errorbar([0.001, 0.01, 0.1],[0.198, 0.406, 0.627], [0.008, 0.014, 0.012], marker='s', label='GOTS') # from IJB-A paper
plt.errorbar([0.001, 0.01, 0.1],[0.104, 0.236, 0.433], [0.014, 0.009, 0.006], marker='s', label='OpenBR') # from IJB-A paper
plt.errorbar([0.01, 0.1],[0.787, 0.947], [0.043, 0.011], marker='s', label='DCNN') #Unconstrained Face Verification using Deep CNN Features
plt.xscale('log')
plt.xlim([0.0005,1])
plt.ylim([0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc=4)


if SAVE4PRES:
	plt.savefig(SAVE4PRES+".png")

if SAVE4LATEX:
	texfig.savefig(SAVE4LATEX)

plt.show()
