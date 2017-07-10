#!/usr/bin/env python3.5
import sys, os
import numpy as np
#import IJB_A_template_lib as itl
#import obj_analysis_lib as oal
import cnn_db_loader

#from scipy.spatial import distance

OUTPUTPATH="/user/HS204/m09113/my_project_folder/Results/IJB_A_template_size"
SAVE4PRES =None
SAVE4LATEX=None

#SAVE4LATEX=OUTPUTPATH
#SAVE4PRES =OUTPUTPATH


if SAVE4LATEX:
	import texfig
import matplotlib.pyplot as plt

if SAVE4LATEX:
	fig = texfig.figure(width=5.8)
	#fig = texfig.figure(width=20) #entire page

if SAVE4PRES:
	#plt.rcParams["font.family"] ="monospace"
	plt.figure(figsize=(10, 8/3*2))


cnn_db_loader.NUMBER_ALPHAS = 0
cnn_db_loader.NUMBER_IMAGES = 1
cnn_db_loader.NUMBER_XYZ = 0

#xmax = 210
#histogram = xmax*[0.0]


db_loader = cnn_db_loader.lazy_dummy('/user/HS204/m09113/my_project_folder/cnn_experiments/05/db_input/')

image_list, labels_list = db_loader.get_training_image_and_label_lists()
total_num=len(labels_list)

unique_labels = sorted(list(set(labels_list)))
xmax = len(unique_labels)
num_imgs_each_id = xmax*[0]


#first count imgs for each id
for label in labels_list:
	num_imgs_each_id[unique_labels.index(label)]+=1

xmax = max(num_imgs_each_id)+1
histogram = xmax*[0]
for num_imgs in num_imgs_each_id:
	histogram[num_imgs]+=1

#print (type(labels_list[5]))
#exit(0)

#bar_width = 0.3
print ('total', total_num)
x_coordinates = np.arange(len(histogram))
y_coordinates = histogram
#plt.errorbar(x_coordinates, y_coordinates, y_deviation, label="L2 error", marker="s",linestyle='None')
#barlist = plt.bar(x_coordinates,y_coordinates,width=bar_width)
barlist = plt.bar(x_coordinates,y_coordinates, log=True, bottom=1 )
#plt.yscale('log')
plt.xlim([0,210])

plt.xlabel("Number of images we have of id")
plt.ylabel("Frequency")


#print (histogram)


if SAVE4PRES:
	plt.savefig(SAVE4PRES+".png")

if SAVE4LATEX:
	texfig.savefig(SAVE4LATEX)

plt.show()
