#!/usr/bin/env python3.5
import sys, os
import numpy as np
import IJB_A_template_lib as itl
import obj_analysis_lib as oal

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


xmax = 210
histogram = xmax*[0.0]
total_num=0

for split in range(1,11):
		print('gathering features of split', split)
		#metadata_file_path = '/vol/vssp/datasets/still/IJB_A/11/split1/verify_metadata_1.csv'
		metadata_file_path = '/vol/vssp/datasets/still/IJB_A/11/split'+str(split)+'/verify_metadata_'+str(split)+'.csv'
		templates_dict = itl.read_IJBA_templates_definition(metadata_file_path)

		for template_key in templates_dict.keys():
			#print (templates_dict[template_key])
			histogram[len(templates_dict[template_key].images)] +=1
			total_num+=1

#bar_width = 0.3
print ('total', total_num)
x_coordinates = np.arange(len(histogram))
y_coordinates = histogram
#plt.errorbar(x_coordinates, y_coordinates, y_deviation, label="L2 error", marker="s",linestyle='None')
#barlist = plt.bar(x_coordinates,y_coordinates,width=bar_width)
barlist = plt.bar(x_coordinates,y_coordinates, log=True, bottom=1 )
#plt.yscale('log')
plt.xlim([0,xmax])

plt.xlabel("Number of images per template")
plt.ylabel("Frequency")


#print (histogram)


if SAVE4PRES:
	plt.savefig(SAVE4PRES+".png")

if SAVE4LATEX:
	texfig.savefig(SAVE4LATEX)

plt.show()
