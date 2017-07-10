#!/usr/bin/env python3.5
import sys, os
import numpy as np
import IJB_A_template_lib as itl



def read_angles_of_single_img_fitting_log_file(log_file_path):
	with open(log_file_path, 'r') as file:
		for line in file:
	
			if not line.startswith('lm_file'): #skip header
				continue
			line = file.readline()
			angles = line.split(', ')[1:4]
			angles = [float(a) for a in angles]
			return angles




verification_exp_base = '/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_91/'
MATCHING_FOLDER = 'matching_cos/'

if not os.path.exists(verification_exp_base):
	os.mkdir(verification_exp_base)
if not os.path.exists(verification_exp_base+MATCHING_FOLDER):
	os.mkdir(verification_exp_base+MATCHING_FOLDER)

merge_A_base = '/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_99/'
#merge_A_base = '/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_03/'
merge_B_base = '/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_13/'
#merge_B_base = '/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_04/'

total_pos = 0
total_neg = 0
ho_pos_used = 0
ho_neg_used = 0
ph_pos_used = 0
ph_neg_used = 0
ho_used = 0
ph_used = 0

ph_template_pairs_better_pos = []
ph_template_pairs_better_neg = []

for split in range(1,11):
	print ('merging split',split)
	matches_file = merge_A_base+MATCHING_FOLDER+'split'+str(split)+'.matches'
	comparisons_A, templates_A = itl.read_matching_output(matches_file)
	#scores = [comparison[2] for comparison in comparisons_A]
	#scores = [item for item in scores if item != 'fte'] 
	#hist, bin_edges = np.histogram(scores, bins=[-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	#print('HO score histo:', bin_edges, hist)

	matches_file = merge_B_base+MATCHING_FOLDER+'split'+str(split)+'.matches'
	comparisons_B, templates_B = itl.read_matching_output(matches_file)
	#scores = [comparison[2] for comparison in comparisons_B]
	#scores = [item for item in scores if item != 'fte'] 
	#hist, bin_edges = np.histogram(scores, bins=[-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	#print('PH score histo:',bin_edges, hist)
	#continue
	#exit(0)

	metadata_file_path = '/vol/vssp/datasets/still/IJB_A/11/split'+str(split)+'/verify_metadata_'+str(split)+'.csv'
	overall_templates_dict = itl.read_IJBA_templates_definition(metadata_file_path)

	if len(comparisons_A)!=len(comparisons_B): #or templates_A!=templates_B:
		print ('shit')
		exit(0)
	comparisons_merged = []
	for i in range(len(comparisons_A)):

		if comparisons_A[i][2]!= 'fte' and comparisons_B[i][2]!= 'fte':

			if True==False: #TAKE_BETTER_WITH_GT
				if templates_A[comparisons_A[i][0]].subject_id == templates_A[comparisons_A[i][1]].subject_id: # if same id
					total_pos +=1
					#take higher score
					if comparisons_A[i][2] > comparisons_B[i][2]:
						better_score = comparisons_A[i][2]
						ho_pos_used+=1
					else:
						better_score = comparisons_B[i][2]
						ph_template_pairs_better_pos.append([comparisons_A[i][0], comparisons_A[i][1], abs(comparisons_A[i][2]-comparisons_B[i][2])])
						ph_pos_used+=1
					#better_score = max([comparisons_A[i][2], comparisons_B[i][2]])
				else: #not same id
					total_neg+=1
					if comparisons_A[i][2] < comparisons_B[i][2]:
						better_score = comparisons_A[i][2]
						ho_neg_used+=1
					else:
						better_score = comparisons_B[i][2]
						ph_template_pairs_better_neg.append([comparisons_A[i][0], comparisons_A[i][1], abs(comparisons_A[i][2]-comparisons_B[i][2])])
						ph_neg_used+=1
					#better_score = min([comparisons_A[i][2], comparisons_B[i][2]]) 
			elif True==False: #TAKE_MEAN
				better_score = (comparisons_A[i][2] + comparisons_B[i][2])/2
			elif True==False: #TAKE LOWER
				if comparisons_A[i][2] < comparisons_B[i][2]:
					better_score = comparisons_A[i][2]
					ho_used+=1
				else:
					better_score = comparisons_B[i][2]
					ph_used+=1
			elif True==False: # fancy merging: default is HO score, only take PH score when one template only 1 image with high pose
				angle_threshold = 65
				use_ho=True
				
				
				#print (len(overall_templates_dict[comparisons_A[i][0]].images))
				fitting_log = '/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30_256/verification_templates/split'+str(split)+'/'+str(overall_templates_dict[comparisons_A[i][1]].template_id)+'/fitting.log'
				if len(overall_templates_dict[comparisons_A[i][0]].images)==1:
					# okay template consists of one image, now let's find out what pose it has!					
					angles = read_angles_of_single_img_fitting_log_file(fitting_log)
					if any( abs(a)>angle_threshold for a in angles): # if large pose, replace score with PH score
						use_ho=False
				if len(overall_templates_dict[comparisons_A[i][1]].images)==1:
					# okay template consists of one image, now let's find out what pose it has!
					angles = read_angles_of_single_img_fitting_log_file(fitting_log)
					if any( abs(a)>angle_threshold for a in angles): # if large pose, replace score with PH score
						use_ho=False
				if use_ho:
					better_score = comparisons_A[i][2]
					ho_used+=1
				else:
					better_score = comparisons_B[i][2]
					ph_used+=1
			elif True==True:
				#default ho
				use_ho=True
				if comparisons_A[i][2] >0.0 and comparisons_A[i][2] <0.7:
					if comparisons_B[i][2] <0.0 or comparisons_B[i][2] >0.7:
						use_ho=False

				if use_ho:
					better_score = comparisons_A[i][2]
					ho_used+=1
				else:
					better_score = comparisons_B[i][2]
					ph_used+=1


					#print (fitting_log)
					#exit(0)

	
		elif comparisons_A[i][2]!= 'fte':
			better_score = comparisons_A[i][2]
			ho_used+=1
		elif comparisons_B[i][2]!= 'fte':
			better_score = comparisons_B[i][2]
			ph_used+=1
		else:
			better_score = 'fte'

		comparisons_merged.append([comparisons_A[i][0], comparisons_A[i][1], better_score])

	itl.write_matching_output(comparisons_merged, templates_A, verification_exp_base+MATCHING_FOLDER+'split'+str(split)+'.matches')
	itl.write_sim_matrix(comparisons_merged, templates_A, verification_exp_base+MATCHING_FOLDER+'split'+str(split)+'.simmmatrix')


print ('total ho', ho_used)
print ('total ph', ph_used)

exit(0)
##### Rest is debug stuff!!!!
print ('total pos', total_pos)
print ('total neg', total_neg)
print ('ho pos used', ho_pos_used)
print ('ho neg used', ho_neg_used)
print ('ph pos used', ph_pos_used)
print ('ph neg used', ph_neg_used)

# sort our better template pairs according to improvement
ph_template_pairs_better_pos = [list(x) for x in set(tuple(x) for x in ph_template_pairs_better_pos)]
ph_template_pairs_better_neg = [list(x) for x in set(tuple(x) for x in ph_template_pairs_better_neg)]
ph_template_pairs_better_pos = sorted(ph_template_pairs_better_pos, key= lambda x: x[2])
ph_template_pairs_better_neg = sorted(ph_template_pairs_better_neg, key= lambda x: x[2])
#print (ph_template_pairs_better_pos[-200:])
#print (ph_template_pairs_better_neg[-200:])

templates_dict = {}
for split in range(1,11):
	metadata_file_path = '/vol/vssp/datasets/still/IJB_A/11/split'+str(split)+'/verify_metadata_'+str(split)+'.csv'
	templates_dict = {**templates_dict, **itl.read_IJBA_templates_definition(metadata_file_path)}
	print (len(templates_dict))

img_source = '/user/HS204/m09113/my_project_folder/IJB_A/input_org/'


pos_improved_debug_folder = '/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_98/pos_improved/'
for ph_template_pair in ph_template_pairs_better_pos[-50:]:
	#print ('templates',str(ph_template_pair[0]),'and',str(ph_template_pair[1]),'improved', str(ph_template_pair[2]))
	#print ('template',str(ph_template_pair[0]),'has images',templates_dict[ph_template_pair[0]].images)
	#print ('template',str(ph_template_pair[1]),'has images',templates_dict[ph_template_pair[1]].images)
	output_folder = pos_improved_debug_folder+"{:.3f}".format(ph_template_pair[2])+'_'+str(ph_template_pair[0])+'_'+str(ph_template_pair[1])+'/'
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)	
	for i in [0,1]:
		for template_img in templates_dict[ph_template_pair[i]].images:
			try:
				os.symlink(img_source+template_img, output_folder+str(ph_template_pair[i])+'_'+template_img.replace('/','_'))
			except FileExistsError:
				pass
neg_improved_debug_folder = '/user/HS204/m09113/my_project_folder/IJB_A/verification_exp_98/neg_improved/'
for ph_template_pair in ph_template_pairs_better_neg[-50:]:
	#print ('templates',str(ph_template_pair[0]),'and',str(ph_template_pair[1]),'improved', str(ph_template_pair[2]))
	#print ('template',str(ph_template_pair[0]),'has images',templates_dict[ph_template_pair[0]].images)
	#print ('template',str(ph_template_pair[1]),'has images',templates_dict[ph_template_pair[1]].images)
	output_folder = neg_improved_debug_folder+"{:.3f}".format(ph_template_pair[2])+'_'+str(ph_template_pair[0])+'_'+str(ph_template_pair[1])+'/'
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)	
	for i in [0,1]:
		for template_img in templates_dict[ph_template_pair[i]].images:
			try:
				os.symlink(img_source+template_img, output_folder+str(ph_template_pair[i])+'_'+template_img.replace('/','_'))
			except FileExistsError:
				pass


