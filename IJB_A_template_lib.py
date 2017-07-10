#!/usr/bin/env python3.5
import sys, os
import numpy as np
import eos_starter_lib as esl
from concurrent.futures import ThreadPoolExecutor
#from scipy.spatial import distance






class Template:

	def __init__(self):
		self.template_id = -1
		self.subject_id = -1
		self.images = []
		self.features = None

	def __str__(self):
		string = ''
		string += 'template_id: '+str(self.template_id)
		string += ', subject_id: '+str(self.subject_id)
		string += ', images: '+str(self.images)
		return string

def read_IJBA_templates_definition(metadata_file_path):
	all_templates = {}

	with open(metadata_file_path, 'r') as file:
		for line in file:
	
			if line.startswith('TEM'): #skip header
				continue
	
			parts = line.split(',')
			if int(parts[0]) in all_templates:
				all_templates[int(parts[0])].images.append(parts[2])
			else:
				temp = Template()
				temp.template_id = int(parts[0])
				temp.subject_id = int(parts[1])
				temp.images.append(parts[2])
				all_templates[temp.template_id]=temp
	return all_templates


def write_template_features(templates, file):
	with open(file, 'w') as out:
		out.write('TEMPLATE_ID,SUBJECT_ID,FEATURE_VECTOR\n')
		for template_key in templates.keys():
			string = ''
			string += str(templates[template_key].template_id) +','
			string += str(templates[template_key].subject_id) +','
			if not templates[template_key].features is None:
				for i in templates[template_key].features:
					string += str(i)+' '
			out.write(string+'\n')


def read_template_features(features_file):
	all_templates = {}

	with open(features_file, 'r') as file:
		for line in file:

			if line.startswith('TEM'): #skip header
				continue

			temp = Template()
			template_id, subject_id, features = line.split(',')
			temp.template_id = int(template_id)
			temp.subject_id = int(subject_id)
			temp.features = [float(x) for x in features.split()]
			all_templates[temp.template_id]=temp
	return all_templates

def read_comparisons(comp_file):
	comparisons=[]

	with open(comp_file, 'r') as file:
		for line in file:
			comparisons.append([int(x) for x in line.split(',')][:2])
	return comparisons

def write_matching_output(comparisons, templates, matches_file):

	with open(matches_file, 'w') as file:
		file.write('ENROLL_TEMPLATE_ID VERIF_TEMPLATE_ID ENROLL_SUBJECT_ID VERIF_SUBJECT_ID SIMILARITY_SCORE\n')
		for comparison in comparisons:
			string = str(comparison[0]) + ' ' + str(comparison[1]) + ' '
			string += str(templates[comparison[0]].subject_id) + ' ' +  str(templates[comparison[1]].subject_id) + ' ' + str(comparison[2])
			file.write(string + '\n')

def read_matching_output(matches_file):
	comparisons=[]
	templates = {}
	with open(matches_file, 'r') as file:
		header = file.readline()
		for line in file:
			enroll_template_id = line.split()[0]
			verif_template_id = line.split()[1]
			enroll_subject_id = line.split()[2]
			verif_subject_id = line.split()[3]
			score = line.split()[4]
			if score != 'fte':
				score = float(score)

			comparisons.append([int(enroll_template_id), int(verif_template_id), score])
			temp = Template()
			temp.template_id = int(enroll_template_id)
			temp.subject_id = int(enroll_subject_id)
			templates[temp.template_id]=temp
			temp = Template()
			temp.template_id = int(verif_template_id)
			temp.subject_id = int(verif_subject_id)
			templates[temp.template_id]=temp

	return comparisons, templates


def write_sim_matrix(comparisons, templates, matrix_file):
	same_id =[]
	different_id=[]

	for comparison in comparisons:
		if comparison[2] != 'fte':
			if templates[comparison[0]].subject_id == templates[comparison[1]].subject_id:
				same_id.append(comparison[2])
			else:
				different_id.append(comparison[2])

	with open(matrix_file, 'w') as file:
		file.write(str(len(comparisons))+'\n')
		for i in same_id:
			file.write(str(i)+" ")
		file.write("\n")
		for i in different_id:
			file.write(str(i)+" ")
		file.write("\n")


def read_cnn_features_file(features_file):
	features = {}
	with open(features_file, 'r') as file:
		for line in file:
			path = line.split()[0]
			file = path.split('/')[-1][:-len('.isomap.png')]
			id_features = [float(x) for x in line.split()[1:]]
			features[file] = id_features
	return features

def read_ho_features_file(features_file):
	features = {}
	with open(features_file, 'r') as file:
		for line in file:
			path = line.split()[0]
			file = path
			#file = path.split('/')[-1][:-len('.isomap.png')]
			id_features = [float(x) for x in line.split()[1:]]
			features[file] = id_features
	return features


def read_fb_cnn_features_file(features_file):
	features = {}
	with open(features_file, 'r') as file:
		for line in file:
			path = line.split()[0]
			file = path.split('/')[-2]+'/'+path.split('/')[-1]
			file = file.split('.')[0]
			id_features = [float(x) for x in line.split()[1:]]
			features[file] = id_features
	return features

def read_merged_cnn_features_file(features_file):
	features = {}
	with open(features_file, 'r') as file:
		for line in file:
			path = line.split()[0]
			file = path.split('/')[-2]+'/'+path.split('/')[-1]
			file = file.split('.')[0]
			id_features = [float(x) for x in line.split()[1:]]
			if file in features:
				print('okay, gotta problem')
			features[file] = id_features
	return features


