#!/usr/bin/env python3.5


import os
import glob
import random
import numpy as np
import cv2
#import tensorflow as tf

ID_CLASS_MAPPING_POSTFIX = 'id_class_mapping.txt'
LIL_ALL_POSTFIX = 'labeled_images_all.txt'
LIL_TRAIN_POSTFIX = 'labeled_images_train.txt'
LIL_EVAL_POSTFIX = 'labeled_images_eval.txt'
MEAN_IMAGE_POSTFIX = 'mean.png'


class DB_loader:

	def __init__(self, outputfolder, db_base):
		self.db_base = db_base
		self.outputfolder = outputfolder

		#self.id_class_mapping_filename = ID_CLASS_MAPPING_POSTFIX
		#self.labeled_image_list_filename = LIL_ALL_POSTFIX
		#self.lil_train_filename = LIL_TRAIN_POSTFIX
		#self.lil_eval_filename = LIL_EVAL_POSTFIX
		
		self.id_class_mapping = {}
		self.number_ids = 0

		self.num_examples_all = 0
		self.images_all = []
		self.labels_all = []

		self.num_examples_train = 0
		self.images_train = []
		self.labels_train = []

		self.num_examples_eval = 0
		self.images_eval = []
		self.labels_eval = []

		self.get_mean_image = None

	
	def read_db(self):
		print ('reading db ...')
		# read or create id mapping
		if os.path.exists(self.outputfolder+'/'+self.id_class_mapping_filename):
			self.read_id_class_mapping(self.outputfolder+'/'+self.id_class_mapping_filename)
		else:
			print('no db id mapping file found. Creating...')
			self.generate_id_class_mapping()
			self.write_id_class_mapping(self.outputfolder+'/'+self.id_class_mapping_filename)

		# read test and eval sets
		#if os.path.exists(self.outputfolder+'/'+self.lil_train_filename) or os.path.exists(self.outputfolder+'/'+self.lil_eval_filename):
		#	if os.path.exists(self.outputfolder+'/'+self.lil_train_filename):
		#		self.images_train, self.labels_train, self.num_examples_train = self.read_labeled_image_list(self.outputfolder+'/'+self.lil_train_filename)
		#	
		#	if os.path.exists(self.outputfolder+'/'+self.lil_eval_filename):
		#		self.images_eval , self.labels_eval , self.num_examples_eval  = self.read_labeled_image_list(self.outputfolder+'/'+self.lil_eval_filename)
		#else:

			# read entire set
		if os.path.exists(self.outputfolder+'/'+self.labeled_image_list_filename):
			self.images_all, self.labels_all, self.num_examples_all = self.read_labeled_image_list(self.outputfolder+'/'+self.labeled_image_list_filename)
		else:
			print('no labeled image list file found. Creating...')
			self.generate_labeled_image_list()
			self.write_labeled_image_list(self.images_all, self.labels_all, self.outputfolder+'/'+self.labeled_image_list_filename)

			#print('no train and eval splits found. Creating...')
			#self.split_train_eval()
			#self.write_labeled_image_list(self.images_train, self.labels_train, self.outputfolder+'/'+self.lil_train_filename)
			#self.write_labeled_image_list(self.images_eval,  self.labels_eval,  self.outputfolder+'/'+self.lil_eval_filename)

	def write_id_class_mapping(self, file_path):
		with open(file_path,'w') as out:
			for item in self.id_class_mapping.items():
				out.write(item[0]+' '+str(item[1])+'\n')

	def read_id_class_mapping(self, file_path):
		id_class_mapping = {}
		self.number_ids=0
		with open(file_path,'r') as file:
			for line in file:
				id_name, id_idx = line[:-1].split(' ')
				self.id_class_mapping[id_name]=int(id_idx)
				self.number_ids+=1


	def write_labeled_image_list(self, images, labels, file_path):
		with open(file_path,'w') as out:
			for idx in range(len(images)):
				out.write(images[idx]+' '+str(labels[idx])+'\n')

	def read_labeled_image_list(self, file_path):
		# copied from http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
		"""Reads a .txt file containing pathes and labeles
		Args:
		   file_path: a .txt file with one /path/to/image per line
		   label: optionally, if set label will be pasted after each line
		Returns:
		   List with all filenames in file file_path
		"""
		images = []
		labels = []
		num_examples = 0
		f = open(file_path, 'r')
		for line in f:
			filename, label = line[:-1].split(' ')
			images.append(filename)
			labels.append(int(label))
			num_examples += 1
		return images, labels, num_examples


	def get_mean_image_path(self):
		if not os.path.exists(self.outputfolder+'/'+self.mean_image_filename):
			print('no mean image found. Creating...')
			isomap_size = cv2.imread(self.images_train[0], cv2.IMREAD_COLOR).shape[0]
			mean = np.zeros([isomap_size, isomap_size, 3], dtype='float32')

			for image_path in self.images_train:
				mean+=cv2.imread(image_path,cv2.IMREAD_COLOR).astype(dtype='float32')/len(self.images_train)
			#mean/=len(self.images_train)
			mean_uint8 = mean.astype(dtype='uint8')
			cv2.imwrite(self.outputfolder+'/'+self.mean_image_filename, mean_uint8)
		return self.outputfolder+'/'+self.mean_image_filename

	def set_all_as_train(self):
		self.num_examples_train = self.num_examples_all
		self.images_train = self.images_all
		self.labels_train = self.labels_all

		self.write_labeled_image_list(self.images_train, self.labels_train, self.outputfolder+'/'+self.lil_train_filename)



class Aggregator(DB_loader):

	def __init__(self, *args):
		DB_loader.__init__(self, outputfolder=args[0].outputfolder, db_base=None)
		self.id_class_mapping_filename = 'total_' + ID_CLASS_MAPPING_POSTFIX
		self.labeled_image_list_filename = 'total_' + LIL_ALL_POSTFIX
		self.lil_train_filename = 'total_' + LIL_TRAIN_POSTFIX
		self.lil_eval_filename = 'total_' + LIL_EVAL_POSTFIX
		self.mean_image_filename = 'total_' + MEAN_IMAGE_POSTFIX

		for arg in args:

			if not isinstance(arg, PaSC_video_loader):
				for item in arg.id_class_mapping.items():
					arg.id_class_mapping.update({item[0]: item[1]+self.number_ids})
					#print (item[1])
					#item[1]+= self.number_ids
				self.id_class_mapping.update(arg.id_class_mapping)
				self.number_ids += arg.number_ids
	
			self.num_examples_all += arg.num_examples_all
			self.images_all.extend(arg.images_all)
			self.labels_all.extend(arg.labels_all)
	
			self.num_examples_train += arg.num_examples_train
			self.images_train.extend(arg.images_train)
			self.labels_train.extend(arg.labels_train)
	
			self.num_examples_eval += arg.num_examples_eval
			self.images_eval.extend(arg.images_eval)
			self.labels_eval.extend(arg.labels_eval)

		random.seed(404)
		random.shuffle(self.images_train)
		random.seed(404)
		random.shuffle(self.labels_train)
		#print ('ids',self.number_ids)
		#print ('imgs all',len(self.images_all))
		#print ('imgs train',len(self.images_train))
		#print ('imgs eval',len(self.images_eval))
		self.write_id_class_mapping(self.outputfolder+'/'+self.id_class_mapping_filename)
		self.write_labeled_image_list(self.images_all,   self.labels_all,   self.outputfolder+'/'+self.labeled_image_list_filename)
		self.write_labeled_image_list(self.images_train, self.labels_train, self.outputfolder+'/'+self.lil_train_filename)
		self.write_labeled_image_list(self.images_eval,  self.labels_eval,  self.outputfolder+'/'+self.lil_eval_filename)



class lazy_dummy(DB_loader):
	def __init__(self, folder):
		DB_loader.__init__(self, outputfolder=folder, db_base=None)
		self.id_class_mapping_filename = 'total_' + ID_CLASS_MAPPING_POSTFIX
		self.labeled_image_list_filename = 'total_' + LIL_ALL_POSTFIX
		self.lil_train_filename = 'total_' + LIL_TRAIN_POSTFIX
		self.lil_eval_filename = 'total_' + LIL_EVAL_POSTFIX
		self.mean_image_filename = 'total_' + MEAN_IMAGE_POSTFIX

		self.read_id_class_mapping(self.outputfolder+'/'+self.id_class_mapping_filename)
		self.images_train, self.labels_train, self.num_examples_train = self.read_labeled_image_list(self.outputfolder+'/'+self.lil_train_filename)
		self.images_eval , self.labels_eval , self.num_examples_eval  = self.read_labeled_image_list(self.outputfolder+'/'+self.lil_eval_filename)




class PaSC_video_loader(DB_loader):


	def __init__(self, outputfolder, db_base):
		DB_loader.__init__(self, outputfolder, db_base)
		self.id_class_mapping_filename = 'pasc_' + ID_CLASS_MAPPING_POSTFIX
		self.labeled_image_list_filename = 'pasc_video_' + LIL_ALL_POSTFIX
		self.lil_train_filename = 'pasc_video_' + LIL_TRAIN_POSTFIX
		self.lil_eval_filename = 'pasc_video_' + LIL_EVAL_POSTFIX
		self.mean_image_filename = 'pasc_video_' + MEAN_IMAGE_POSTFIX
		self.read_db()



	def generate_id_class_mapping(self):
		print ('don\'t use PaSC video id class mapping as not all ids have videos. Use PaSC still instead!')
		exit()
		#folders = glob.glob(self.db_base+'/*')
		#folders = [os.path.basename(x) for x in folders]
		#folders = [x[:5] for x in folders]
		#ids = list(set(folders))
		#ids.sort()
		#self.id_class_mapping = {}
		#for indx, id_ in enumerate(ids):
		#	self.id_class_mapping[id_]=indx
		#self.number_ids = len(self.id_class_mapping)
	

			
	def generate_labeled_image_list(self):
		folders = glob.glob(self.db_base+'/*')
		self.images_all = []
		self.labels_all = []
		self.num_examples_all = 0
		for folder in folders:
			id_ = self.id_class_mapping[os.path.basename(folder)[:5]]
			folder_images = glob.glob(folder+'/*isomap.png')
			for folder_image in folder_images:
				self.images_all.append(folder_image)
				self.labels_all.append(id_)
				self.num_examples_all += 1

	def split_train_eval(self, train_proportion=0.8):
		#random.seed(404)
		#random.shuffle(self.images_all)
		#random.seed(404)
		#random.shuffle(self.labels_all)

		self.num_examples_train = int(self.num_examples_all*train_proportion)
		self.num_examples_eval = self.num_examples_all - self.num_examples_train

		self.images_train = self.images_all[:self.num_examples_train]
		self.labels_train = self.labels_all[:self.num_examples_train]

		self.images_eval = self.images_all[self.num_examples_train:]
		self.labels_eval = self.labels_all[self.num_examples_train:]

		self.write_labeled_image_list(self.images_train, self.labels_train, self.outputfolder+'/'+self.lil_train_filename)
		self.write_labeled_image_list(self.images_eval,  self.labels_eval,  self.outputfolder+'/'+self.lil_eval_filename)


class PaSC_still_loader(DB_loader):


	def __init__(self, outputfolder, db_base):
		DB_loader.__init__(self, outputfolder, db_base)
		self.id_class_mapping_filename = 'pasc_' + ID_CLASS_MAPPING_POSTFIX
		self.labeled_image_list_filename = 'pasc_still_' + LIL_ALL_POSTFIX
		self.lil_train_filename = 'pasc_still_' + LIL_TRAIN_POSTFIX
		self.lil_eval_filename = 'pasc_still_' + LIL_EVAL_POSTFIX
		self.mean_image_filename = 'pasc_still_' + MEAN_IMAGE_POSTFIX
		self.read_db()



	def generate_id_class_mapping(self):
		folders = glob.glob(self.db_base+'/*')
		folders = [os.path.basename(x) for x in folders]
		#folders = [x[:6] for x in folders]
		ids = list(set(folders))
		ids.sort()
		self.id_class_mapping = {}
		for indx, id_ in enumerate(ids):
			self.id_class_mapping[id_]=indx
		self.number_ids = len(self.id_class_mapping)
	

			
	def generate_labeled_image_list(self):
		folders = glob.glob(self.db_base+'/*')
		self.images_all = []
		self.labels_all = []
		self.num_examples_all = 0
		for folder in folders:
			id_ = self.id_class_mapping[os.path.basename(folder)[:5]]
			folder_images = glob.glob(folder+'/*isomap.png')
			for folder_image in folder_images:
				self.images_all.append(folder_image)
				self.labels_all.append(id_)
				self.num_examples_all += 1


class CASIA_webface_loader(DB_loader):


	def __init__(self, outputfolder, db_base):
		DB_loader.__init__(self, outputfolder, db_base)
		self.id_class_mapping_filename = 'casia_' + ID_CLASS_MAPPING_POSTFIX
		self.labeled_image_list_filename = 'casia_' + LIL_ALL_POSTFIX
		self.lil_train_filename = 'casia_' + LIL_TRAIN_POSTFIX
		self.lil_eval_filename = 'casia_' + LIL_EVAL_POSTFIX
		self.mean_image_filename = 'casia_' + MEAN_IMAGE_POSTFIX
		self.read_db()



	def generate_id_class_mapping(self):
		folders = glob.glob(self.db_base+'/*')
		folders = [os.path.basename(x) for x in folders]
		#folders = [x[:6] for x in folders]
		ids = list(set(folders))
		ids.sort()
		self.id_class_mapping = {}
		for indx, id_ in enumerate(ids):
			self.id_class_mapping[id_]=indx
		self.number_ids = len(self.id_class_mapping)
	

			
	def generate_labeled_image_list(self):
		folders = glob.glob(self.db_base+'/*')
		self.images_all = []
		self.labels_all = []
		self.num_examples_all = 0
		for folder in folders:
			id_ = self.id_class_mapping[os.path.basename(folder)[:7]]
			folder_images = glob.glob(folder+'/*isomap.png')
			for folder_image in folder_images:
				self.images_all.append(folder_image)
				self.labels_all.append(id_)
				self.num_examples_all += 1











