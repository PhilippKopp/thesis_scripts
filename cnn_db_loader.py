#!/usr/bin/env python3.5


import os, sys
import glob
import random
import numpy as np
import cv2
import obj_analysis_lib as oal
#import tensorflow as tf

ID_CLASS_MAPPING_POSTFIX = 'id_class_mapping.txt'
LIL_ALL_POSTFIX = 'labeled_images_all.txt'
LIL_TRAIN_POSTFIX = 'labeled_images_train.txt'
LIL_EVAL_POSTFIX = 'labeled_images_eval.txt'
MEAN_IMAGE_POSTFIX = 'mean.png'
MEAN_XYZ_POSTFIX = 'mean_xyz.png'

NUMBER_IMAGES = None
NUMBER_ALPHAS = None
NUMBER_XYZ = None
LEN_ALPHAS = 63
IMAGE_FILE_ENDING = '/*isomap.png'


class Training_example:
	def __init__(self):
		self.images = []
		self.xyz = []
		self.alphas = []
		self.label = -1

	def __str__(self):
		return str(self.images)+' '+str(self.alphas)+' '+str(self.label)

def _calc_isomap_coverage(isomap):
	# small helper function
	visible = isomap[:,:,3]>0 
	pixels = isomap.shape[0]*isomap.shape[1]
	return np.sum(visible)/pixels

def _get_gradient_magnitude(im):
	"Get magnitude of gradient for given image"
	ddepth = cv2.CV_32F
	dx = cv2.Sobel(im, ddepth, 1, 0)
	dy = cv2.Sobel(im, ddepth, 0, 1)
	dxabs = cv2.convertScaleAbs(dx)
	dyabs = cv2.convertScaleAbs(dy)
	mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)

	return np.average(mag)

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

		self.examples_all = []
		self.examples_train = []
		self.examples_eval = []

	
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
			self.examples_all = self.read_labeled_image_list(self.outputfolder+'/'+self.labeled_image_list_filename)
		else:
			print('no labeled image list file found. Creating...')
			self.generate_labeled_image_list()
			if NUMBER_IMAGES>1:
				print('grouping images...')
				self.group_to_more_than_1_image()
			self.write_labeled_image_list(self.examples_all, self.outputfolder+'/'+self.labeled_image_list_filename)

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


	def write_labeled_image_list(self, examples, file_path):
		with open(file_path,'w') as out:
			for example in examples:
				for i in range(NUMBER_IMAGES):
					out.write(example.images[i]+' ')
				for i in range(NUMBER_XYZ):
					out.write(example.xyz[i]+' ')
				for i in range(NUMBER_ALPHAS):
					for a in example.alphas[i]:
						out.write(str(a)+' ')
				out.write(str(example.label)+'\n')

	def read_labeled_image_list(self, file_path):
		# copied from http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
		"""Reads a .txt file containing pathes and labeles
		Args:
		   file_path: a .txt file with one /path/to/image per line
		   label: optionally, if set label will be pasted after each line
		Returns:
		   List with all filenames in file file_path
		"""
		examples = []
		f = open(file_path, 'r')
		for line in f:
			parts = line[:-1].split(' ')
			example = Training_example()
			example.images+=parts[:NUMBER_IMAGES]
			example.xyz += parts[NUMBER_IMAGES:NUMBER_IMAGES+NUMBER_XYZ]
			for i in range(NUMBER_ALPHAS):
				example.alphas.append([float(x) for x in parts[NUMBER_IMAGES+NUMBER_XYZ + i*LEN_ALPHAS: NUMBER_IMAGES+NUMBER_XYZ + (i+1)*LEN_ALPHAS]])
			example.label = int(parts[-1])
			examples.append(example)
		return examples


	def get_mean_image_path(self):
		if not os.path.exists(self.outputfolder+'/'+self.mean_image_filename):
			print('no mean image found. Creating...')
			isomap_size = cv2.imread(self.examples_train[0].images[0], cv2.IMREAD_COLOR).shape[0]
			mean = np.zeros([isomap_size, isomap_size, 3], dtype='float32')

			for example in self.examples_train:#
				try:
					mean+=cv2.imread(example.images[0],cv2.IMREAD_COLOR).astype(dtype='float32')/len(self.examples_train)
				except:
					e = sys.exc_info()[0]
					print (str(e))
					print ('image', example.images[0])
					exit(0)
			#mean/=len(self.images_train)
			mean_uint8 = mean.astype(dtype='uint8')
			cv2.imwrite(self.outputfolder+'/'+self.mean_image_filename, mean_uint8)
		return self.outputfolder+'/'+self.mean_image_filename

	def get_mean_xyz_path(self):
		if not os.path.exists(self.outputfolder+'/'+self.mean_xyz_filename):
			print('no mean image found. Creating...')
			isomap_size = cv2.imread(self.examples_train[0].xyz[0], cv2.IMREAD_COLOR).shape[0]
			mean = np.zeros([isomap_size, isomap_size, 3], dtype='float32')

			for example in self.examples_train:
				mean+=cv2.imread(example.xyz[0],cv2.IMREAD_COLOR).astype(dtype='float32')/len(self.examples_train)
			#mean/=len(self.images_train)
			mean_uint8 = mean.astype(dtype='uint8')
			cv2.imwrite(self.outputfolder+'/'+self.mean_xyz_filename, mean_uint8)
		return self.outputfolder+'/'+self.mean_xyz_filename

	def analyse_isomaps(self):
		print ('analysing isomaps...')
		for example in self.examples_all:
			img = cv2.imread(example.images[0], cv2.IMREAD_UNCHANGED)
			#blurryness_map = cv2.Laplacian(img, cv2.CV_64F)
			#blurryness_map[np.logical_or(blurryness_map<-700, blurryness_map>700)]=0 #try to filter out the edges
			#example.blurryness = blurryness_map.var()
			example.blurryness = _get_gradient_magnitude(img)

			example.coverage = _calc_isomap_coverage(img)

	def remove_bad_isomaps(self, threshold=15):
		for example in self.examples_all[:]:
			if example.coverage*example.coverage*example.blurryness < threshold:
				self.examples_all.remove(example)

#	def make_sure_nothings_empty(self):
#		for example in self.examples_train[:]:
#			for img in example.images:
#				if os.path.getsize(img)<=1:
#					print('found some empty image', img)
#					self.examples_train.remove(example)

	def set_all_as_train(self):
		self.examples_train = self.examples_all

		self.write_labeled_image_list(self.examples_train, self.outputfolder+'/'+self.lil_train_filename)

	def group_to_more_than_1_image(self):
		if not NUMBER_IMAGES >1 :
			return None
		example_id_dict={}
		for example in self.examples_all:
			if not example.label in example_id_dict:
				example_id_dict[example.label]=[example]
			else:
				example_id_dict[example.label].append(example)
		new_examples_list = []
		seed = random.seed(448)
		for key in example_id_dict.keys():
			label = key
			images = [example.images[0] for example in example_id_dict[key]]
			if NUMBER_IMAGES <= len(images):
				for i in range(4*len(images)):
					example = Training_example()
					example.label = key
					example.images = random.sample(images, NUMBER_IMAGES)
					new_examples_list.append(example)
		self.examples_all = new_examples_list


	def show_isomaps(self):
		for example in self.examples_train:
			print (example)
			img = cv2.imread(example.images[0],cv2.IMREAD_COLOR)
			cv2.imshow('img', img)
			cv2.waitKey()
		cv2.destroyAllWindows()		

	def get_training_image_and_label_lists(self):
		images = []
		labels = []
		for example in self.examples_train:
			images.append(example.images[0])
			labels.append(example.label)
		return images, labels

	def get_eval_image_and_label_lists(self):
		images = []
		labels = []
		for example in self.examples_eval:
			images.append(example.images[0])
			labels.append(example.label)
		return images, labels

	def get_training_multi_image_and_label_lists(self):
		images = []
		labels = []
		for example in self.examples_train:
			images.append(example.images)
			labels.append(example.label)
		return images, labels

	def get_eval_multi_image_and_label_lists(self):
		images = []
		labels = []
		for example in self.examples_eval:
			images.append(example.images)
			labels.append(example.label)
		return images, labels


	def get_training_xyz_and_label_lists(self):
		images = []
		labels = []
		for example in self.examples_train:
			images.append(example.xyz[0])
			labels.append(example.label)
		return images, labels

	def get_eval_xyz_and_label_lists(self):
		images = []
		labels = []
		for example in self.examples_eval:
			images.append(example.xyz[0])
			labels.append(example.label)
		return images, labels

	def get_training_image_xyz_and_label_lists(self):
		images = []
		xyz    = []
		labels = []
		for example in self.examples_train:
			images.append(example.images[0])
			xyz.append(example.xyz[0])
			labels.append(example.label)
		return images, xyz, labels

	def get_eval_image_xyz_and_label_lists(self):
		images = []
		xyz    = []
		labels = []
		for example in self.examples_eval:
			images.append(example.images[0])
			xyz.append(example.xyz[0])
			labels.append(example.label)
		return images, xyz, labels

	def get_training_image_alphas_and_label_lists(self):
		images = []
		labels = []
		alphas = []
		for example in self.examples_train:
			images.append(example.images[0])
			alphas.append(example.alphas[0])
			labels.append(example.label)
		return images, alphas, labels

	def get_eval_image_alphas_and_label_lists(self):
		images = []
		labels = []
		alphas = []
		for example in self.examples_eval:
			images.append(example.images[0])
			alphas.append(example.alphas[0])
			labels.append(example.label)
		return images, alphas, labels



class Aggregator(DB_loader):

	def __init__(self, *args):
		DB_loader.__init__(self, outputfolder=args[0].outputfolder, db_base=None)
		self.id_class_mapping_filename = 'total_' + ID_CLASS_MAPPING_POSTFIX
		self.labeled_image_list_filename = 'total_' + LIL_ALL_POSTFIX
		self.lil_train_filename = 'total_' + LIL_TRAIN_POSTFIX
		self.lil_eval_filename = 'total_' + LIL_EVAL_POSTFIX
		self.mean_image_filename = 'total_' + MEAN_IMAGE_POSTFIX
		self.mean_xyz_filename = 'total_' + MEAN_XYZ_POSTFIX

		if os.path.exists(self.outputfolder+'/'+self.id_class_mapping_filename) and os.path.exists(self.outputfolder+'/'+self.lil_train_filename) and os.path.exists(self.outputfolder+'/'+self.lil_eval_filename):
			print ('Reading total train and eval set...')
			self.read_id_class_mapping(self.outputfolder+'/'+self.id_class_mapping_filename)
			self.examples_train = self.read_labeled_image_list(self.outputfolder+'/'+self.lil_train_filename)
			self.examples_eval = self.read_labeled_image_list(self.outputfolder+'/'+self.lil_eval_filename)

		else:
			print ('Aggregating dbs...')
			for arg in args:
				if not isinstance(arg, PaSC_video_loader):
					for item in arg.id_class_mapping.items():
						arg.id_class_mapping.update({item[0]: item[1]+self.number_ids})
						#print (item[1])
						#item[1]+= self.number_ids
					self.id_class_mapping.update(arg.id_class_mapping)
					self.number_ids += arg.number_ids
		
				self.examples_all.extend(arg.examples_all)
		
				self.examples_train.extend(arg.examples_train)
		
				self.examples_eval.extend(arg.examples_eval)

			random.seed(404)
			random.shuffle(self.examples_train)
		
			self.write_id_class_mapping(self.outputfolder+'/'+self.id_class_mapping_filename)
			self.write_labeled_image_list(self.examples_all,   self.outputfolder+'/'+self.labeled_image_list_filename)
			self.write_labeled_image_list(self.examples_train, self.outputfolder+'/'+self.lil_train_filename)
			self.write_labeled_image_list(self.examples_eval,  self.outputfolder+'/'+self.lil_eval_filename)



class lazy_dummy(DB_loader):
	def __init__(self, folder):
		DB_loader.__init__(self, outputfolder=folder, db_base=None)
		self.id_class_mapping_filename = 'total_' + ID_CLASS_MAPPING_POSTFIX
		self.labeled_image_list_filename = 'total_' + LIL_ALL_POSTFIX
		self.lil_train_filename = 'total_' + LIL_TRAIN_POSTFIX
		self.lil_eval_filename = 'total_' + LIL_EVAL_POSTFIX
		self.mean_image_filename = 'total_' + MEAN_IMAGE_POSTFIX
		self.mean_xyz_filename = 'total_' + MEAN_XYZ_POSTFIX

		self.read_id_class_mapping(self.outputfolder+'/'+self.id_class_mapping_filename)
		self.examples_train = self.read_labeled_image_list(self.outputfolder+'/'+self.lil_train_filename)
		self.examples_eval  = self.read_labeled_image_list(self.outputfolder+'/'+self.lil_eval_filename)


class PaSC_video_loader(DB_loader):


	def __init__(self, outputfolder, db_base):
		DB_loader.__init__(self, outputfolder, db_base)
		self.id_class_mapping_filename = 'pasc_' + ID_CLASS_MAPPING_POSTFIX
		self.labeled_image_list_filename = 'pasc_video_' + LIL_ALL_POSTFIX
		self.lil_train_filename = 'pasc_video_' + LIL_TRAIN_POSTFIX
		self.lil_eval_filename = 'pasc_video_' + LIL_EVAL_POSTFIX
		self.mean_image_filename = 'pasc_video_' + MEAN_IMAGE_POSTFIX
		self.mean_xyz_filename = 'pasc_video_' + MEAN_XYZ_POSTFIX
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
		self.examples_all = []
		for folder in folders:
			id_ = self.id_class_mapping[os.path.basename(folder)[:5]]
			folder_images = glob.glob(folder+IMAGE_FILE_ENDING)
			if NUMBER_ALPHAS>0:
				alphas, _ = oal.read_fitting_log(folder+'/fitting.log')
			for folder_image in folder_images:
				example = Training_example()
				example.images.append(folder_image)
				example.xyz.append(folder_image.replace('isomap','xyzmap'))
				example.label = id_
				if NUMBER_ALPHAS>0:
					example.alphas.append(alphas)
				self.examples_all.append(example)

	def split_train_eval(self, train_proportion=0.8):
		#random.seed(404)
		#random.shuffle(self.images_all)
		#random.seed(404)
		#random.shuffle(self.labels_all)

		num_examples_train = int(len(self.examples_all)*train_proportion)
		

		self.examples_train = self.examples_all[:num_examples_train]
		self.examples_eval  = self.examples_all[num_examples_train:]

		self.write_labeled_image_list(self.examples_train, self.outputfolder+'/'+self.lil_train_filename)
		self.write_labeled_image_list(self.examples_eval,  self.outputfolder+'/'+self.lil_eval_filename)


class PaSC_still_loader(DB_loader):


	def __init__(self, outputfolder, db_base):
		DB_loader.__init__(self, outputfolder, db_base)
		self.id_class_mapping_filename = 'pasc_' + ID_CLASS_MAPPING_POSTFIX
		self.labeled_image_list_filename = 'pasc_still_' + LIL_ALL_POSTFIX
		self.lil_train_filename = 'pasc_still_' + LIL_TRAIN_POSTFIX
		self.lil_eval_filename = 'pasc_still_' + LIL_EVAL_POSTFIX
		self.mean_image_filename = 'pasc_still_' + MEAN_IMAGE_POSTFIX
		self.mean_xyz_filename = 'pasc_still_' + MEAN_XYZ_POSTFIX
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
		self.examples_all = []
		for folder in folders:
			id_ = self.id_class_mapping[os.path.basename(folder)[:5]]
			folder_images = glob.glob(folder+IMAGE_FILE_ENDING)
			if NUMBER_ALPHAS>0:
				alphas, _ = oal.read_fitting_log(folder+'/fitting.log')
			for folder_image in folder_images:
				example = Training_example()
				example.images.append(folder_image)
				example.xyz.append(folder_image.replace('isomap','xyzmap'))
				example.label = id_
				if NUMBER_ALPHAS>0:
					example.alphas.append(alphas)
				self.examples_all.append(example)


class CASIA_webface_loader(DB_loader):


	def __init__(self, outputfolder, db_base):
		DB_loader.__init__(self, outputfolder, db_base)
		self.id_class_mapping_filename = 'casia_' + ID_CLASS_MAPPING_POSTFIX
		self.labeled_image_list_filename = 'casia_' + LIL_ALL_POSTFIX
		self.lil_train_filename = 'casia_' + LIL_TRAIN_POSTFIX
		self.lil_eval_filename = 'casia_' + LIL_EVAL_POSTFIX
		self.mean_image_filename = 'casia_' + MEAN_IMAGE_POSTFIX
		self.mean_xyz_filename = 'casia_' + MEAN_XYZ_POSTFIX
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
		self.examples_all = []
		for folder in folders:
			id_ = self.id_class_mapping[os.path.basename(folder)[:7]]
			folder_images = glob.glob(folder+IMAGE_FILE_ENDING)
			if NUMBER_ALPHAS>0:
				alphas, _ = oal.read_fitting_log(folder+'/fitting.log')
			for folder_image in folder_images:
				example = Training_example()
				example.images.append(folder_image)
				example.xyz.append(folder_image.replace('isomap','xyzmap'))
				example.label = id_
				if NUMBER_ALPHAS>0:
					example.alphas.append(alphas)
				self.examples_all.append(example)










