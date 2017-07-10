#!/usr/bin/env python3.5


import os
import glob
from datetime import datetime
import time
import tensorflow as tf


NUM_CHANNELS = 3



def read_images_from_disk(input_queue):
	# copied from http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels
	"""Consumes a single filename and label as a ' '-delimited string.
	Args:
	  filename_and_label_tensor: A scalar string tensor.
	Returns:
	  Two tensors: the decoded image, and the string label.
	"""
	#label = input_queue[1]
	label = input_queue[-1]
	alphas = input_queue[1]

	file_contents = tf.read_file(input_queue[0])
	example = tf.image.decode_image(file_contents, channels=NUM_CHANNELS)
	return example, alphas, label

def subtract_mean(image_tensor, mean_image_path, image_size=512):

	mean_image = tf.convert_to_tensor(mean_image_path, dtype=tf.string)

	image_tensor.set_shape([image_size, image_size, NUM_CHANNELS])
	image = tf.cast(image_tensor, tf.float32)

	#subtract mean image
	mean_file_contents = tf.read_file(mean_image)
	mean_uint8 = tf.image.decode_image(mean_file_contents, channels=NUM_CHANNELS)
	mean_uint8.set_shape([image_size, image_size, NUM_CHANNELS])
	image_mean_free = tf.subtract(image, tf.cast(mean_uint8, tf.float32))

	return image_mean_free

def subtract_mean_multi(image_tensors, mean_image_path, channels=NUM_CHANNELS, image_size=512):

	mean_image = tf.convert_to_tensor(mean_image_path, dtype=tf.string)
	mean_file_contents = tf.read_file(mean_image)
	mean_uint8 = tf.image.decode_png(mean_file_contents, channels=channels)
	mean_uint8.set_shape([image_size, image_size, channels])


	images_mean_free = []
	for image_tensor in image_tensors:
		image_tensor.set_shape([image_size, image_size, channels])
		image = tf.cast(image_tensor, tf.float32)

		#subtract mean image
		image_mean_free = tf.subtract(image, tf.cast(mean_uint8, tf.float32))
		images_mean_free.append(image_mean_free)

	return images_mean_free



def inputs(image_list, label_list, batch_size, mean_image_path, image_size=512):
	"""Construct input for CIFAR evaluation using the Reader ops.
	Args:
	  eval_data: bool, indicating if one should use the train or eval data set.
	  data_dir: Path to the CIFAR-10 data directory.
	  batch_size: Number of images per batch.
	Returns:
	  images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	  labels: Labels. 1D tensor of [batch_size] size.
	"""	  

	images = tf.convert_to_tensor(image_list, dtype=tf.string)
	labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

	# Makes an input queue
	input_queue = tf.train.slice_input_producer([images, labels], shuffle=True, capacity=10*batch_size)

	uint8image, _, label = read_images_from_disk(input_queue)

	image_mean_free = subtract_mean(uint8image, mean_image_path, image_size=image_size)

	# Optional Preprocessing or Data Augmentation
	# tf.image implements most of the standard image augmentation
	#image = preprocess_image(image)
	#label = preprocess_label(label)

	# Generate a batch of images and labels by building up a queue of examples.
	num_preprocess_threads = 10
	return tf.train.batch([image_mean_free, label], #tf.train.shuffle_batch(
								   batch_size=batch_size,
								   capacity=10*batch_size,
								   num_threads=num_preprocess_threads)


def inputs_multi(images_list, label_list, batch_size, mean_image_path, png_with_alpha=False, image_size=512):
	"""Construct input for CIFAR evaluation using the Reader ops.
	Args:
	  eval_data: bool, indicating if one should use the train or eval data set.
	  data_dir: Path to the CIFAR-10 data directory.
	  batch_size: Number of images per batch.
	Returns:
	  images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	  labels: Labels. 1D tensor of [batch_size] size.
	"""	  
	images =[]
	for i in range(len(images_list[0])):
		#print ([x[i] for x in images_list])
		images.append( tf.convert_to_tensor([x[i] for x in images_list], dtype=tf.string) )
	labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

	# Makes an input queue
	images.append(labels)
	input_queue = tf.train.slice_input_producer(images, shuffle=True, capacity=10*batch_size)

	if png_with_alpha:
		img_channels=4
	else:
		img_channels=3

	uint8images = []
	for i in range(len(images_list[0])):
		if png_with_alpha:
			uint8image = tf.image.decode_png(tf.read_file(input_queue[i]), channels=img_channels)
		else:
			uint8image = tf.image.decode_image(tf.read_file(input_queue[i]), channels=NUM_CHANNELS)
		uint8images.append(uint8image)

	images_mean_free = subtract_mean_multi(uint8images, mean_image_path, channels=img_channels, image_size=image_size)

	# Optional Preprocessing or Data Augmentation
	# tf.image implements most of the standard image augmentation
	#image = preprocess_image(image)
	#label = preprocess_label(label)

	# Generate a batch of images and labels by building up a queue of examples.
	num_preprocess_threads = 10
	images_mean_free.append(input_queue[-1])
	return tf.train.batch(images_mean_free, #tf.train.shuffle_batch(
								   batch_size=batch_size,
								   capacity=10*batch_size,
								   num_threads=num_preprocess_threads)	


def inputs_stack_image_and_xyz(image_list, xyz_list, label_list, batch_size, mean_image_path, mean_xyz_path):
	"""Construct input for CIFAR evaluation using the Reader ops.
	Args:
	  eval_data: bool, indicating if one should use the train or eval data set.
	  data_dir: Path to the CIFAR-10 data directory.
	  batch_size: Number of images per batch.
	Returns:
	  images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	  labels: Labels. 1D tensor of [batch_size] size.
	"""	  

	images = tf.convert_to_tensor(image_list, dtype=tf.string)
	xyz = tf.convert_to_tensor(image_list, dtype=tf.string)
	labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

	# Makes an input queue
	input_queue = tf.train.slice_input_producer([images, xyz, labels], shuffle=True, capacity=10*batch_size)

	#uint8image, _, label = read_images_from_disk(input_queue)
	uint8image = tf.image.decode_image(tf.read_file(input_queue[0]), channels=NUM_CHANNELS)
	uint8xyz   = tf.image.decode_image(tf.read_file(input_queue[1]), channels=NUM_CHANNELS)

	image_mean_free = subtract_mean(uint8image, mean_image_path)
	xyz_mean_free = subtract_mean(uint8xyz, mean_xyz_path)

	img_xyz_stack = tf.concat([image_mean_free, xyz_mean_free], axis=2)

	# Optional Preprocessing or Data Augmentation
	# tf.image implements most of the standard image augmentation
	#image = preprocess_image(image)
	#label = preprocess_label(label)

	# Generate a batch of images and labels by building up a queue of examples.
	num_preprocess_threads = 10
	return tf.train.batch([img_xyz_stack, input_queue[-1]], #tf.train.shuffle_batch(
								   batch_size=batch_size,
								   capacity=10*batch_size,
								   num_threads=num_preprocess_threads)

def inputs_with_alphas(image_list, alphas_list, label_list, batch_size, mean_image_path):
	"""Construct input for CIFAR evaluation using the Reader ops.
	Args:
	  eval_data: bool, indicating if one should use the train or eval data set.
	  data_dir: Path to the CIFAR-10 data directory.
	  batch_size: Number of images per batch.
	Returns:
	  images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	  labels: Labels. 1D tensor of [batch_size] size.
	"""	  

	images = tf.convert_to_tensor(image_list, dtype=tf.string)
	labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
	alphas = tf.convert_to_tensor(alphas_list, dtype=tf.float32)
	mean_image = tf.convert_to_tensor(mean_image_path, dtype=tf.string)

	# Makes an input queue
	input_queue = tf.train.slice_input_producer([images, alphas, labels],
												shuffle=True)

	uint8image, alpha, label = read_images_from_disk(input_queue)

	image_mean_free = subtract_mean(uint8image, mean_image)

	# Optional Preprocessing or Data Augmentation
	# tf.image implements most of the standard image augmentation
	#image = preprocess_image(image)
	#label = preprocess_label(label)

	# Generate a batch of images and labels by building up a queue of examples.
	num_preprocess_threads = 4
	return tf.train.batch([image_mean_free, alpha, label], #tf.train.shuffle_batch(
								   batch_size=batch_size,
								   num_threads=num_preprocess_threads)


def single_input_image(image_str, mean_image_path, png_with_alpha=False, image_size=512):

	mean_image_str = tf.convert_to_tensor(mean_image_path, dtype=tf.string)

	file_contents = tf.read_file(image_str)
	if png_with_alpha:
		uint8image = tf.image.decode_png(file_contents, channels=4)
		uint8image.set_shape([image_size, image_size, 4])
	else:
		uint8image = tf.image.decode_image(file_contents, channels=NUM_CHANNELS)
		uint8image.set_shape([image_size, image_size, NUM_CHANNELS])
	image = tf.cast(uint8image, tf.float32)

	#subtract mean image
	mean_file_contents = tf.read_file(mean_image_str)
	if png_with_alpha:
		mean_uint8 = tf.image.decode_png(mean_file_contents, channels=4)
		mean_uint8.set_shape([image_size, image_size, 4])
	else:
		mean_uint8 = tf.image.decode_image(mean_file_contents, channels=NUM_CHANNELS)
		mean_uint8.set_shape([image_size, image_size, NUM_CHANNELS])
	image_mean_free = tf.subtract(image, tf.cast(mean_uint8, tf.float32))

	return image_mean_free



