#!/usr/bin/env python3.5
import sys, os
import numpy as np
import cv2
import glob

ISOMAP_SIZE = 256


isomap_paths = glob.glob('/user/HS204/m09113/my_project_folder/IJB_A/multi_iter75_reg30/540/*isomap.png')
isomap_merged = None


def merge( isomaps):
	merge = np.zeros([ISOMAP_SIZE, ISOMAP_SIZE, 4], dtype='uint32')

	threshould_angle = 90 #between 0 (facing camera) and 90 (edge to camera)
	threshould_alpha = (-255.0 / 90.0) * threshould_angle + 255.0

	confidence_maps =  np.zeros([ISOMAP_SIZE, ISOMAP_SIZE, len(isomaps)], dtype='uint32')
	for i in range(len(isomaps)):
		confidence_maps[:,:,i] = isomaps[i][:,:,3].astype(dtype='uint32')
		rule_mask = confidence_maps[:,:,i] < threshould_alpha
		#confidence_maps[:,:,i] = np.exp(confidence_maps[:,:,i]/50)
		confidence_maps[rule_mask, i] = 0
	#confidence_sum = confidence1+confidence2

	confidence_sum = np.sum(confidence_maps,2)
	merge[:,:,3] = confidence_sum
	conf_over_0 = merge[:,:,3] > 0

	#merge[conf_over_0,:3] = (isomap1[conf_over_0,:3]*confidence1[conf_over_0,None]+isomap2[conf_over_0,:3]*confidence2[conf_over_0,None])/(confidence1[conf_over_0,None]+confidence2[conf_over_0,None])
	for i in range(len(isomaps)):
		merge[:,:,:3] += (isomaps[i][:,:,:3]/confidence_sum[:,:,None]*confidence_maps[:,:,i,None]).astype(dtype='uint32')

	#merge[:,:,:3] = isomaps[0][:,:,:3]/confidence_sum[:,:,None]*confidence_maps[:,:,0,None]+isomaps[1][:,:,:3]/confidence_sum[:,:,None]*confidence_maps[:,:,1,None]#confidence2[:,:,None]
	
	conf_over_255 = merge[:,:,3] > 255
	merge[conf_over_255,3] = 255

	return merge.astype(dtype="uint8")

def merge_sm_with_tf(isomap_lists, confidence_lists, output_list):
	import tensorflow as tf
	import cnn_tf_graphs
	from shutil import copyfile
	
	#zipped_input = zip(isomap_lists, confidence_lists, output_list)
	#zipped_input.sort(key=lambda x: len(x[0]))
	#isomap_lists, confidence_lists, output_list = zip(*zipped_input)

	sorted_idx_list = sorted(range(len(isomap_lists)), key=lambda x: len(isomap_lists[x]))
	#print (sorted_idx_list)
	isomap_lists = [isomap_lists[i] for i in sorted_idx_list]
	confidence_lists = [confidence_lists[i] for i in sorted_idx_list]
	output_list = [output_list[i] for i in sorted_idx_list]
	
	#print ('HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
	#for i in range(len(isomap_lists)):
	#	print (isomap_lists[i])
	#	print (confidence_lists[i])
	#	print (output_list[i])

	#isomap_lists.sort(key=len)
	merge_legth = -1
	sess = None
	for j, isomap_list in enumerate(isomap_lists):
		
		with tf.Graph().as_default():
			if len(isomap_list) == 0:
				continue
			elif len(isomap_list) ==1:
				copyfile(isomap_list[0],output_list[j])
			else:
				if len(isomap_list) != merge_legth:
					if sess:
						sess.close()
					placeholders = []
					outpath = tf.placeholder(tf.string)
					for i in range(len(isomap_list)):
						colour = tf.placeholder(tf.float32, shape=(1, ISOMAP_SIZE, ISOMAP_SIZE, 3))
						conf = tf.placeholder(tf.float32, shape=(1, ISOMAP_SIZE, ISOMAP_SIZE, 1))
						placeholders.append([colour, conf])

					merged = tf.squeeze(cnn_tf_graphs.merge_isomaps_softmax(placeholders))
					merged_uint8 = tf.cast(merged, tf.uint8)
					encoded = tf.image.encode_png(merged_uint8)
					write_file_op = tf.write_file(outpath, encoded)

					merge_legth = len(isomap_list)
					sess = tf.Session()
				print ('merging',merge_legth,'images (max',len(isomap_lists[-1]),') idx',j,'of',len(isomap_lists))

				feed_dict = {}
				for i in range(len(isomap_list)):
					feed_dict[placeholders[i][0]] = np.expand_dims(cv2.imread(isomap_list[i], cv2.IMREAD_UNCHANGED)[:,:,:3].astype(np.float32)[:,:,::-1], axis=0)
					feed_dict[placeholders[i][1]] = np.expand_dims(np.load(confidence_lists[j][i]).astype(np.float32), axis=0)
				feed_dict[outpath] = output_list[j]
				sess.run(write_file_op, feed_dict=feed_dict)


def write_cnn_confidences(saved_model_path, images, output_paths):
	import tensorflow as tf
	import tf_utils
	import cnn_tf_graphs

	with tf.device('/gpu:0'):
		with tf.Graph().as_default():
			
			image_path_tensor = tf.placeholder(tf.string)
			image_tf = tf_utils.single_input_image(image_path_tensor, os.path.dirname(saved_model_path)+'/../db_input/total_mean.png' ,png_with_alpha=True, image_size=256)
			image_tf = tf.expand_dims(image_tf,0)

			# Build a Graph that computes the logits predictions from the inference model.
			with tf.variable_scope("confidence_estimation") as scope:
				confidence_map = cnn_tf_graphs.confidence_cnn13(image_tf, input_size=256)

			saver = tf.train.Saver()

			config = tf.ConfigProto( allow_soft_placement=False, log_device_placement=False)
			config.gpu_options.allow_growth = True

			with tf.Session(config=config) as sess:
				print('restore model')
				saver.restore(sess, saved_model_path)
				print ('restoring done')

				#print('we have',db_loader.num_examples_eval, 'images to evaluate')
				for idx, image_path in enumerate(images):
					if idx%1000==0:
						print (idx,'of',len(images))
					
					confidence_pic = sess.run(confidence_map, feed_dict={image_path_tensor: image_path})
					confidence_output_path = output_paths[idx] 
					np.save(confidence_output_path, confidence_pic[0,...])



def calc_isomap_coverage(isomap):
	visible = isomap[:,:,3]>0 
	pixels = isomap.shape[0]*isomap.shape[1]
	return np.sum(visible)/pixels

def isomap_playground():
	isomaps =[]
	for i in range(len(isomap_paths)):
		isomaps.append(cv2.imread(isomap_paths[i], cv2.IMREAD_UNCHANGED))
	
	old_isomap_merged = np.zeros([ISOMAP_SIZE, ISOMAP_SIZE, 4], dtype='uint8')
	
	all_isomaps_merged = merge(isomaps)
	show_isomap('all_isomaps_merged', all_isomaps_merged)
	#cv2.waitKey()
	#cv2.destroyAllWindows()
	#exit()
	
	for i in range(len(isomaps)):
		new_isomap_merged = merge([old_isomap_merged, isomaps[i]])
		#blurryness = cv2.Laplacian(isomaps[i], cv2.CV_64F).var()
		blurryness_map = cv2.Laplacian(isomaps[i], cv2.CV_64F)
		blurryness_map[np.logical_or(blurryness_map<-700, blurryness_map>700)]=0 #try to filter out the edges
		blurryness = blurryness_map.var()
		#show_isomap('laplac',cv2.Laplacian(isomaps[i], cv2.CV_8U))
		#print ('max', np.max(cv2.Laplacian(isomaps[i], cv2.CV_64F)), 'min', np.min(cv2.Laplacian(isomaps[i], cv2.CV_64F)))
		coverage = calc_isomap_coverage(isomaps[i])
		print(isomap_paths[i]," isomap coverage:",coverage,"blur detection:",blurryness, "overall score", coverage*coverage*blurryness)
		show_isomap('new isomap', isomaps[i])
		show_isomap('merge', new_isomap_merged)
		cv2.waitKey()
	
		old_isomap_merged = new_isomap_merged
	

	#cv2.imwrite('/user/HS204/m09113/Desktop/merge_test.png', isomap_merged)

	#cv2.waitKey()
	#cv2.destroyAllWindows()

def show_isomap(window, isomap):
	#isomap_copy = isomap.copy()
	background = np.zeros([ISOMAP_SIZE, ISOMAP_SIZE, 4], dtype='uint8')
	background[:,:,3]=10
	mask = np.array([[int(x/8) %2==int(y/8) %2 for x in range(isomap.shape[0])] for y in range(isomap.shape[1])])
	#mask = np.array([[int(x/8) %2==0 for x in range(isomap.shape[0])] for y in range(isomap.shape[1])])
	background[mask,:3]=[200,200,200]
	mask = np.invert(mask)
	background[mask,:3]=[150,150,150]

	cv2.imshow(window, merge([background,isomap]))

def merge_isomaps_pg():
	isomap_paths = ['/user/HS204/m09113/my_project_folder/Boris/new_isomaps/image-00058.isomap.png', '/user/HS204/m09113/my_project_folder/Boris/new_isomaps/image-00456.isomap.png']
	isomaps =[]
	for i in range(len(isomap_paths)):
		isomaps.append(cv2.imread(isomap_paths[i], cv2.IMREAD_UNCHANGED))
	isomap_merged = merge(isomaps)
	cv2.imwrite('/user/HS204/m09113/my_project_folder/Boris/new_isomaps/merged.png', isomap_merged[:,:,:3])

def pseudocolor(val, minval, maxval):
	# from here: http://stackoverflow.com/questions/10901085/range-values-to-pseudocolor
	import colorsys

	# convert val in range minval..maxval to the range 0..120 degrees which
	# correspond to the colors red..green in the HSV colorspace
	h = (float(val-minval) / (maxval-minval)) * 240
	#h=240-h
	# convert hsv color (h,1,1) to its rgb equivalent
	# note: the hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
	r, g, b = colorsys.hsv_to_rgb(h/360, 1., 1.)
	r*=255
	g*=255
	b*=255
	return r, g, b

def color_isomap_alpha(isomap):

	colored = np.zeros([ISOMAP_SIZE, ISOMAP_SIZE, 3], dtype='uint8')
	for x in range(colored.shape[0]):
		for y in range(colored.shape[1]):
			#print (isomap[x,y,3])
			#print (pseudocolor(isomap[x,y,3],0, 255)*255)
			colored[x,y,:] = pseudocolor(isomap[x,y,3],0, 255)
			#print (colored[x,y,:])
	return colored

def color_alpha_only(confidence, minval = None, maxval = None):
	#print (confidence.shape)
	if not maxval:
		maxval = np.max(confidence)
	if not minval:
		minval = 0
	colored = np.zeros([confidence.shape[0], confidence.shape[1], 3], dtype='uint8')
	for x in range(colored.shape[0]):
		for y in range(colored.shape[1]):
			#print (isomap[x,y,3])
			#print (pseudocolor(isomap[x,y,3],0, 255)*255)
			colored[x,y,:] = pseudocolor(confidence[x,y,0],minval, maxval)
			#print (colored[x,y,:])
	return colored


if __name__ == "__main__":
	
	#isomap_playground()
	#show_isomap(cv2.imread(isomap_paths[0], cv2.IMREAD_UNCHANGED))
	merge_isomaps_pg()
	#cv2.imwrite('/user/HS204/m09113/Desktop/Boris/new_isomaps/image-00456.isomap_colored.png',color_isomap_alpha(cv2.imread('/user/HS204/m09113/Desktop/Boris/new_isomaps/image-00456.isomap.png', cv2.IMREAD_UNCHANGED)))
	#cv2.imwrite('/user/HS204/m09113/my_project_folder/Boris/new_isomaps/image-00058_conf01_colored.png',color_alpha_only(np.load('/user/HS204/m09113/my_project_folder/Boris/new_isomaps/image-00058_cnn_conf01.npy')))
	

	#cv2.waitKey()
	#cv2.destroyAllWindows()	

